"""逐帧线性混合蒙皮（LBS）：把 17 关节四元数驱动到 Mixamo rigged 模型。

蒙皮公式（architecture 约定）：
    G_j(t) = G_parent(t) × LocalRest_j × ΔQ_j(t)
    skinMatrix_j(t) = G_j(t) × inverseBindMatrix_j
    v' = Σ_k w_k · skinMatrix_{j_k}(t) · v

重定向换基（关键）：retargeter 产出的四元数描述「入骨方向」（parent→joint），
且目标方向在 MediaPipe 像素系（y 向下），与本模型世界系（y 向上）不一致。
直接把 ΔQ 当局部旋转会得到错误姿态。因此 :meth:`SkinAnimator.skin_frame` 会：
1. 由 ΔQ 反解各「入骨」目标方向，并翻转 y 换算到 y-up 世界系；
2. 对每个受驱 Mixamo 骨骼，按其「出骨」（our-hierarchy 子关节方向）重解局部旋转，
   并对父链当前动画旋转做补偿（ΔQ_local = quat_between(offset_local, R_j⁻¹·R_parent⁻¹·T)）。

未映射骨骼（手指/脚趾/锁骨/Spine2/HeadTop_End 等）保持 rest pose。
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as _R

from app.pipeline.gltf_loader import GLBModel, quat_wxyz_to_matrix
from app.pipeline.retargeter import MODEL_DEFINITION, quat_between, quat_rotate

# 17 关节 → Mixamo 骨骼名（与团队下发的映射表一致）
JOINT_TO_MIXAMO: dict[str, str] = {
    "Hips": "mixamorig:Hips",
    "Spine": "mixamorig:Spine",
    "Chest": "mixamorig:Spine1",
    "Neck": "mixamorig:Neck",
    "Head": "mixamorig:Head",
    "L_Shoulder": "mixamorig:LeftArm",
    "L_Elbow": "mixamorig:LeftForeArm",
    "L_Wrist": "mixamorig:LeftHand",
    "R_Shoulder": "mixamorig:RightArm",
    "R_Elbow": "mixamorig:RightForeArm",
    "R_Wrist": "mixamorig:RightHand",
    "L_Hip": "mixamorig:LeftUpLeg",
    "L_Knee": "mixamorig:LeftLeg",
    "L_Ankle": "mixamorig:LeftFoot",
    "R_Hip": "mixamorig:RightUpLeg",
    "R_Knee": "mixamorig:RightLeg",
    "R_Ankle": "mixamorig:RightFoot",
}

# 受驱 Mixamo 节点 → (出骨末端 Mixamo 节点, 目标方向来源 our-joint)
# 出骨目标方向 = our-hierarchy 子关节的「入骨」目标方向（同一根骨头）
BONE_SOLVE: dict[str, tuple[str, str]] = {
    "mixamorig:Spine": ("mixamorig:Spine1", "Chest"),
    "mixamorig:Spine1": ("mixamorig:Neck", "Neck"),
    "mixamorig:Neck": ("mixamorig:Head", "Head"),
    "mixamorig:LeftArm": ("mixamorig:LeftForeArm", "L_Elbow"),
    "mixamorig:LeftForeArm": ("mixamorig:LeftHand", "L_Wrist"),
    "mixamorig:LeftUpLeg": ("mixamorig:LeftLeg", "L_Knee"),
    "mixamorig:LeftLeg": ("mixamorig:LeftFoot", "L_Ankle"),
    "mixamorig:RightArm": ("mixamorig:RightForeArm", "R_Elbow"),
    "mixamorig:RightForeArm": ("mixamorig:RightHand", "R_Wrist"),
    "mixamorig:RightUpLeg": ("mixamorig:RightLeg", "R_Knee"),
    "mixamorig:RightLeg": ("mixamorig:RightFoot", "R_Ankle"),
}


# 受驱 Mixamo 节点 → (our grandparent, our parent, our child) 三关节（三关节法线法）
TRI_JOINTS: dict[str, tuple[str, str, str]] = {
    "mixamorig:Spine": ("Hips", "Spine", "Chest"),
    "mixamorig:Spine1": ("Spine", "Chest", "Neck"),
    "mixamorig:Neck": ("Chest", "Neck", "Head"),
    "mixamorig:LeftArm": ("Chest", "L_Shoulder", "L_Elbow"),
    "mixamorig:LeftForeArm": ("L_Shoulder", "L_Elbow", "L_Wrist"),
    "mixamorig:RightArm": ("Chest", "R_Shoulder", "R_Elbow"),
    "mixamorig:RightForeArm": ("R_Shoulder", "R_Elbow", "R_Wrist"),
    "mixamorig:LeftUpLeg": ("Hips", "L_Hip", "L_Knee"),
    "mixamorig:LeftLeg": ("L_Hip", "L_Knee", "L_Ankle"),
    "mixamorig:RightUpLeg": ("Hips", "R_Hip", "R_Knee"),
    "mixamorig:RightLeg": ("R_Hip", "R_Knee", "R_Ankle"),
}
# 末端关节（方向向量法）：Mixamo 节点 → (our parent, our joint)
END_JOINTS: dict[str, tuple[str, str]] = {
    "mixamorig:Head": ("Neck", "Head"),
    "mixamorig:LeftHand": ("L_Elbow", "L_Wrist"),
    "mixamorig:RightHand": ("R_Elbow", "R_Wrist"),
    "mixamorig:LeftFoot": ("L_Knee", "L_Ankle"),
    "mixamorig:RightFoot": ("R_Knee", "R_Ankle"),
}


def _norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v if n < 1e-9 else v / n


def _perp(a: np.ndarray) -> np.ndarray:
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(a, ref))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return _norm(np.cross(a, ref))


def _frame(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """用入骨 a、出骨 b 构造右手正交基 [a, n, a×n]（n 为平面法线，编码 twist）。"""
    n = np.cross(a, b)
    if float(np.linalg.norm(n)) < 1e-6:
        n = _perp(a)
    else:
        n = _norm(n)
    t = np.cross(a, n)
    return np.column_stack([a, n, t])


def _pure_rot(M: np.ndarray) -> np.ndarray:
    """把 3x3 矩阵投影到最近的旋转矩阵（SVD 去掉缩放，保证 det=+1）。"""
    u, _, vt = np.linalg.svd(M.astype(np.float64))
    R = u @ vt
    if np.linalg.det(R) < 0:
        u[:, -1] *= -1
        R = u @ vt
    return R


def _quat_to_matrix4(q: Any) -> np.ndarray:
    """四元数 ``[w,x,y,z]`` → 4x4 齐次旋转矩阵。"""
    m = np.eye(4, dtype=np.float32)
    m[:3, :3] = quat_wxyz_to_matrix(q)
    return m


def _our_rest_dirs() -> dict[str, np.ndarray]:
    """our 17 关节骨架的入骨 rest 方向（y-up，来自 retargeter MODEL_DEFINITION）。"""
    rest = MODEL_DEFINITION["rest_pose"]
    dirs: dict[str, np.ndarray] = {}
    for parent, child in MODEL_DEFINITION["bones"]:
        v = np.asarray(rest[child], dtype=np.float64) - np.asarray(
            rest[parent], dtype=np.float64
        )
        n = float(np.linalg.norm(v))
        if n > 1e-9:
            dirs[child] = (v / n).astype(np.float32)
    return dirs


class SkinAnimator:
    """GLB 蒙皮动画器：按帧应用 17 关节四元数并输出蒙皮后顶点。"""

    def __init__(self, model: GLBModel) -> None:
        self.model: GLBModel = model
        self.num_nodes: int = len(model.nodes)
        self.parent: list[int] = [n["parent"] for n in model.nodes]
        self.local: np.ndarray = np.stack([n["local"] for n in model.nodes])  # (N,4,4)

        # 拓扑序（父先于子），BFS 自根节点
        self.order: list[int] = self._topological_order()
        # rest pose 世界矩阵
        self.rest_world: np.ndarray = self._compute_world_rest()

        # 17 关节 → 节点索引
        self.joint_node: dict[str, int] = {}
        for joint, mixamo_name in JOINT_TO_MIXAMO.items():
            if mixamo_name not in model.node_index:
                raise KeyError(f"GLB 缺少骨骼节点：{mixamo_name}")
            self.joint_node[joint] = model.node_index[mixamo_name]

        # 换基重解所需的预计算量
        self.our_rest_dirs: dict[str, np.ndarray] = _our_rest_dirs()
        self._offset_local: dict[str, np.ndarray] = {}
        self._local_rot_inv: dict[str, np.ndarray] = {}
        for node_name, (child_name, _target) in BONE_SOLVE.items():
            j_idx = model.node_index[node_name]
            c_idx = model.node_index[child_name]
            bone = self.rest_world[c_idx][:3, 3] - self.rest_world[j_idx][:3, 3]
            norm = float(np.linalg.norm(bone))
            if norm < 1e-9:
                continue
            bone = bone / norm
            r_bind = self.rest_world[j_idx][:3, :3]
            self._offset_local[node_name] = (r_bind.T @ bone).astype(np.float32)
            self._local_rot_inv[node_name] = np.linalg.inv(
                self.local[j_idx][:3, :3].astype(np.float64)
            ).astype(np.float32)

    def _topological_order(self) -> list[int]:
        children: list[list[int]] = [[] for _ in range(self.num_nodes)]
        for i, p in enumerate(self.parent):
            if p >= 0:
                children[p].append(i)
        order: list[int] = []
        stack: list[int] = list(self.model.roots)
        while stack:
            node = stack.pop(0)
            order.append(node)
            stack.extend(children[node])
        return order

    def _compute_world_rest(self) -> np.ndarray:
        world = np.zeros((self.num_nodes, 4, 4), dtype=np.float32)
        for i in self.order:
            p = self.parent[i]
            world[i] = world[p] @ self.local[i] if p >= 0 else self.local[i]
        return world

    # ------------------------------------------------------------------
    def compute_world(self, delta: np.ndarray) -> np.ndarray:
        """按 ``G_j = G_parent × LocalRest_j × ΔQ_j`` 计算每节点世界矩阵。"""
        world = np.zeros((self.num_nodes, 4, 4), dtype=np.float32)
        for i in self.order:
            p = self.parent[i]
            local_anim = self.local[i] @ delta[i]
            world[i] = world[p] @ local_anim if p >= 0 else local_anim
        return world

    def skin_frame(self, joint_quats: dict[str, Any]) -> list[np.ndarray]:
        """对单帧应用四元数（含换基重解），返回每 mesh 蒙皮顶点 ``[(N_i,3), ...]``。"""
        # 1) 反解各 our-joint 入骨目标方向，翻转 y 换算到 y-up 世界系
        targets: dict[str, np.ndarray] = {}
        for joint, q in joint_quats.items():
            rest_dir = self.our_rest_dirs.get(joint)
            if rest_dir is None:
                continue  # 根关节（Hips）无入骨
            t = quat_rotate(q, rest_dir)
            targets[joint] = np.array([t[0], -t[1], t[2]], dtype=np.float32)

        # 2) 自顶向下遍历：对受驱节点按出骨重解局部旋转（父链补偿）
        delta = np.tile(np.eye(4, dtype=np.float32)[None, :, :], (self.num_nodes, 1, 1))
        world = np.zeros((self.num_nodes, 4, 4), dtype=np.float32)
        for i in self.order:
            p = self.parent[i]
            name = self.model.nodes[i]["name"]
            if p >= 0 and name in BONE_SOLVE:
                _child, target_joint = BONE_SOLVE[name]
                target = targets.get(target_joint)
                offset = self._offset_local.get(name)
                if target is not None and offset is not None:
                    parent_rot = world[p][:3, :3]  # R(W_parent)
                    # ΔQ_local = quat_between(offset_local, R_j⁻¹ · R_parent⁻¹ · T)
                    desired = self._local_rot_inv[name] @ (parent_rot.T @ target)
                    delta[i] = _quat_to_matrix4(quat_between(offset, desired))

            local_anim = self.local[i] @ delta[i]
            world[i] = world[p] @ local_anim if p >= 0 else local_anim

        # 3) 蒙皮
        result: list[np.ndarray] = []
        for mi, mesh in enumerate(self.model.meshes):
            skin = world[self.model.skin_joints[mi]] @ self.model.inverse_bind_matrices[mi]
            result.append(self._skin_mesh(mesh, skin))
        return result

    def skin_frame_direct(self, joint_quats: dict[str, Any]) -> list[np.ndarray]:
        """字面应用（LocalRest × ΔQ，无换基）——仅用于对照/调试。"""
        delta = np.tile(np.eye(4, dtype=np.float32)[None, :, :], (self.num_nodes, 1, 1))
        for joint, q in joint_quats.items():
            node = self.joint_node[joint]
            delta[node] = _quat_to_matrix4(q)
        world = self.compute_world(delta)
        return self._skin_all(world)

    def _build_pos(self, kps_frame: np.ndarray) -> dict[str, np.ndarray]:
        """由单帧 ``[33,4]`` 关键点构建 17 关节 3D 位置（y 翻转为向上）。"""
        def lm(i):
            x, y, z = float(kps_frame[i, 0]), float(kps_frame[i, 1]), float(kps_frame[i, 2])
            # 观察者视角 → 模型视角：x 左右镜像、y 上下翻转
            return np.array([-x, -y, z * 200.0], dtype=np.float64)

        hl, hr = lm(23), lm(24)
        sl, sr = lm(11), lm(12)
        hips = (hl + hr) / 2.0
        chest = (sl + sr) / 2.0
        spine = (hips + chest) / 2.0
        neck = chest + (chest - spine) * 0.5
        return {
            "Hips": hips, "Spine": spine, "Chest": chest, "Neck": neck, "Head": lm(0),
            "L_Shoulder": lm(11), "L_Elbow": lm(13), "L_Wrist": lm(15),
            "R_Shoulder": lm(12), "R_Elbow": lm(14), "R_Wrist": lm(16),
            "L_Hip": lm(23), "L_Knee": lm(25), "L_Ankle": lm(27),
            "R_Hip": lm(24), "R_Knee": lm(26), "R_Ankle": lm(28),
        }

    def _child_node(self, node: int) -> int:
        for c in range(self.num_nodes):
            if self.parent[c] == node:
                return c
        return -1

    @staticmethod
    def _m4(r: np.ndarray) -> np.ndarray:
        m = np.eye(4, dtype=np.float32)
        m[:3, :3] = r.astype(np.float32)
        return m

    def _hips_rot(self, pos: dict[str, np.ndarray]) -> np.ndarray:
        up = _norm(pos["Chest"] - pos["Hips"])
        right = _norm(pos["R_Hip"] - pos["L_Hip"])
        fwd = np.cross(right, up)
        if float(np.linalg.norm(fwd)) < 1e-6:
            fwd = _perp(up)
        fwd = _norm(fwd)
        right2 = np.cross(up, fwd)  # 正交化 right，保证 det=+1（右手系）
        return np.column_stack([right2, up, fwd])

    def skin_frame_from_kps(self, kps_frame: np.ndarray) -> list[np.ndarray]:
        """三关节法线法重定向 + LBS：直接由关键点帧驱动（保留 twist，动作不乱）。

        对每根受驱骨骼，用「入骨方向 + 出骨方向」构造正交基（含平面法线编码 twist），
        目标基相对 rest 基的旋转作为 world 增量，再换基到父关节 rest 局部系得到局部 ΔQ。
        """
        pos = self._build_pos(kps_frame)
        delta = np.tile(np.eye(4, dtype=np.float32)[None, :, :], (self.num_nodes, 1, 1))
        pose_world = self.rest_world.copy()  # 默认 rest，未映射节点保持 rest

        hn = self.joint_node["Hips"]
        lr_hips = self.local[hn][:3, :3]
        R_hips = self._hips_rot(pos)
        delta[hn] = self._m4(np.linalg.inv(lr_hips) @ R_hips)
        pose_world[hn] = self.rest_world[hn] @ delta[hn]  # 保持 rest 缩放

        for i in self.order:
            p = self.parent[i]
            if p < 0 or i == hn:
                continue
            name = self.model.nodes[i]["name"]
            if name not in TRI_JOINTS and name not in END_JOINTS:
                continue  # 未映射节点保持 rest
            lr = self.local[i][:3, :3]
            parent_pose = _pure_rot(pose_world[p][:3, :3])
            parent_rest = _pure_rot(self.rest_world[p][:3, :3])

            if name in TRI_JOINTS:
                _gp, pj, cj = TRI_JOINTS[name]
                b_t = pos[cj] - pos[pj]  # 目标出骨方向
                cnode = self._child_node(i)
                b_r = (
                    self.rest_world[cnode][:3, 3] - self.rest_world[i][:3, 3]
                    if cnode >= 0
                    else self.rest_world[i][:3, 3] - self.rest_world[p][:3, 3]
                )
                if np.linalg.norm(b_t) < 1e-6 or np.linalg.norm(b_r) < 1e-6:
                    dR = np.eye(3)  # 关键点缺失/重合，保持 rest
                else:
                    dR = _R.align_vectors(
                        [_norm(b_t)], [_norm(b_r)]
                    )[0].as_matrix()
            else:  # END_JOINTS
                pj, cj = END_JOINTS[name]
                a_t = pos[cj] - pos[pj]
                a_r = self.rest_world[i][:3, 3] - self.rest_world[p][:3, 3]
                if np.linalg.norm(a_t) < 1e-6 or np.linalg.norm(a_r) < 1e-6:
                    dR = np.eye(3)
                else:
                    dR = _R.align_vectors(
                        [_norm(a_t)], [_norm(a_r)]
                    )[0].as_matrix()

            dq = np.linalg.inv(lr) @ parent_pose.T @ dR @ parent_rest @ lr
            delta[i] = self._m4(dq)
            pose_world[i] = pose_world[p] @ self.local[i] @ delta[i]

        result: list[np.ndarray] = []
        for mi, mesh in enumerate(self.model.meshes):
            skin = pose_world[self.model.skin_joints[mi]] @ self.model.inverse_bind_matrices[mi]
            result.append(self._skin_mesh(mesh, skin))
        return result

    def skin_rest(self) -> list[np.ndarray]:
        """返回 rest pose 蒙皮顶点（用于包围盒/抽帧自验）。"""
        return self._skin_all(self.rest_world)

    # ------------------------------------------------------------------
    def _skin_all(self, world: np.ndarray) -> list[np.ndarray]:
        result: list[np.ndarray] = []
        for mi, mesh in enumerate(self.model.meshes):
            skin = world[self.model.skin_joints[mi]] @ self.model.inverse_bind_matrices[mi]
            result.append(self._skin_mesh(mesh, skin))
        return result

    @staticmethod
    def _skin_mesh(mesh: Any, skin: np.ndarray) -> np.ndarray:
        """对单个 mesh 做加权蒙皮，返回 ``(N,3)``。"""
        n = mesh.positions.shape[0]
        v_h = np.concatenate(
            [mesh.positions, np.ones((n, 1), dtype=np.float32)], axis=1
        )  # (N,4)
        acc = np.zeros((n, 4), dtype=np.float32)
        for k in range(4):
            sk = skin[mesh.joints[:, k]]  # (N,4,4)
            tv = np.einsum("nij,nj->ni", sk, v_h)  # (N,4)
            acc += mesh.weights[:, k : k + 1] * tv
        return acc[:, :3].astype(np.float32)
