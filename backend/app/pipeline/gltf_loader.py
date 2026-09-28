"""GLB 解析：从 rigged 模型提取顶点/索引/蒙皮权重/关节层级。

只依赖 ``pygltflib`` 与 ``numpy``，不做渲染。支持 glTF 节点的两种局部变换形式：
``matrix``（4x4）或 ``translation/rotation/scale``（TRS）。顶点蒙皮属性约定：
``JOINTS_0``（uint8/uint16/uint32 的 VEC4）与 ``WEIGHTS_0``（float VEC4）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from pygltflib import GLTF2

# ---------------------------------------------------------------------------
# accessor 组件类型 → numpy dtype
# ---------------------------------------------------------------------------
_COMPONENT_DTYPE: dict[int, np.dtype] = {
    5120: np.dtype(np.int8),
    5121: np.dtype(np.uint8),
    5122: np.dtype(np.int16),
    5123: np.dtype(np.uint16),
    5125: np.dtype(np.uint32),
    5126: np.dtype(np.float32),
}

_TYPE_COMPONENTS: dict[str, int] = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT4": 16,
}


@dataclass
class GLBMesh:
    """单个 mesh primitive 的蒙皮数据（均为 numpy 数组）。"""

    positions: np.ndarray  # (N, 3) float32
    joints: np.ndarray  # (N, 4) int32（skin joints 的本地索引）
    weights: np.ndarray  # (N, 4) float32
    indices: np.ndarray  # (T, 3) int32（三角形）
    base_color: tuple[float, float, float] = (0.8, 0.8, 0.8)


@dataclass
class GLBModel:
    """GLB 蒙皮模型：节点层级 + 各 mesh 的蒙皮与 IBM。"""

    nodes: list[dict[str, Any]] = field(default_factory=list)  # name/parent/local
    node_index: dict[str, int] = field(default_factory=dict)
    roots: list[int] = field(default_factory=list)
    meshes: list[GLBMesh] = field(default_factory=list)
    skin_joints: list[list[int]] = field(default_factory=list)  # 每 mesh 的 skin joint 节点索引
    inverse_bind_matrices: list[np.ndarray] = field(default_factory=list)  # 每 mesh (J,4,4)


def _read_accessor(gltf: GLTF2, accessor_index: int) -> np.ndarray:
    """按 accessor 读取数据为 numpy 数组（处理 bufferView stride 与 MAT4）。"""
    acc = gltf.accessors[accessor_index]
    bv = gltf.bufferViews[acc.bufferView]
    blob = gltf.binary_blob()

    dtype = _COMPONENT_DTYPE[acc.componentType]
    ncomp = _TYPE_COMPONENTS.get(acc.type)
    if ncomp is None:
        raise ValueError(f"未知 accessor 类型：{acc.type}")

    base = int(bv.byteOffset or 0) + int(acc.byteOffset or 0)
    itemsize = int(dtype.itemsize)
    stride = int(bv.byteStride or (itemsize * ncomp))

    if stride == itemsize * ncomp:
        data = np.frombuffer(blob, dtype=dtype, count=acc.count * ncomp, offset=base)
        data = data.reshape(acc.count, ncomp)
    else:
        # 交错存储：逐行读取
        data = np.empty((acc.count, ncomp), dtype=dtype)
        for i in range(acc.count):
            off = base + i * stride
            data[i] = np.frombuffer(blob, dtype=dtype, count=ncomp, offset=off)

    # glTF 的 MAT4 以列主序存储，转置为行主序便于矩阵乘法
    if acc.type == "MAT4":
        data = data.reshape(acc.count, 4, 4).transpose(0, 2, 1)
    return data


# ---------------------------------------------------------------------------
# 四元数 / 矩阵工具
# ---------------------------------------------------------------------------
def quat_xyzw_to_matrix(q: Any) -> np.ndarray:
    """glTF 节点四元数 ``[x,y,z,w]`` → 3x3 旋转矩阵。"""
    x, y, z, w = (float(v) for v in q)
    return quat_wxyz_to_matrix([w, x, y, z])


def quat_wxyz_to_matrix(q: Any) -> np.ndarray:
    """四元数 ``[w,x,y,z]`` → 3x3 旋转矩阵（与 animation 数据约定一致）。"""
    w, x, y, z = (float(v) for v in q)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n > 1e-12:
        w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def node_local_matrix(node: Any) -> np.ndarray:
    """节点局部变换（matrix 或 TRS）→ 4x4 矩阵。"""
    if getattr(node, "matrix", None) is not None:
        return np.array(node.matrix, dtype=np.float32).reshape(4, 4)

    t = np.array(node.translation or [0.0, 0.0, 0.0], dtype=np.float32)
    r = np.array(node.rotation or [0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    s = np.array(node.scale or [1.0, 1.0, 1.0], dtype=np.float32)

    m = np.eye(4, dtype=np.float32)
    m[:3, :3] = quat_xyzw_to_matrix(r) @ np.diag(s.astype(np.float32))
    m[:3, 3] = t
    return m


def _base_color(gltf: GLTF2, material_index: int | None) -> tuple[float, float, float]:
    """提取材质 baseColorFactor 的 RGB（缺省灰白色）。"""
    try:
        if material_index is not None and gltf.materials is not None:
            mat = gltf.materials[material_index]
            pbr = mat.pbrMetallicRoughness
            if pbr is not None and pbr.baseColorFactor is not None:
                c = pbr.baseColorFactor
                return (float(c[0]), float(c[1]), float(c[2]))
    except Exception:  # noqa: BLE001 - 材质解析失败回退默认色
        pass
    return (0.8, 0.8, 0.8)


def load_glb(path: Any) -> GLBModel:
    """解析 GLB，返回 :class:`GLBModel`。"""
    gltf = GLTF2().load(str(path))

    model = GLBModel()

    # 1) 节点层级
    parents: list[int] = []
    for i, node in enumerate(gltf.nodes):
        name = node.name or f"node_{i}"
        model.nodes.append(
            {"name": name, "parent": -1, "local": node_local_matrix(node)}
        )
        model.node_index[name] = i
        parents.append(-1)

    for i, node in enumerate(gltf.nodes):
        for child in node.children or []:
            parents[child] = i
    for i, node in enumerate(model.nodes):
        node["parent"] = parents[i]
        if parents[i] == -1:
            model.roots.append(i)

    # 2) 每 mesh 的蒙皮数据 + skin 信息
    for mesh in gltf.meshes:
        for prim in mesh.primitives:
            pos = _read_accessor(gltf, prim.attributes.POSITION).astype(np.float32)
            joints = _read_accessor(gltf, prim.attributes.JOINTS_0).astype(np.int32)
            weights = _read_accessor(gltf, prim.attributes.WEIGHTS_0).astype(np.float32)
            indices = _read_accessor(gltf, prim.indices).astype(np.int32).reshape(-1, 3)

            model.meshes.append(
                GLBMesh(
                    positions=pos,
                    joints=joints,
                    weights=weights,
                    indices=indices,
                    base_color=_base_color(gltf, prim.material),
                )
            )

    # 3) skin joints 与 inverseBindMatrices（每 mesh 一个 skin）
    for skin in gltf.skins:
        model.skin_joints.append(list(skin.joints))
        ibm = _read_accessor(gltf, skin.inverseBindMatrices).astype(np.float32)
        model.inverse_bind_matrices.append(ibm.reshape(-1, 4, 4))

    return model
