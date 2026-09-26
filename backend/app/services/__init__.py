"""后端服务层（T04）。

包含视频处理服务、注解服务、人工校准回写服务，以及统一的文件存储读写工具。
服务层通过 :mod:`app.services.storage` 读写 ``data/`` 下的元数据/关键点/动画，
通过 :class:`app.pipeline.pipeline.PipelineOrchestrator` 驱动离线处理管线。
"""
