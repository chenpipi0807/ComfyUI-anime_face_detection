from .PIP_AnimeFaceDetect import PIP_AnimeFaceDetect
from .collage import PIP_Collage

# 节点映射配置
NODE_CLASS_MAPPINGS = {
    "PIP_AnimeFaceDetect": PIP_AnimeFaceDetect,
    "PIP_Collage": PIP_Collage
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "PIP_AnimeFaceDetect": "PIP 动漫人脸检测",
    "PIP_Collage": "PIP 随机拼图"
}
