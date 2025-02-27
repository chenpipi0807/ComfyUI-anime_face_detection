from .PIP_AnimeFaceDetect import PIP_AnimeFaceDetect
from .collage import PIP_Collage
from .grid_comic import PIP_GridComic

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "PIP_AnimeFaceDetect": PIP_AnimeFaceDetect,
    "PIP_Collage": PIP_Collage,
    "PIP_GridComic": PIP_GridComic
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "PIP_AnimeFaceDetect": "PIP 动漫人脸检测",
    "PIP_Collage": "PIP 随机拼图",
    "PIP_GridComic": "PIP 格漫拼图"
}
