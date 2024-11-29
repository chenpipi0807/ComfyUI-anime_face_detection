# PIP 动漫人脸检测

这玩意是用来检测人脸的，我试了一下对动漫和真人都有效。

基于 [Hugging Face deepghs/anime_face_detection](https://huggingface.co/deepghs/anime_face_detection) 项目的 ComfyUI 实现。这个项目用的是 ONNX 模型来检测动漫人脸，还能调节置信度和 IOU 阈值。
![微信截图_20241129164819](https://github.com/user-attachments/assets/28d28635-157d-447f-862f-79b65259dcf1)

## 功能

- **人脸检测**：能找出图里的动漫人脸。
- **可调节参数**：
  - 置信度阈值（`score_threshold`）
  - IOU 阈值（`iou_threshold`）
- **输出**：
  - 有框的图像
  - 裁剪出最大的人脸
  - 没找到人脸就输出原图

## 安装

得装这些依赖：

pip install -r requirements.txt

下载模型：

https://huggingface.co/deepghs/anime_face_detection/tree/main/face_detect_v0_nms

把模型放这儿：

ComfyUI\custom_nodes\ComfyUI-anime_face_detection\models

## 使用

在 ComfyUI 里加载 `PIP_AnimeFaceDetect` 节点，放图进去，调调参数，看能不能检测得更好。

## 错误处理

- 如果没找到人脸，就直接输出原图。
- 只输出最大的那张人脸。

## 参考

- [Hugging Face deepghs/anime_face_detection](https://huggingface.co/deepghs/anime_face_detection)

## 许可证

MIT License
