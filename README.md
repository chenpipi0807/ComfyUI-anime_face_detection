# PIP 动漫人脸检测

这玩意是用来检测人脸的，我试了一下对动漫和真人都有效。

基于 [Hugging Face deepghs/anime_face_detection](https://huggingface.co/deepghs/anime_face_detection) 项目的 ComfyUI 实现。这个项目用的是 ONNX 模型来检测动漫人脸，还能调节置信度和 IOU 阈值。
![微信截图_20241129164819](https://github.com/user-attachments/assets/28d28635-157d-447f-862f-79b65259dcf1)

# PIP 更新日志
0217 新增了图像随机拼接 基于稳定的人脸检测确保每张裁切的图面部都不被裁切
0218 新增mode选项，多头模式和单头模式
<img width="1498" alt="783E4F40-63D6-4100-BB15-4B0AFE197093" src="https://github.com/user-attachments/assets/71ba4e74-7937-4e26-ba56-b593b8504d5f" />



## 功能

- **人脸检测**：能找出图里的动漫人脸。
- **可调节参数**：
  - 置信度阈值（`score_threshold`）
  - IOU 阈值（`iou_threshold`）
- **输出**：
  - 有框的图像
  - 裁剪出最大的人脸
  - 没找到人脸就输出原图

## 更新记录
  - 12月9日
  - 新增裁切人脸的扩展参数，避免默认裁切人脸过小的问题
  - 新增裁切人脸区域的MASK输出，便于用于其他应用场景

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
