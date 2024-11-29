import onnxruntime as ort
from imgutils import detect
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import os
import io

class PIP_AnimeFaceDetect:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_in": ("IMAGE", {}),
                "score_threshold": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "label": "置信度阈值"}),
                "iou_threshold": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01, "label": "IOU 阈值"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("带框图像", "裁剪人脸")
    CATEGORY = "PIP 动漫人脸检测"
    FUNCTION = "detect_and_crop_faces"

    def detect_and_crop_faces(self, image_in, score_threshold=0.25, iou_threshold=0.7):
        # 确保图像是正确的维度 (batch_size, height, width, channels)
        if image_in.dim() == 3:
            image_in = image_in.unsqueeze(0)  # 添加批次维度

        batch_size, height, width, channels = image_in.shape
        print(f"输入图像尺寸: {image_in.shape}")  # 调试信息

        # 处理每个批次的图像
        image_with_boxes_list = []
        cropped_faces_list = []

        for i in range(batch_size):
            img = image_in[i]

            # 确保图像数据在0-1范围内
            img = img.float() / 255.0 if img.dtype == torch.uint8 else img.float()
            img = torch.clamp(img, 0, 1)

            # 转换为 NumPy 数组之前调用 .cpu()
            img_np = (img.cpu().numpy() * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np, mode='RGB')

            # 使用ONNX Runtime加载本地模型并指定CUDA执行提供程序
            script_dir = os.path.dirname(__file__)  # 获取当前脚本的目录
            model_path = os.path.join(script_dir, 'models', 'model.onnx')  # 构建模型路径
            session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])

            # 使用正确的参数名称
            faces = detect.detect_faces(
                img_pil,
                conf_threshold=score_threshold,
                iou_threshold=iou_threshold
            )

            # 打印检测结果
            print("检测到的人脸:", faces)

            # 错误处理：如果没有检测到人脸，直接返回原图
            if not faces:
                print("未检测到人脸，返回原图")
                image_with_boxes_list.append(img)
                cropped_faces_list.append(img)
                continue

            # 选择面积最大的人脸
            largest_face = max(faces, key=lambda x: (x[0][2] - x[0][0]) * (x[0][3] - x[0][1]))

            # 使用matplotlib进行可视化
            fig, ax = plt.subplots(1)
            ax.imshow(img_pil)

            # 绘制边界框
            (x1, y1, x2, y2), label, score = largest_face
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.text(x1, y1, f'{label}: {score:.2f}', color='red', fontsize=12)

            # 将图像转换为 numpy 数组
            fig.canvas.draw()
            image_with_boxes = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image_with_boxes = image_with_boxes.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            # 转换为 PyTorch 张量
            image_with_boxes_tensor = torch.from_numpy(image_with_boxes.astype(np.float32) / 255.0)

            # 裁剪检测到的最大面部
            cropped_face = np.array(img_pil.crop((x1, y1, x2, y2)))
            cropped_face_tensor = torch.from_numpy(cropped_face.astype(np.float32) / 255.0)

            image_with_boxes_list.append(image_with_boxes_tensor)
            cropped_faces_list.append(cropped_face_tensor)

            plt.close(fig)  # 关闭图形以释放内存

        return image_with_boxes_list, cropped_faces_list

# 包含所有要导出的节点的字典，以及它们的名称
NODE_CLASS_MAPPINGS = {
    "PIP_AnimeFaceDetect": PIP_AnimeFaceDetect
}

# 包含节点的友好/人类可读标题的字典
NODE_DISPLAY_NAME_MAPPINGS = {
    "PIP_AnimeFaceDetect": "PIP 动漫人脸检测"
}
