import torch
import random
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFilter
import copy
from .PIP_AnimeFaceDetect import PIP_AnimeFaceDetect

def tensor2pil(image):
    """将tensor转换为PIL图像"""
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    """将PIL图像转换为tensor"""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def add_corners_to_image(image, radius):
    """给单个图片添加圆角"""
    mask = Image.new('L', image.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle([(0, 0), image.size], radius=radius, fill=255)
    output = Image.new('RGBA', image.size, (0, 0, 0, 0))
    output.paste(image, mask=mask)
    return output

def draw_rounded_rectangle(image, radius, bboxes, scale_factor=4, color='black'):
    """绘制圆角矩形"""
    w, h = image.size
    big_size = (int(w * scale_factor), int(h * scale_factor))
    mask = Image.new('L', big_size, 0)
    draw = ImageDraw.Draw(mask)

    for bbox in bboxes:
        x1, y1, x2, y2 = [int(coord * scale_factor) for coord in bbox]
        draw.rounded_rectangle([x1, y1, x2, y2], radius=radius * scale_factor, fill=255)

    mask = mask.resize(image.size, Image.LANCZOS)
    result = Image.new('RGB', image.size, color=color)
    result.putalpha(mask)
    return result

def gaussian_blur(image, radius):
    """应用高斯模糊"""
    return image.filter(ImageFilter.GaussianBlur(radius))

def mask_area(mask):
    """计算mask的边界框"""
    mask_np = np.array(mask)
    y_indices, x_indices = np.nonzero(mask_np)
    
    if len(x_indices) == 0 or len(y_indices) == 0:
        return (0, 0, mask_np.shape[1], mask_np.shape[0])
    
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    return (x_min, y_min, x_max - x_min, y_max - y_min)

class LS_CollageGenerator:
    """随机分割生成指定数量的不规则小矩形。"""
    def __init__(self, width, height, num, border_width, r, uniformity, seed):
        self.width = width
        self.height = height
        self.num = num
        self.border_width = int((self.width + self.height) * border_width / 200)
        self.r = r
        self.seed = seed
        self.split_num = int(1e18)
        self.uniformity = uniformity
        self.rectangles = self.adjust_bboxes_with_gaps(self.split_rec())

    def split_rec(self):
        random.seed(self.seed)
        if self.num <= 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("Value must be positive integer")

        current_rectangles = [(0, 0, self.width, self.height, 0)]

        while len(current_rectangles) < self.num:
            split_counts = [rect[4] for rect in current_rectangles]
            min_splits = min(split_counts)
            max_splits = max(split_counts)
            probabilities = []

            for rect in current_rectangles:
                split_count = rect[4]
                normalized_splits = (split_count - min_splits) / (
                    max_splits - min_splits if max_splits > min_splits else 1)
                probability = 1 - (normalized_splits * (1 - self.uniformity))
                probabilities.append(probability)
            if sum(probabilities) > 0:
                probabilities = [p / sum(probabilities) for p in probabilities]
            else:
                probabilities = [1.0 / len(probabilities)] * len(probabilities)

            rect_index = random.choices(range(len(current_rectangles)),
                                     weights=probabilities, k=1)[0]

            x, y, w, h, split_count = current_rectangles.pop(rect_index)

            if w > h or (w == h and random.choice([True, False])):
                split = random.uniform(0.3, 0.7) * w
                rect1 = (x, y, split, h, split_count + 1)
                rect2 = (x + split, y, w - split, h, split_count + 1)
            else:
                split = random.uniform(0.3, 0.7) * h
                rect1 = (x, y, w, split, split_count + 1)
                rect2 = (x, y + split, w, h - split, split_count + 1)

            current_rectangles.extend([rect1, rect2])

        rectangles = [(int(x), int(y), int(w), int(h))
                     for x, y, w, h, _ in current_rectangles]

        return rectangles

    def adjust_bboxes_with_gaps(self, rectangles):
        MIN_SIZE = 1
        adjusted_bboxes = []

        for x, y, w, h in rectangles:
            new_x = min(x + self.border_width, self.width - MIN_SIZE)
            new_y = min(y + self.border_width, self.height - MIN_SIZE)
            new_w = max(MIN_SIZE, w - 2 * self.border_width)
            new_h = max(MIN_SIZE, h - 2 * self.border_width)

            if new_x + new_w > self.width:
                new_x = max(0, self.width - new_w)
            if new_y + new_h > self.height:
                new_y = max(0, self.height - new_h)

            adjusted_bboxes.append((new_x, new_y, new_w, new_h))

        return adjusted_bboxes

    def draw_mask(self):
        bboxes = []
        for bbox in self.rectangles:
            bboxes.append((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
        scale_factor = 4  # 增大scale_factor
        img = Image.new('RGB', (self.width, self.height), color='white')
        img = draw_rounded_rectangle(img, self.r, bboxes, scale_factor, color='black')
        return img

class PIP_Collage:
    def __init__(self):
        self.NODE_NAME = 'PIP_Collage'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"description": "输入图像"}),
                "canvas_width": ("INT", {
                    "default": 2048,
                    "min": 512,
                    "max": 8192,
                    "step": 16,
                    "display": "画布宽度"
                }),
                "canvas_height": ("INT", {
                    "default": 2048,
                    "min": 512,
                    "max": 8192,
                    "step": 16,
                    "display": "画布高度"
                }),
                "border_width": ("FLOAT", {
                    "default": 0.5,
                    "min": 0,
                    "max": 20,
                    "step": 0.1,
                    "display": "边框宽度"
                }),
                "rounded_rect_radius": ("INT", {
                    "default": 32,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "display": "圆角半径"
                }),
                "uniformity": ("FLOAT", {
                    "default": 0.5,
                    "min": 0,
                    "max": 1,
                    "step": 0.1,
                    "display": "均匀度"
                }),
                "background_color": ("STRING", {
                    "default": "#000000",
                    "display": "背景颜色"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": int(1e18),
                    "step": 1,
                    "display": "随机种子"
                }),
                "expand_ratio": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "人脸扩展比例"
                }),
                "mode": (["multi", "single"], {  # 修改后的关键部分
                    "default": "multi",
                    "display": "模式",
                    "description": "选择多人模式或单人模式"
                })
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("输出图像", "遮罩",)
    FUNCTION = "collage"
    CATEGORY = 'PIP 动漫人脸检测'

    def collage(self, images, canvas_width, canvas_height, border_width, rounded_rect_radius,
                uniformity, background_color, seed, expand_ratio=0.2, mode="multi"):
        batch_size = images.shape[0]
        
        # 创建随机顺序的索引
        indices = list(range(batch_size))
        random.seed(seed)
        random.shuffle(indices)
        
        images = images[indices]
        
        rects = LS_CollageGenerator(width=canvas_width,
                                  height=canvas_height,
                                  num=batch_size,
                                  border_width=border_width,
                                  r=rounded_rect_radius,
                                  uniformity=uniformity,
                                  seed=seed)

        canvas = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))
        mask = Image.new("L", (canvas_width, canvas_height), 0)
        mask_draw = ImageDraw.Draw(mask)

        if mode == "single":
            detector = PIP_AnimeFaceDetect()
            face_areas = []
            face_masks_all = []
            for i in range(batch_size):
                _, _, face_mask = detector.detect_and_crop_faces(
                    image_in=images[i].unsqueeze(0),
                    expand_ratio=expand_ratio
                )
                face_mask_np = face_mask.squeeze(0).cpu().numpy()
                area = np.sum(face_mask_np > 0)
                face_areas.append(area)
                face_masks_all.append(face_mask.squeeze(0))
            max_idx = int(np.argmax(face_areas))
        
        for i in tqdm(range(batch_size)):
            img = tensor2pil(images[i]).convert("RGBA")
            img_x, img_y = rects.rectangles[i][0], rects.rectangles[i][1]
            img_w, img_h = rects.rectangles[i][2], rects.rectangles[i][3]

            if mode == "multi":
                detector = PIP_AnimeFaceDetect()
                _, _, face_mask = detector.detect_and_crop_faces(
                    image_in=images[i].unsqueeze(0),
                    expand_ratio=expand_ratio
                )
                cur_mask = face_mask.squeeze(0)
            else:
                cur_mask = face_masks_all[i] if i == max_idx else torch.ones_like(face_masks_all[i]) - face_masks_all[i]
            
            resized_img = self.image_auto_crop_v3(img, img_w, img_h, cur_mask)
            rounded_img = add_corners_to_image(resized_img, rounded_rect_radius)
            
            canvas.paste(rounded_img, (img_x, img_y), mask=rounded_img.split()[3])
            mask_draw.rectangle([img_x, img_y, img_x + img_w, img_y + img_h], fill=255)

        final_image = Image.new("RGB", (canvas_width, canvas_height), background_color)
        final_image.paste(canvas, mask=canvas.split()[3])
        
        return (pil2tensor(final_image), pil2tensor(mask))

    def image_auto_crop_v3(self, image, proportional_width, proportional_height, mask):
        _image = image
        ratio = proportional_width / proportional_height
        
        if ratio > 1:
            target_width = proportional_width
            target_height = int(target_width / ratio)
        else:
            target_width = proportional_width
            target_height = int(target_width / ratio)

        bluredmask = gaussian_blur(tensor2pil(mask), 20).convert('L')
        mask_x, mask_y, mask_w, mask_h = mask_area(bluredmask)
        orig_ratio = _image.width / _image.height
        
        if orig_ratio > (target_width / target_height):
            crop_w = int(_image.height * (target_width / target_height))
            crop_h = _image.height
        else:
            crop_w = _image.width
            crop_h = int(_image.width / (target_width / target_height))
            
        crop_x = min(max(mask_w // 2 + mask_x - crop_w // 2, 0), _image.width - crop_w)
        crop_y = min(max(mask_h // 2 + mask_y - crop_h // 2, 0), _image.height - crop_h)
            
        return _image.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h)).resize(
            (target_width, target_height), Image.LANCZOS)

NODE_CLASS_MAPPINGS = {
    "PIP_Collage": PIP_Collage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PIP_Collage": "PIP 随机拼图",
}
