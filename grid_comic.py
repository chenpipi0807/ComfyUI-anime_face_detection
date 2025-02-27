import torch
import random
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFilter
import copy

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

class LS_CollageGenerator:
    """根据图像比例生成合适的四分格布局"""
    def __init__(self, width, height, border_width, r, seed):
        self.width = width
        self.height = height
        self.border_width = int((self.width + self.height) * border_width / 200)
        self.r = r
        self.seed = seed
        # 生成适合1:2整体比例的矩形布局
        self.rectangles = self.generate_optimized_layout()

    def generate_optimized_layout(self):
        """根据图像原始比例生成优化布局"""
        random.seed(self.seed)
        
        # 整体画布是1:2比例，即高度是宽度的2倍
        # 将布局分为三行：顶部为表情(2:1)，中部为角色(1:2)，底部为场景和物品(都是1:1)
        
        # 布局变化的随机性
        rand_factor = 0.05
        h_variance = int(self.height * rand_factor)
        w_variance = int(self.width * rand_factor)
        
        # 计算基础高度
        # 表情区域：占20%-25%的高度
        expression_height = int(self.height * (0.20 + random.uniform(0, 0.05)))
        
        # 角色区域：占45%-55%的高度
        character_height = int(self.height * (0.45 + random.uniform(0, 0.10)))
        
        # 底部区域：剩余高度，分为场景和物品两个1:1区域
        bottom_height = self.height - expression_height - character_height
        
        # 添加一点随机变化到布局
        expression_height += random.randint(-h_variance, h_variance)
        character_height += random.randint(-h_variance, h_variance)
        
        # 确保高度和不超过总高度
        if expression_height + character_height > self.height - bottom_height/2:
            excess = (expression_height + character_height) - (self.height - bottom_height/2)
            expression_height -= excess // 2
            character_height -= excess // 2
        
        # 1. 表情区域(2:1) - 顶部横向矩形
        expression_rect = (0, 0, self.width, expression_height)
        
        # 2. 角色区域(1:2) - 中部垂直矩形，放在左侧
        character_width = int(self.width * 0.6) + random.randint(-w_variance, w_variance)
        character_rect = (0, expression_height, character_width, character_height)
        
        # 3-4. 场景和物品区域(1:1) - 右侧和底部
        # 场景放在角色右侧
        scene_rect = (character_width, expression_height, 
                      self.width - character_width, character_height)
        
        # 物品放在底部
        item_rect = (0, expression_height + character_height, 
                     self.width, bottom_height)
        
        rectangles = [character_rect, expression_rect, scene_rect, item_rect]
        return self.adjust_bboxes_with_gaps(rectangles)

    def adjust_bboxes_with_gaps(self, rectangles):
        MIN_SIZE = 1
        adjusted_bboxes = []

        for x, y, w, h in rectangles:
            new_x = min(x + self.border_width, self.width - MIN_SIZE)
            new_y = min(y + self.border_width, self.height - MIN_SIZE)
            new_w = max(MIN_SIZE, w - 2 * self.border_width)
            new_h = max(MIN_SIZE, h - 2 * self.border_width)

            if new_x + new_w > self.width:
                new_w = max(MIN_SIZE, self.width - new_x - self.border_width)
            if new_y + new_h > self.height:
                new_h = max(MIN_SIZE, self.height - new_y - self.border_width)

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

class PIP_GridComic:
    def __init__(self):
        self.NODE_NAME = 'PIP_GridComic'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "character_image": ("IMAGE", {"description": "角色图像 (1:2)"}),
                "expression_image": ("IMAGE", {"description": "表情图像 (2:1)"}),
                "scene_image": ("IMAGE", {"description": "场景图像 (1:1)"}),
                "item_image": ("IMAGE", {"description": "物品图像 (1:1)"}),
                "canvas_width": ("INT", {
                    "default": 1024,
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
                })
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("输出图像", "遮罩",)
    FUNCTION = "collage"
    CATEGORY = 'PIP 动漫人脸检测'

    def resize_image(self, image, target_width, target_height, img_type=None):
        """将图像调整到目标尺寸，根据图像类型优化裁剪"""
        orig_ratio = image.width / image.height
        target_ratio = target_width / target_height
        
        # 针对不同类型图像的裁剪策略
        if img_type == "character":  # 角色图(1:2)优先保留上半部分(头部)
            new_width = target_width
            new_height = int(new_width / orig_ratio)
            resized = image.resize((new_width, new_height), Image.LANCZOS)
            
            if new_height > target_height:
                # 保留上部(头部)，裁剪底部
                top = 0
                resized = resized.crop((0, top, target_width, top + target_height))
            else:
                # 需要填充
                new_img = Image.new("RGBA", (target_width, target_height), (0, 0, 0, 0))
                new_img.paste(resized, (0, 0))
                resized = new_img
                
        elif img_type == "expression":  # 表情图(2:1)居中裁剪
            if orig_ratio > target_ratio:
                new_height = target_height
                new_width = int(orig_ratio * new_height)
                resized = image.resize((new_width, new_height), Image.LANCZOS)
                # 居中裁剪
                left = (new_width - target_width) // 2
                resized = resized.crop((left, 0, left + target_width, target_height))
            else:
                new_width = target_width
                new_height = int(new_width / orig_ratio)
                resized = image.resize((new_width, new_height), Image.LANCZOS)
                # 居中裁剪
                top = (new_height - target_height) // 2
                resized = resized.crop((0, top, target_width, top + target_height))
        
        else:  # 场景和物品图(1:1)，默认居中裁剪
            if orig_ratio > target_ratio:
                new_height = target_height
                new_width = int(orig_ratio * new_height)
                resized = image.resize((new_width, new_height), Image.LANCZOS)
                # 居中裁剪
                left = (new_width - target_width) // 2
                resized = resized.crop((left, 0, left + target_width, target_height))
            else:
                new_width = target_width
                new_height = int(new_width / orig_ratio)
                resized = image.resize((new_width, new_height), Image.LANCZOS)
                # 居中裁剪
                top = (new_height - target_height) // 2
                resized = resized.crop((0, top, target_width, top + target_height))
            
        return resized

    def collage(self, character_image, expression_image, scene_image, item_image, 
                canvas_width, canvas_height, border_width, rounded_rect_radius,
                background_color, seed):
        
        # 将四张单图像组合成一个列表，顺序很重要
        # 顺序为：角色(0), 表情(1), 场景(2), 物品(3)
        images = [character_image[0], expression_image[0], scene_image[0], item_image[0]]
        img_types = ["character", "expression", "scene", "item"]
        
        # 创建生成器
        rects = LS_CollageGenerator(width=canvas_width,
                                   height=canvas_height,
                                   border_width=border_width,
                                   r=rounded_rect_radius,
                                   seed=seed)

        canvas = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))
        mask = Image.new("L", (canvas_width, canvas_height), 0)
        mask_draw = ImageDraw.Draw(mask)
        
        # 绘制每个图像到指定位置
        for i, img_tensor in enumerate(images):
            img = tensor2pil(img_tensor).convert("RGBA")
            img_x, img_y = rects.rectangles[i][0], rects.rectangles[i][1]
            img_w, img_h = rects.rectangles[i][2], rects.rectangles[i][3]
            
            # 根据图像类型调整大小
            resized_img = self.resize_image(img, img_w, img_h, img_types[i])
            rounded_img = add_corners_to_image(resized_img, rounded_rect_radius)
            
            canvas.paste(rounded_img, (img_x, img_y), mask=rounded_img.split()[3])
            mask_draw.rectangle([img_x, img_y, img_x + img_w, img_y + img_h], fill=255)

        final_image = Image.new("RGB", (canvas_width, canvas_height), background_color)
        final_image.paste(canvas, mask=canvas.split()[3])
        
        return (pil2tensor(final_image), pil2tensor(mask))

NODE_CLASS_MAPPINGS = {
    "PIP_GridComic": PIP_GridComic,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PIP_GridComic": "PIP 格漫拼图",
}