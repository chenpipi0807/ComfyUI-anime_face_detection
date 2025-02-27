import torch
import random
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import copy
import math

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

class MangaStyleLayoutGenerator:
    """创意漫画布局生成器，使用真实漫画的多样化布局风格"""
    def __init__(self, width, height, border_width, r, seed, style_index=None):
        self.width = width
        self.height = height
        self.border_width = int((self.width + self.height) * border_width / 200)
        self.r = r
        self.seed = seed
        random.seed(self.seed)
        
        # 布局样式选择，可以指定或随机
        self.style_index = style_index if style_index is not None else random.randint(0, 9)
        print(f"Using manga layout style: {self.style_index}")
        
        # 生成布局
        self.rectangles, self.shapes, self.rotation_angles = self.generate_layout()
        
    def generate_layout(self):
        """根据选定的布局样式生成对应的布局"""
        layout_functions = [
            self.dynamic_layout_style_0,    # 基本四格布局但有角度变化
            self.dynamic_layout_style_1,    # 中心重点布局
            self.dynamic_layout_style_2,    # 阶梯式布局
            self.dynamic_layout_style_3,    # 斜线分割布局
            self.dynamic_layout_style_4,    # 散射式布局
            self.dynamic_layout_style_5,    # 随机大小不规则布局
            self.dynamic_layout_style_6,    # Z字形布局
            self.dynamic_layout_style_7,    # 放射状布局 
            self.dynamic_layout_style_8,    # 碎片式布局
            self.dynamic_layout_style_9,    # 多层次重叠布局
        ]
        
        return layout_functions[self.style_index]()

    def dynamic_layout_style_0(self):
        """基本四格但添加轻微角度和偏移变化"""
        # 基本区域
        width_mid = self.width // 2
        height_mid = self.height // 2
        
        # 添加随机变化
        jitter = min(self.width, self.height) * 0.05
        width_mid += random.uniform(-jitter, jitter)
        height_mid += random.uniform(-jitter, jitter)
        
        # 创建四个基本面板
        panels = []
        
        # 左上面板
        panels.append((0, 0, width_mid, height_mid))
        
        # 右上面板
        panels.append((width_mid, 0, self.width - width_mid, height_mid))
        
        # 左下面板
        panels.append((0, height_mid, width_mid, self.height - height_mid))
        
        # 右下面板
        panels.append((width_mid, height_mid, self.width - width_mid, self.height - height_mid))
        
        # 随机调整顺序以适应角色图片等内容
        random.shuffle(panels)
        
        # 指定形状(默认为矩形)和旋转角度
        shapes = ["rect", "rect", "rect", "rect"]
        
        # 轻微旋转角度
        angles = [
            random.uniform(-5, 5),
            random.uniform(-5, 5),
            random.uniform(-5, 5),
            random.uniform(-5, 5)
        ]
        
        # 调整面板大小，添加边框
        adjusted_panels = self.adjust_bboxes_with_gaps(panels)
        
        return adjusted_panels, shapes, angles

    def dynamic_layout_style_1(self):
        """中心重点布局 - 中心有一个大面板，周围有多个小面板"""
        panels = []
        
        # 计算中心面板大小 - 占据40-60%的画布
        center_width = self.width * random.uniform(0.4, 0.6)
        center_height = self.height * random.uniform(0.4, 0.6)
        
        # 中心面板位置
        center_x = (self.width - center_width) / 2
        center_y = (self.height - center_height) / 2
        
        # 添加中心面板 - 用于角色
        panels.append((center_x, center_y, center_width, center_height))
        
        # 添加上方面板 - 用于表情
        top_panel_height = center_y * random.uniform(0.8, 1.0)
        panels.append((center_x, 0, center_width, top_panel_height))
        
        # 添加左侧面板 - 用于场景
        left_panel_width = center_x * random.uniform(0.8, 1.0)
        panels.append((0, center_y, left_panel_width, center_height))
        
        # 添加右侧面板 - 用于物品
        right_panel_x = center_x + center_width
        right_panel_width = self.width - right_panel_x
        panels.append((right_panel_x, center_y, right_panel_width, center_height))
        
        # 指定形状和旋转角度
        shapes = ["rect", "rect", "rect", "rect"]
        angles = [0, 0, 0, 0]  # 这种布局不需要旋转
        
        # 调整面板大小，添加边框
        adjusted_panels = self.adjust_bboxes_with_gaps(panels)
        
        return adjusted_panels, shapes, angles

    def dynamic_layout_style_2(self):
        """阶梯式布局 - 面板按阶梯状排列"""
        panels = []
        
        # 将画布分为多个区域
        step_width = self.width / 3
        step_height = self.height / 3
        
        # 角色面板 (大尺寸)
        character_x = 0
        character_y = 0
        character_w = step_width * 2
        character_h = step_height * 2
        panels.append((character_x, character_y, character_w, character_h))
        
        # 表情面板
        expr_x = character_w
        expr_y = 0
        expr_w = self.width - character_w
        expr_h = step_height
        panels.append((expr_x, expr_y, expr_w, expr_h))
        
        # 场景面板
        scene_x = character_w
        scene_y = expr_h
        scene_w = self.width - character_w
        scene_h = step_height
        panels.append((scene_x, scene_y, scene_w, scene_h))
        
        # 物品面板
        item_x = 0
        item_y = character_h
        item_w = self.width
        item_h = self.height - character_h
        panels.append((item_x, item_y, item_w, item_h))
        
        # 指定形状和旋转角度
        shapes = ["rect", "rect", "rect", "rect"]
        angles = [0, 0, 0, 0]
        
        # 调整面板大小，添加边框
        adjusted_panels = self.adjust_bboxes_with_gaps(panels)
        
        return adjusted_panels, shapes, angles

    def dynamic_layout_style_3(self):
        """斜线分割布局 - 使用对角线分割画布"""
        panels = []
        
        # 使用对角线分割画布，创建不规则四边形
        # 随机选择对角线方向
        if random.random() > 0.5:
            # 左上到右下对角线
            diagonal_points = [(0, 0), (self.width, self.height)]
            
            # 左上三角形区域 - 角色
            top_left_w = self.width * 0.6
            top_left_h = self.height * 0.6
            panels.append((0, 0, top_left_w, top_left_h))
            
            # 右上三角形区域 - 表情
            top_right_x = top_left_w
            top_right_y = 0
            panels.append((top_right_x, top_right_y, self.width - top_right_x, self.height * 0.45))
            
            # 左下三角形区域 - 场景
            bottom_left_x = 0
            bottom_left_y = top_left_h
            panels.append((bottom_left_x, bottom_left_y, self.width * 0.45, self.height - bottom_left_y))
            
            # 右下三角形区域 - 物品
            bottom_right_x = self.width * 0.55
            bottom_right_y = self.height * 0.55
            panels.append((bottom_right_x, bottom_right_y, 
                           self.width - bottom_right_x, self.height - bottom_right_y))
        else:
            # 右上到左下对角线
            diagonal_points = [(self.width, 0), (0, self.height)]
            
            # 右上三角形区域 - 角色
            top_right_w = self.width * 0.6
            top_right_h = self.height * 0.6
            panels.append((self.width - top_right_w, 0, top_right_w, top_right_h))
            
            # 左上三角形区域 - 表情
            top_left_x = 0
            top_left_y = 0
            panels.append((top_left_x, top_left_y, self.width * 0.45, self.height * 0.45))
            
            # 右下三角形区域 - 场景
            bottom_right_x = self.width * 0.55
            bottom_right_y = top_right_h
            panels.append((bottom_right_x, bottom_right_y, 
                           self.width - bottom_right_x, self.height - bottom_right_y))
            
            # 左下三角形区域 - 物品
            bottom_left_x = 0
            bottom_left_y = self.height * 0.55
            panels.append((bottom_left_x, bottom_left_y, 
                           self.width * 0.45, self.height - bottom_left_y))
        
        # 指定形状(使用多边形)和旋转角度
        shapes = ["rect", "rect", "rect", "rect"]
        angles = [0, 0, 0, 0]
        
        # 调整面板大小，添加边框
        adjusted_panels = self.adjust_bboxes_with_gaps(panels)
        
        return adjusted_panels, shapes, angles

    def dynamic_layout_style_4(self):
        """散射式布局 - 从中心点向外扩散的面板"""
        panels = []
        
        # 中心点
        center_x = self.width / 2
        center_y = self.height / 2
        
        # 角色面板(中心)
        char_width = self.width * 0.4
        char_height = self.height * 0.4
        char_x = center_x - char_width / 2
        char_y = center_y - char_height / 2
        panels.append((char_x, char_y, char_width, char_height))
        
        # 表情面板(左上)
        expr_width = self.width * 0.35
        expr_height = self.height * 0.3
        expr_x = random.uniform(0, center_x - expr_width)
        expr_y = random.uniform(0, center_y - expr_height)
        panels.append((expr_x, expr_y, expr_width, expr_height))
        
        # 场景面板(右上)
        scene_width = self.width * 0.35
        scene_height = self.height * 0.3
        scene_x = random.uniform(center_x, self.width - scene_width)
        scene_y = random.uniform(0, center_y - scene_height)
        panels.append((scene_x, scene_y, scene_width, scene_height))
        
        # 物品面板(下方)
        item_width = self.width * 0.5
        item_height = self.height * 0.3
        item_x = random.uniform(0, self.width - item_width)
        item_y = random.uniform(center_y + char_height/2, self.height - item_height)
        panels.append((item_x, item_y, item_width, item_height))
        
        # 指定形状和旋转角度
        shapes = ["rect", "rect", "rect", "rect"]
        angles = [
            0,
            random.uniform(-15, 15),
            random.uniform(-15, 15),
            random.uniform(-15, 15)
        ]
        
        # 调整面板大小，添加边框
        adjusted_panels = self.adjust_bboxes_with_gaps(panels)
        
        return adjusted_panels, shapes, angles

    def dynamic_layout_style_5(self):
        """随机大小不规则布局 - 完全随机的面板大小和位置"""
        panels = []
        
        # 设置面板数量
        num_panels = 4
        
        # 计算可用空间
        used_space = set()
        grid_size = 20  # 网格尺寸
        grid_w = self.width // grid_size
        grid_h = self.height // grid_size
        
        # 针对每个面板创建随机尺寸和位置
        for i in range(num_panels):
            max_attempts = 100
            panel_placed = False
            
            for attempt in range(max_attempts):
                # 为每个面板类型设置合适的尺寸范围
                if i == 0:  # 角色面板 - 可能更高
                    panel_w = random.randint(grid_w // 4, grid_w // 2)
                    panel_h = random.randint(grid_h // 3, grid_h // 1)
                elif i == 1:  # 表情面板 - 较宽
                    panel_w = random.randint(grid_w // 3, grid_w // 2)
                    panel_h = random.randint(grid_h // 4, grid_h // 3)
                else:  # 场景和物品
                    panel_w = random.randint(grid_w // 4, grid_w // 2)
                    panel_h = random.randint(grid_h // 4, grid_h // 2)
                
                # 随机位置
                panel_x = random.randint(0, grid_w - panel_w)
                panel_y = random.randint(0, grid_h - panel_h)
                
                # 检查是否与现有面板重叠
                overlap = False
                panel_coords = set((x, y) for x in range(panel_x, panel_x + panel_w) 
                                for y in range(panel_y, panel_y + panel_h))
                
                if panel_coords.intersection(used_space):
                    overlap = True
                
                if not overlap:
                    used_space.update(panel_coords)
                    
                    # 转换回实际像素尺寸
                    real_x = panel_x * grid_size
                    real_y = panel_y * grid_size
                    real_w = panel_w * grid_size
                    real_h = panel_h * grid_size
                    
                    panels.append((real_x, real_y, real_w, real_h))
                    panel_placed = True
                    break
            
            # 如果无法放置，创建备用位置
            if not panel_placed:
                if i == 0:  # 角色
                    panels.append((0, 0, self.width//2, self.height//2))
                elif i == 1:  # 表情
                    panels.append((self.width//2, 0, self.width//2, self.height//3))
                elif i == 2:  # 场景
                    panels.append((0, self.height//2, self.width//2, self.height//2))
                else:  # 物品
                    panels.append((self.width//2, self.height//3, self.width//2, self.height*2//3))
        
        # 如果面板不足4个(可能由于冲突)，添加剩余面板
        while len(panels) < 4:
            missing_index = len(panels)
            if missing_index == 0:  # 角色
                panels.append((0, 0, self.width//2, self.height//2))
            elif missing_index == 1:  # 表情
                panels.append((self.width//2, 0, self.width//2, self.height//3))
            elif missing_index == 2:  # 场景
                panels.append((0, self.height//2, self.width//2, self.height//2))
            else:  # 物品
                panels.append((self.width//2, self.height//3, self.width//2, self.height*2//3))
        
        # 指定形状和旋转角度
        shapes = ["rect", "rect", "rect", "rect"]
        angles = [
            random.uniform(-10, 10),
            random.uniform(-10, 10),
            random.uniform(-10, 10),
            random.uniform(-10, 10)
        ]
        
        # 调整面板大小，添加边框
        adjusted_panels = self.adjust_bboxes_with_gaps(panels)
        
        return adjusted_panels, shapes, angles

    def dynamic_layout_style_6(self):
        """Z字形布局 - 面板按Z形排列"""
        panels = []
        
        # 将画布分为Z形的四个区域
        top_height = self.height * random.uniform(0.3, 0.4)
        bottom_height = self.height * random.uniform(0.3, 0.4)
        middle_height = self.height - top_height - bottom_height
        
        left_width = self.width * random.uniform(0.4, 0.6)
        right_width = self.width - left_width
        
        # 角色面板(左上)
        char_x = 0
        char_y = 0
        char_w = left_width
        char_h = top_height
        panels.append((char_x, char_y, char_w, char_h))
        
        # 表情面板(右上)
        expr_x = left_width
        expr_y = 0
        expr_w = right_width
        expr_h = top_height
        panels.append((expr_x, expr_y, expr_w, expr_h))
        
        # 场景面板(中间 - 斜跨)
        scene_x = left_width - self.width * 0.2
        scene_y = top_height
        scene_w = self.width * 0.4
        scene_h = middle_height
        panels.append((scene_x, scene_y, scene_w, scene_h))
        
        # 物品面板(底部)
        item_x = 0
        item_y = top_height + middle_height
        item_w = self.width
        item_h = bottom_height
        panels.append((item_x, item_y, item_w, item_h))
        
        # 指定形状和旋转角度
        shapes = ["rect", "rect", "rect", "rect"]
        angles = [0, 0, random.uniform(-15, 15), 0]
        
        # 调整面板大小，添加边框
        adjusted_panels = self.adjust_bboxes_with_gaps(panels)
        
        return adjusted_panels, shapes, angles

    def dynamic_layout_style_7(self):
        """放射状布局 - 一个中心面板，其他面板围绕"""
        panels = []
        
        # 中心面板 - 角色
        center_width = self.width * 0.5
        center_height = self.height * 0.5
        center_x = (self.width - center_width) / 2
        center_y = (self.height - center_height) / 2
        panels.append((center_x, center_y, center_width, center_height))
        
        # 其他面板围绕中心
        # 上方面板 - 表情
        top_width = self.width * 0.4
        top_height = center_y * 0.9
        top_x = (self.width - top_width) / 2
        top_y = 0
        panels.append((top_x, top_y, top_width, top_height))
        
        # 左侧面板 - 场景
        left_width = center_x * 0.9
        left_height = self.height * 0.4
        left_x = 0
        left_y = (self.height - left_height) / 2
        panels.append((left_x, left_y, left_width, left_height))
        
        # 底部面板 - 物品
        bottom_width = self.width * 0.4
        bottom_height = (self.height - center_y - center_height) * 0.9
        bottom_x = (self.width - bottom_width) / 2
        bottom_y = center_y + center_height
        panels.append((bottom_x, bottom_y, bottom_width, bottom_height))
        
        # 指定形状
        shapes = ["rect", "rect", "rect", "rect"]
        
        # 设置旋转角度 - 中心面板不旋转，其他面板轻微旋转
        angles = [
            0,  # 中心面板
            random.uniform(-5, 5),  # 上方面板
            random.uniform(-5, 5),  # 左侧面板
            random.uniform(-5, 5),  # 底部面板
        ]
        
        # 调整面板大小，添加边框
        adjusted_panels = self.adjust_bboxes_with_gaps(panels)
        
        return adjusted_panels, shapes, angles

    def dynamic_layout_style_8(self):
        """碎片式布局 - 将画布分割成不规则的碎片"""
        panels = []
        
        # 基准点和随机偏移
        base_w = self.width / 2
        base_h = self.height / 2
        offset_range = min(self.width, self.height) * 0.15
        
        # 创建不规则分割点
        split_x = base_w + random.uniform(-offset_range, offset_range)
        split_y = base_h + random.uniform(-offset_range, offset_range)
        
        # 角色面板(主要区域)
        char_choice = random.choice(["top_left", "top_right", "bottom_left", "bottom_right"])
        
        if char_choice == "top_left":
            # 角色在左上
            panels.append((0, 0, split_x, split_y))
            
            # 表情在右上
            panels.append((split_x, 0, self.width - split_x, split_y * 0.6))
            
            # 场景在左下
            panels.append((0, split_y, split_x * 0.6, self.height - split_y))
            
            # 物品在右下
            panels.append((split_x, split_y * 0.6, self.width - split_x, self.height - split_y * 0.6))
        
        elif char_choice == "top_right":
            # 角色在右上
            panels.append((split_x, 0, self.width - split_x, split_y))
            
            # 表情在左上
            panels.append((0, 0, split_x, split_y * 0.6))
            
            # 场景在右下
            panels.append((split_x, split_y, self.width - split_x, self.height - split_y))
            
            # 物品在左下
            panels.append((0, split_y * 0.6, split_x, self.height - split_y * 0.6))
        
        elif char_choice == "bottom_left":
            # 角色在左下
            panels.append((0, split_y, split_x, self.height - split_y))
            
            # 表情在左上
            panels.append((0, 0, split_x, split_y))
            
            # 场景在右下
            panels.append((split_x, split_y, self.width - split_x, self.height - split_y))
            
            # 物品在右上
            panels.append((split_x, 0, self.width - split_x, split_y))
        
        else:  # bottom_right
            # 角色在右下
            panels.append((split_x, split_y, self.width - split_x, self.height - split_y))
            
            # 表情在右上
            panels.append((split_x, 0, self.width - split_x, split_y))
            
            # 场景在左下
            panels.append((0, split_y, split_x, self.height - split_y))
            
            # 物品在左上
            panels.append((0, 0, split_x, split_y))
        
        # 指定形状
        shapes = ["rect", "rect", "rect", "rect"]
        
        # 随机旋转角度
        angles = [
            0,  # 角色不旋转
            random.uniform(-8, 8),
            random.uniform(-8, 8),
            random.uniform(-8, 8)
        ]
        
        # 调整面板大小，添加边框
        adjusted_panels = self.adjust_bboxes_with_gaps(panels)
        
        return adjusted_panels, shapes, angles

    def dynamic_layout_style_9(self):
        """多层次重叠布局 - 带有部分重叠效果"""
        panels = []
        
        # 基本尺寸
        panel_w = self.width * 0.6
        panel_h = self.height * 0.6
        
        # 角色(主面板)
        char_x = (self.width - panel_w) / 2
        char_y = (self.height - panel_h) / 2
        panels.append((char_x, char_y, panel_w, panel_h))
        
        # 表情(较小面板，位于左上)
        expr_size = min(self.width, self.height) * 0.3
        expr_x = char_x - expr_size * 0.3
        expr_y = char_y - expr_size * 0.3
        panels.append((expr_x, expr_y, expr_size, expr_size))
        
        # 场景(较小面板，位于右上)
        scene_size = min(self.width, self.height) * 0.3
        scene_x = char_x + panel_w - scene_size * 0.7
        scene_y = char_y - scene_size * 0.3
        panels.append((scene_x, scene_y, scene_size, scene_size))
        
        # 物品(较小面板，位于底部)
        item_width = panel_w * 0.7
        item_height = min(self.width, self.height) * 0.25
        item_x = char_x + (panel_w - item_width) / 2
        item_y = char_y + panel_h - item_height * 0.4
        panels.append((item_x, item_y, item_width, item_height))
        
        # 指定形状
        shapes = ["rect", "circle", "rect", "rect"]
        
        # 随机旋转角度
        angles = [
            0,  # 主面板不旋转
            random.uniform(-15, 15),
            random.uniform(-15, 15),
            random.uniform(-5, 5)
        ]
        
        # 调整面板大小，添加边框
        adjusted_panels = self.adjust_bboxes_with_gaps(panels)
        
        return adjusted_panels, shapes, angles

    def adjust_bboxes_with_gaps(self, rectangles):
        """调整面板以添加边框间距"""
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
            
        # 确保所有值都是整数
        adjusted_bboxes = [(int(x), int(y), int(w), int(h)) for x, y, w, h in adjusted_bboxes]
        
        return adjusted_bboxes

    def draw_mask(self):
        """绘制布局遮罩"""
        bboxes = []
        for i, bbox in enumerate(self.rectangles):
            if self.shapes[i] == "circle":
                # 圆形面板处理为其外接正方形
                center_x = bbox[0] + bbox[2] / 2
                center_y = bbox[1] + bbox[3] / 2
                radius = min(bbox[2], bbox[3]) / 2
                bboxes.append((
                    center_x - radius, 
                    center_y - radius, 
                    center_x + radius, 
                    center_y + radius
                ))
            else:
                # 标准矩形面板
                bboxes.append((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
                
        scale_factor = 4
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
                "manga_style": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 9,
                    "step": 1,
                    "display": "漫画风格(-1为随机)"
                }),
                "add_onomatopoeia": ("BOOLEAN", {
                    "default": False,
                    "display": "添加拟声词效果"
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

    def resize_image(self, image, target_width, target_height, img_type=None, angle=0):
        """将图像调整到目标尺寸，根据图像类型优化裁剪，并可选择旋转"""
        # 确保目标尺寸为整数
        target_width = int(target_width)
        target_height = int(target_height)
        
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
        
        # 应用旋转
        if angle != 0:
            # 旋转图像
            resized = resized.rotate(angle, expand=True, resample=Image.BICUBIC)
            
            # 重新调整尺寸
            orig_w, orig_h = resized.size
            if orig_w > target_width or orig_h > target_height:
                # 调整比例以适应目标尺寸
                scale = min(target_width / orig_w, target_height / orig_h)
                new_w = int(orig_w * scale)
                new_h = int(orig_h * scale)
                resized = resized.resize((new_w, new_h), Image.LANCZOS)
                
                # 居中放置
                if new_w < target_width or new_h < target_height:
                    temp = Image.new("RGBA", (target_width, target_height), (0, 0, 0, 0))
                    paste_x = (target_width - new_w) // 2
                    paste_y = (target_height - new_h) // 2
                    temp.paste(resized, (paste_x, paste_y))
                    resized = temp
            
        return resized
    
    def add_manga_effects(self, canvas, onomatopoeia=False):
        """添加漫画风格效果，如速度线和拟声词"""
        draw = ImageDraw.Draw(canvas)
        w, h = canvas.size
        
        # 添加速度线
        line_count = random.randint(10, 30)
        line_color = (220, 220, 220, 180)  # 半透明白色
        
        # 随机选择速度线方向和位置
        center_x = w // 2 + random.randint(-w//4, w//4)
        center_y = h // 2 + random.randint(-h//4, h//4)
        
        for _ in range(line_count):
            # 速度线从中心向外
            angle = random.uniform(0, 2 * math.pi)
            length = random.uniform(w * 0.1, w * 0.4)
            thickness = random.randint(1, 3)
            
            end_x = center_x + length * math.cos(angle)
            end_y = center_y + length * math.sin(angle)
            
            draw.line([(center_x, center_y), (end_x, end_y)], fill=line_color, width=thickness)
        
        # 添加拟声词(可选)
        if onomatopoeia:
            # 尝试加载字体
            try:
                font_size = int(min(w, h) * 0.08)
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                # 如果没有找到字体，使用默认字体
                font = None
            
            # 选择随机拟声词
            effects = ["BOOM!", "CRASH!", "POW!", "BANG!", "WHAM!", "BAM!", "SMASH!", "BLAST!"]
            effect_text = random.choice(effects)
            
            # 找到放置位置(不覆盖中心内容)
            text_x = random.randint(w//10, w*9//10)
            text_y = random.randint(h//10, h*9//10)
            
            # 添加文本阴影效果
            shadow_offset = max(1, font_size // 15)
            draw.text((text_x + shadow_offset, text_y + shadow_offset), 
                      effect_text, fill=(0, 0, 0, 200), font=font)
            
            # 添加文本
            draw.text((text_x, text_y), effect_text, fill=(255, 255, 255, 255), font=font)
        
        return canvas

    def collage(self, character_image, expression_image, scene_image, item_image, 
                canvas_width, canvas_height, border_width, rounded_rect_radius,
                background_color, manga_style, add_onomatopoeia, seed):
        
        # 将四张单图像组合成一个列表，顺序很重要
        # 顺序为：角色(0), 表情(1), 场景(2), 物品(3)
        images = [character_image[0], expression_image[0], scene_image[0], item_image[0]]
        img_types = ["character", "expression", "scene", "item"]
        
        # 创建漫画风格布局生成器
        style_index = manga_style if manga_style >= 0 else None
        layout_gen = MangaStyleLayoutGenerator(
            width=canvas_width,
            height=canvas_height,
            border_width=border_width,
            r=rounded_rect_radius,
            seed=seed,
            style_index=style_index
        )

        canvas = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))
        mask = Image.new("L", (canvas_width, canvas_height), 0)
        mask_draw = ImageDraw.Draw(mask)
        
        # 绘制每个图像到指定位置
        for i, img_tensor in enumerate(images):
            img = tensor2pil(img_tensor).convert("RGBA")
            img_x, img_y = layout_gen.rectangles[i][0], layout_gen.rectangles[i][1]
            img_w, img_h = layout_gen.rectangles[i][2], layout_gen.rectangles[i][3]
            
            # 获取适合此面板的形状和旋转角度
            panel_shape = layout_gen.shapes[i]
            rotation_angle = layout_gen.rotation_angles[i]
            
            # 根据图像类型和旋转角度调整大小
            resized_img = self.resize_image(
                img, img_w, img_h, img_types[i], rotation_angle)
            
            # 如果是圆形面板，需要特殊处理
            if panel_shape == "circle":
                # 创建圆形蒙版
                size = min(img_w, img_h)
                circle_mask = Image.new("L", (img_w, img_h), 0)
                circle_draw = ImageDraw.Draw(circle_mask)
                circle_draw.ellipse([(0, 0), (img_w, img_h)], fill=255)
                
                # 调整圆形图像
                circle_img = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
                temp_img = resized_img.resize((img_w, img_h), Image.LANCZOS)
                circle_img.paste(temp_img, mask=circle_mask)
                resized_img = circle_img
            else:
                # 矩形面板加圆角
                resized_img = add_corners_to_image(resized_img, rounded_rect_radius)
            
            # 将图像粘贴到画布上 - 确保坐标是整数
            canvas.paste(resized_img, (int(img_x), int(img_y)), mask=resized_img.split()[3])
            
            # 更新蒙版 - 确保所有坐标都是整数
            if panel_shape == "circle":
                # 圆形面板
                mask_draw.ellipse(
                    [int(img_x), int(img_y), int(img_x + img_w), int(img_y + img_h)], fill=255)
            else:
                # 矩形面板
                mask_draw.rectangle(
                    [int(img_x), int(img_y), int(img_x + img_w), int(img_y + img_h)], fill=255)

        # 添加漫画风格效果
        canvas = self.add_manga_effects(canvas, onomatopoeia=add_onomatopoeia)

        # 创建最终图像
        final_image = Image.new("RGB", (canvas_width, canvas_height), background_color)
        final_image.paste(canvas, mask=canvas.split()[3])
        
        return (pil2tensor(final_image), pil2tensor(mask))

NODE_CLASS_MAPPINGS = {
    "PIP_GridComic": PIP_GridComic,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PIP_GridComic": "PIP 格漫拼图",
}
