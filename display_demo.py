from PIL import Image, ImageDraw, ImageFont

def create_galgame_dialogue(image_path, text, character_name, output_path):
    character_name = "[{}]".format(character_name)
    text = "「{}」".format(text)

    # 加载背景图片
    bg_image = Image.open(image_path)
    bg_width, bg_height = bg_image.size

    # 创建对话框区域
    dialog_height = bg_height // 4  # 对话框高度为图片高度的四分之一

    # 加载自定义素材图片作为对话框背景
    dialog_texture = Image.open(r".\_resources\srcs\对话框\Frame_湊.png")  # 替换为你的素材图片路径
    dialog_texture = dialog_texture.resize((bg_width, dialog_height))  # 调整素材大小
    dialog_texture = dialog_texture.convert("RGBA")  # 确保素材是RGBA模式

    # 设置素材图片的透明度（半透明）
    alpha = 128  # 透明度值
    dialog_texture = dialog_texture.copy()
    dialog_texture.putalpha(alpha)  # 设置透明度

    # 合成背景和对话框
    combined_image = Image.new("RGBA", (bg_width, bg_height), (0, 0, 0, 0))
    combined_image.paste(bg_image, (0, 0))  # 粘贴背景图
    combined_image.paste(dialog_texture, (0, bg_height - dialog_height), dialog_texture)  # 粘贴对话框
    draw = ImageDraw.Draw(combined_image)

    # 加载字体  from: https://www.fonts.net.cn/fonts-zh-1.html
    try:
        font_name = ImageFont.truetype(r".\_resources\font\AaYuanWeiTuSi-2.ttf", 40)
        font_text = ImageFont.truetype(r".\_resources\font\SanJiHuaChaoTi-Cu-2.ttf", 30)
    except IOError:
        font_name = ImageFont.load_default()
        font_text = ImageFont.load_default()

    # 加载头像图片
    avatar = Image.open("./_resources/AI/avatar.png")  # 替换为头像图片路径
    avatar = avatar.convert("RGBA")  # 确保头像图片是RGBA模式

    # 调整头像大小（略微大于对话框高度）
    avatar_height = int(dialog_height * 1.2)  # 头像高度为对话框高度的1.2倍
    avatar_width = int(avatar_height * (avatar.width / avatar.height))  # 保持宽高比
    avatar = avatar.resize((avatar_width, avatar_height))

    # 计算头像位置（左侧，略微超出对话框上边界）
    avatar_x = 20  # 距离左侧的间距
    avatar_y = bg_height - dialog_height - int(avatar_height * 0.2)  # 略微超出对话框上边界

    # 将头像粘贴到合成图像中（仅覆盖对话框区域）
    combined_image.paste(avatar, (avatar_x, avatar_y), avatar)  # 使用透明度遮罩

    # 绘制对话内容
    white_color = (255, 255, 255, 255)  # 白色文字
    pink_color = (255, 182, 193, 128)  # 粉色文字
    padding = 20  # 文字与对话框边缘的间距
    max_text_width = bg_width - 2 * padding - avatar_width - 100  # 最大文本宽度（考虑头像宽度）

    # 自动换行函数
    def wrap_text(text, font, max_width):
        lines = [character_name]
        words = text.split()
        print(words)
        current_line = ""
        for word in text:
            test_line = f"{current_line} {word}".strip()
            test_bbox = draw.textbbox((0, 0), test_line, font=font)
            test_width = test_bbox[2] - test_bbox[0]
            if test_width <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        return lines

    # 对文本进行自动换行
    wrapped_lines = wrap_text(text, font_text, max_text_width)

    # 绘制名字和每一行文字
    font = font_name
    text_color = pink_color
    y_text = bg_height - dialog_height + 50  # 文字起始位置（距离对话框顶部的间距）
    for line in wrapped_lines:
        line_bbox = draw.textbbox((0, 0), line, font=font)
        line_height = line_bbox[3] - line_bbox[1]
        x_text = avatar_x + avatar_width + 20  # 文字距离头像右侧的间距
        draw.text((x_text, y_text), line, font=font, fill=text_color)
        y_text += line_height + 10  # 换行时增加行间距
        font = font_text
        text_color = white_color
        

    # 保存图片
    combined_image = combined_image.convert("RGB")
    combined_image.save(output_path)
    print(f"对话图片已保存到 {output_path}")


# 使用示例
create_galgame_dialogue(
    image_path="./_resources/AI/bg_up.png",  # 背景图路径
    text="你好！我是你的人工智能课程助教，有任何需要帮助的都可以找我！",
    character_name="野兽先辈",
    output_path="outputs/display_demo.jpg"  # 输出路径
)