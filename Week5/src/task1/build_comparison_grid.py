import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


GRID_CELL_SIZE = 512
GRID_PADDING = 24
GRID_HEADER_HEIGHT = 220
GRID_LABEL_HEIGHT = 150
GRID_MAX_SOURCE_SIZE = 768
GRID_TITLE_FONT_SIZE = 28
GRID_PROMPT_FONT_SIZE = 20
GRID_LABEL_FONT_SIZE = 18
GRID_TIME_FONT_SIZE = 18


def _wrap_text(text: str, draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, max_width: int):
    words = text.split()
    if not words:
        return [""]

    lines = []
    current_line = words[0]
    for word in words[1:]:
        trial_line = f"{current_line} {word}"
        if draw.textbbox((0, 0), trial_line, font=font)[2] <= max_width:
            current_line = trial_line
        else:
            lines.append(current_line)
            current_line = word
    lines.append(current_line)
    return lines


def _load_grid_font(size: int):
    font_candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for font_path in font_candidates:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default()


def _line_height(draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, sample_text: str = "Ag") -> int:
    bbox = draw.textbbox((0, 0), sample_text, font=font)
    return bbox[3] - bbox[1]


def _build_comparison_grid(saved_images, prompt: str, seed: int, output_dir: Path, prompt_slug: str):
    if not saved_images:
        return None

    title_font = _load_grid_font(GRID_TITLE_FONT_SIZE)
    prompt_font = _load_grid_font(GRID_PROMPT_FONT_SIZE)
    label_font = _load_grid_font(GRID_LABEL_FONT_SIZE)
    time_font = _load_grid_font(GRID_TIME_FONT_SIZE)
    cols = min(3, len(saved_images))
    rows = math.ceil(len(saved_images) / cols)
    cell_width = GRID_CELL_SIZE
    cell_height = GRID_CELL_SIZE + GRID_LABEL_HEIGHT
    canvas_width = GRID_PADDING * (cols + 1) + cell_width * cols
    canvas_height = GRID_HEADER_HEIGHT + \
        GRID_PADDING * (rows + 1) + cell_height * rows

    canvas = Image.new("RGB", (canvas_width, canvas_height),
                       color=(245, 244, 239))
    draw = ImageDraw.Draw(canvas)

    header_x = GRID_PADDING
    header_width = canvas_width - 2 * GRID_PADDING
    prompt_lines = _wrap_text(
        f"Prompt: {prompt}", draw, prompt_font, header_width)
    draw.text((header_x, 20), f"Diffusion Model Comparison | seed={seed}", fill=(
        30, 30, 30), font=title_font)

    current_y = 20 + _line_height(draw, title_font) + 18
    for line in prompt_lines:
        draw.text((header_x, current_y), line,
                  fill=(60, 60, 60), font=prompt_font)
        current_y += _line_height(draw, prompt_font) + 8

    for index, item in enumerate(saved_images):
        row = index // cols
        col = index % cols
        origin_x = GRID_PADDING + col * (cell_width + GRID_PADDING)
        origin_y = GRID_HEADER_HEIGHT + GRID_PADDING + \
            row * (cell_height + GRID_PADDING)

        image = item["image"].copy().convert("RGB")
        image.thumbnail(
            (GRID_MAX_SOURCE_SIZE, GRID_MAX_SOURCE_SIZE), Image.Resampling.LANCZOS)
        image.thumbnail((cell_width, GRID_CELL_SIZE))
        image_x = origin_x + (cell_width - image.width) // 2
        image_y = origin_y + (GRID_CELL_SIZE - image.height) // 2

        draw.rectangle(
            [origin_x, origin_y, origin_x + cell_width, origin_y + GRID_CELL_SIZE],
            fill=(255, 255, 255),
            outline=(210, 210, 210),
            width=2,
        )
        canvas.paste(image, (image_x, image_y))

        label_text = item.get("display_name", item["model_name"])
        if item["model_name"] != "reference_image":
            label_text = f"model: {label_text}"
        model_lines = _wrap_text(
            label_text, draw, label_font, cell_width - 12)
        label_y = origin_y + GRID_CELL_SIZE + 10
        for line in model_lines[:3]:
            draw.text((origin_x + 6, label_y), line,
                      fill=(40, 40, 40), font=label_font)
            label_y += _line_height(draw, label_font) + 6
        if item.get("generation_time_s") is not None:
            draw.text(
                (origin_x + 6, label_y + 4),
                f"time: {item['generation_time_s']:.2f}s",
                fill=(70, 70, 70),
                font=time_font,
            )
            draw.text(
                (origin_x + 6, label_y + _line_height(draw, time_font) + 10),
                f"steps: {item['num_inference_steps']}",
                fill=(70, 70, 70),
                font=time_font,
            )

    grid_path = output_dir / f"comparison__seed_{seed}__{prompt_slug}.png"
    canvas.save(grid_path)
    print(f"Saved comparison grid -> {grid_path}")
    return grid_path
