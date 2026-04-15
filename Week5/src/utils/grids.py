from math import ceil, sqrt

from PIL import Image, ImageDraw, ImageFont


def make_image_grid(images, background_color=(255, 255, 255), title=None):
    if not images:
        raise ValueError("Cannot create a grid from an empty image list.")

    cols = ceil(sqrt(len(images)))
    rows = ceil(len(images) / cols)

    width, height = images[0].size
    
    # Calculate space for title if provided
    title_height = 0
    if title:
        title_height = 80  # Reserve space for title
    
    grid = Image.new("RGB", (cols * width, rows * height + title_height), color=background_color)

    # Add title text if provided
    if title:
        draw = ImageDraw.Draw(grid)
        try:
            # Try to use a larger font, fall back to default if not available
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        except (OSError, IOError):
            font = ImageFont.load_default()
        
        # Wrap text if it's too long
        max_chars_per_line = 80
        lines = []
        words = title.split()
        current_line = []
        
        for word in words:
            current_line.append(word)
            if len(" ".join(current_line)) > max_chars_per_line:
                current_line.pop()
                lines.append(" ".join(current_line))
                current_line = [word]
        if current_line:
            lines.append(" ".join(current_line))
        
        # Draw each line of text
        y_offset = 5
        for line in lines:
            draw.text((10, y_offset), line, fill=(0, 0, 0), font=font)
            y_offset += 30

    for index, image in enumerate(images):
        x = (index % cols) * width
        y = (index // cols) * height + title_height
        grid.paste(image, (x, y))

    return grid
