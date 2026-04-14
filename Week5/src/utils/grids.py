from math import ceil, sqrt

from PIL import Image


def make_image_grid(images, background_color=(255, 255, 255)):
    if not images:
        raise ValueError("Cannot create a grid from an empty image list.")

    cols = ceil(sqrt(len(images)))
    rows = ceil(len(images) / cols)

    width, height = images[0].size
    grid = Image.new("RGB", (cols * width, rows * height), color=background_color)

    for index, image in enumerate(images):
        x = (index % cols) * width
        y = (index // cols) * height
        grid.paste(image, (x, y))

    return grid
