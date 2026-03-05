from typing import Literal

import cv2
import numpy as np


def select_point_from_image(
    image: np.ndarray,
    source_cspace: Literal["rgb", "bgr"] = "rgb",
    format: Literal["ij", "xy"] = "xy",
):
    """Select a point from an image"""
    ix, iy = 0, 0

    def select_point(event, x, y, flags, param):
        nonlocal ix, iy
        if event == cv2.EVENT_LBUTTONDBLCLK:
            ix, iy = x, y

    bgr_image = image
    if source_cspace == "rgb":
        bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_RGB2BGR)

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", select_point)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if format == "ij":
        iy, ix = ix, iy

    coords = np.array([ix, iy])
    return coords
