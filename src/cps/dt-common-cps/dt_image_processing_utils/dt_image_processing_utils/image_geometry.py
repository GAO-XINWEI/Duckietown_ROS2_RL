#!/usr/bin/env python3

import numpy as np

from typing import Tuple

__all__ = [
    'find_normal',
    'normalize_lines'
]


def find_normal(map, lines):
    """Calculates the centers of the line segments and their normals.
    Args:
        map (:obj:`numpy array`):  binary image with the regions of the
            image that fall in a given color range lines
        (:obj:`numpy array`):
            An ``Nx4`` array where each row represents a line.
            If no lines were detected, returns an empty list.
    Returns:
        :obj:`tuple`: a tuple containing:
             * :obj:`numpy array`: An ``Nx2`` array where each row represents
                the center point of a line. If no lines were detected returns
                an empty list.
             * :obj:`numpy array`: An ``Nx2`` array where each row represents
             the normal of a line. If no lines were detected returns an empty list.
    """
    normals = []
    centers = []
    if len(lines) > 0:
        length = np.sum((lines[:, 0:2] - lines[:, 2:4]) ** 2, axis=1, keepdims=True) ** 0.5
        dx = 1.0 * (lines[:, 3:4] - lines[:, 1:2]) / length
        dy = 1.0 * (lines[:, 0:1] - lines[:, 2:3]) / length

        centers = np.hstack([(lines[:, 0:1] + lines[:, 2:3]) / 2, (lines[:, 1:2] + lines[:, 3:4]) / 2])
        x3 = (centers[:, 0:1] - 3.0 * dx).astype("int")
        y3 = (centers[:, 1:2] - 3.0 * dy).astype("int")
        x4 = (centers[:, 0:1] + 3.0 * dx).astype("int")
        y4 = (centers[:, 1:2] + 3.0 * dy).astype("int")

        np.clip(x3, 0, map.shape[1] - 1, out=x3)
        np.clip(y3, 0, map.shape[0] - 1, out=y3)
        np.clip(x4, 0, map.shape[1] - 1, out=x4)
        np.clip(y4, 0, map.shape[0] - 1, out=y4)

        flag_signs = (np.logical_and(map[y3, x3] > 0, map[y4, x4] == 0)).astype("int") * 2 - 1
        normals = np.hstack([dx, dy]) * flag_signs

    return centers, normals


def normalize_lines(lines: np.array, cutoff: int, image_size: Tuple[int, int]):
    """Remove the offset in coordinates coming from the removing of the top part
        of the image.
    :param lines: Each row in the array represents a single line. Each row
        must have 4 values, representing (x1, y1, x2, y2).
    :type lines: np.array
    :param cutoff: _description_
    :type cutoff: int
    :param image_size: _description_
    :type image_size: Tuple[int, int]
    :raises ValueError: _description_
    """
    if lines.shape[1] != 4:
        raise ValueError("Each row must have 4 values.")

    arr_cutoff = np.array([0, cutoff, 0, cutoff])
    arr_ratio = np.array([
        1.0/image_size[1],
        1.0/image_size[0],
        1.0/image_size[1],
        1.0/image_size[0]])

    return (lines + arr_cutoff) * arr_ratio
