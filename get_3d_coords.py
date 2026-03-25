def get_3d_coords(pixel_x, pixel_y, w, h):
    import math
    angle_rad = math.radians(0.0000)
    cx, cy = 998.6, 166.4

    # Shift to rotation center
    nx = pixel_x - cx
    ny = pixel_y - cy

    # Rotate
    rx = (nx * math.cos(angle_rad) - ny * math.sin(angle_rad) + cx - w/2)/(w/2)
    ry = (nx * math.sin(angle_rad) + ny * math.cos(angle_rad) + cy)/h

    # 2. Calculate Distance Z (feet)
    z = 3.82850530 / (ry + 0.07729443) + -2.0802465

    # 3. Calculate Horizontal X (feet)
    # Formula: X = Z * (pixel_x * m + b)
    x = z * (rx * 0.50112527)

    return x, z