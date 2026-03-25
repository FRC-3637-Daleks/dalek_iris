def get_3d_coords(pixel_x, pixel_y, w, h):
    import math
    angle_rad = math.radians(0.0000)
    cx, cy = 998.6, 166.4

    # Shift to rotation center
    nx = pixel_x - cx
    ny = pixel_y - cy

    # Rotate
    rx = (pixel_x - (w/2))/(w/2)
    ry = (pixel_y)/(h)

    # 2. Calculate Distance Z (feet)
    z = 3.82850530 / (ry + 0.07729443) + -0.74802465

    # 3. Calculate Horizontal X (feet)
    # Formula: X = Z * (pixel_x * m + b)
    x = z * (rx * 0.65112527)

    return x, z