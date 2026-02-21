def get_3d_coords(pixel_x, pixel_y):
    # 1. Rotate point to correct camera tilt
    import math
    angle_rad = math.radians(0.0200)
    cx, cy = 998.6, 166.4

    # Shift to rotation center
    nx = pixel_x - cx
    ny = pixel_y - cy

    # Rotate
    rx = nx * math.cos(angle_rad) - ny * math.sin(angle_rad) + cx
    ry = nx * math.sin(angle_rad) + ny * math.cos(angle_rad) + cy

    # 2. Calculate Distance Z (feet)
    z = 0.00010640 * (ry**2) + -0.10211633 * ry + 26.47698986

    # 3. Calculate Horizontal X (feet)
    # Formula: X = Z * (pixel_x * m + b)
    x = z * (rx * 0.00065063 + (-0.20293485))

    return x, z