import math


def calculate_intersection_X(xk, yk, x0, y0, v0, theta, x, y, g=9.81):
    """
    计算箭与瞄镜-窥孔连线交点的真实水平距离 X

    参数:
    xk, yk: 窥孔在设定坐标系中的坐标
    x0, y0: 箭头初始坐标
    v0: 出弓速度 (m/s)
    theta: 俯仰角 (弧度)
    x, y: 瞄镜在设定坐标系中的坐标
    g: 重力加速度 (默认 9.81 m/s²)

    返回:
    X: 真实水平距离，若无解返回 None
    """
    # 1. 坐标转换到真实水平坐标系
    def rotate_coords(x, y, theta):
        X = x * math.cos(theta) - y * math.sin(theta)
        Y = x * math.sin(theta) + y * math.cos(theta)
        return X, Y

    # 转换所有坐标
    X_m, Y_m = rotate_coords(x, y, theta)
    X_k, Y_k = rotate_coords(xk, yk, theta)
    X_0, Y_0 = rotate_coords(x0, y0, theta)

    # 2. 处理瞄镜-窥孔连线
    if abs(X_m - X_k) > 1e-9:  # 非垂直直线
        # 计算斜率和截距
        m = (Y_m - Y_k) / (X_m - X_k)

        # 构建二次方程 At² + Bt + C = 0
        A = -0.5 * g
        B = v0 * math.sin(theta) - m * v0 * math.cos(theta)
        C = Y_0 - Y_k - m*(X_0 - X_k)

        # 计算判别式
        discriminant = B**2 - 4*A*C

        if discriminant < 0:
            return None  # 无实数解

        # 求根
        sqrt_d = math.sqrt(discriminant)
        t1 = (-B + sqrt_d) / (2*A)
        t2 = (-B - sqrt_d) / (2*A)

        # 选择有效时间解
        valid_ts = [t for t in [t1, t2] if t >= 1e-6]

        if not valid_ts:
            return None

        # 计算对应的X值
        t = min(valid_ts)
        X = X_0 + v0 * math.cos(theta) * t
        return X

    else:  # 垂直直线 X = X_m = X_k
        # 解 X_0 + v0*cosθ*t = X_k
        cos_theta = math.cos(theta)
        if abs(cos_theta) < 1e-9:
            return None  # 初始速度方向垂直，无法水平移动

        t = (X_k - X_0) / (v0 * cos_theta)
        if t < 0:
            return None

        # 验证Y是否在瞄镜和窥孔之间
        Y_t = Y_0 + v0 * math.sin(theta) * t - 0.5 * g * t**2
        if min(Y_k, Y_m) <= Y_t <= max(Y_k, Y_m):
            return X_k  # 因为X_k = X_m
        else:
            return None


# 示例用法
xk, yk = 0.5, 0.1  # 窥孔坐标
x0, y0 = 0.2, 0.0  # 箭头初始位置
v0 = 50  # 50 m/s
theta = math.radians(10)  # 10度俯仰角
x, y = 0.2, 0.2  # 瞄镜坐标

X = calculate_intersection_X(xk, yk, x0, y0, v0, theta, x, y)
if X is not None:
    print(f"交点真实水平距离 X = {X:.3f} 米")
else:
    print("轨迹不相交")
