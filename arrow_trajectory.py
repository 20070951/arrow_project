import math
import matplotlib.pyplot as plt
import numpy as np
from config import ARROW_CONFIG


def calculate_intersection_distance(aim_x, aim_y, return_all=False):
    """
    计算箭的轨迹与窥孔和瞄镜连线的交点距离

    参数:
        aim_x: 瞄镜的x坐标（在箭轴坐标系中）
        aim_y: 瞄镜的y坐标（在箭轴坐标系中）
        return_all: 是否返回所有交点距离，默认False只返回第一个交点

    返回:
        如果return_all=False: 返回第一个交点的距离，如果没有交点则返回None
        如果return_all=True: 返回所有交点距离的列表，如果没有交点则返回空列表
    """
    # 从配置中获取参数
    xk = ARROW_CONFIG['xk']  # 窥孔x坐标 (S系)
    yk = ARROW_CONFIG['yk']  # 窥孔y坐标 (S系)
    v0 = ARROW_CONFIG['v0']  # 出弓速度
    theta = ARROW_CONFIG['theta']  # S系相对于H系的旋转角度
    x0 = ARROW_CONFIG['x0']  # 箭头初始x坐标 (S系)
    y0 = ARROW_CONFIG['y0']  # 箭头初始y坐标 (S系)
    g = ARROW_CONFIG['g']    # 重力加速度

    # 1. 坐标转换: S系 -> H系
    # 瞄镜坐标转换
    X_m = aim_x * math.cos(theta) - aim_y * math.sin(theta)
    Y_m = aim_x * math.sin(theta) + aim_y * math.cos(theta)

    # 窥孔坐标转换
    X_k = xk * math.cos(theta) - yk * math.sin(theta)
    Y_k = xk * math.sin(theta) + yk * math.cos(theta)

    # 2. 箭的初始位置转换
    X_0 = x0 * math.cos(theta) - y0 * math.sin(theta)
    Y_0 = x0 * math.sin(theta) + y0 * math.cos(theta)

    # 4. 瞄镜与窥孔的连线方程
    if abs(X_m - X_k) < 1e-10:  # 避免除以零
        # 特殊情况: 垂直线
        t = (X_k - X_0) / (v0 * math.cos(theta))
        if t < 0:  # 时间不能为负
            return [] if return_all else None

        Y_t = Y_0 + v0 * math.sin(theta) * t - 0.5 * g * t**2
        X_distance = X_k - X_0
        return [X_distance] if return_all else X_distance
    else:
        # 一般情况: 计算斜率
        m = (Y_m - Y_k) / (X_m - X_k)

        # 5. 联立方程求解时间t
        # Y_0 + v0*sin(theta)*t - 0.5*g*t² = Y_k + m*(X_0 + v0*cos(theta)*t - X_k)
        # 整理为标准形式: A*t² + B*t + C = 0
        A = -0.5 * g
        B = v0 * math.sin(theta) - m * v0 * math.cos(theta)
        C = Y_0 - Y_k - m * (X_0 - X_k)

        # 判别式
        discriminant = B**2 - 4*A*C

        if discriminant < 0:
            # 没有实数解，表示箭不会与瞄准线相交
            return [] if return_all else None

        # 求解二次方程的两个根
        t1 = (-B + math.sqrt(discriminant)) / (2*A)
        t2 = (-B - math.sqrt(discriminant)) / (2*A)

        # 选择有效的时间（正值）并计算对应的距离
        valid_times = sorted([t for t in [t1, t2] if t > 0])
        distances = []

        for t in valid_times:
            X_t = X_0 + v0 * math.cos(theta) * t
            X_distance = X_t - X_0
            distances.append(X_distance)

        if return_all:
            return distances
        else:
            # 返回第一个交点的距离，如果没有交点则返回None
            return distances[0] if distances else None


def plot_trajectory(aim_x, aim_y):
    """
    绘制箭的轨迹和瞄准线
    """
    # 从配置中获取参数
    xk = ARROW_CONFIG['xk']
    yk = ARROW_CONFIG['yk']
    v0 = ARROW_CONFIG['v0']
    theta = ARROW_CONFIG['theta']
    x0 = ARROW_CONFIG['x0']
    y0 = ARROW_CONFIG['y0']
    g = ARROW_CONFIG['g']

    # 坐标转换: S系 -> H系
    # 瞄镜坐标转换
    X_m = aim_x * math.cos(theta) - aim_y * math.sin(theta)
    Y_m = aim_x * math.sin(theta) + aim_y * math.cos(theta)

    # 窥孔坐标转换
    X_k = xk * math.cos(theta) - yk * math.sin(theta)
    Y_k = xk * math.sin(theta) + yk * math.cos(theta)

    # 箭的初始位置转换
    X_0 = x0 * math.cos(theta) - y0 * math.sin(theta)
    Y_0 = x0 * math.sin(theta) + y0 * math.cos(theta)

    # 估计飞行时间
    # 粗略估计：如果箭水平飞行时能到达的最远距离
    t_flight = 2 * v0 * math.sin(theta) / g if v0 * math.sin(theta) > 0 else 20

    # 创建时间点
    t = np.linspace(0, t_flight * 1.2, 1000)

    # 计算箭在H系中的轨迹
    X_traj = X_0 + v0 * math.cos(theta) * t
    Y_traj = Y_0 + v0 * math.sin(theta) * t - 0.5 * g * t**2

    # 计算瞄准线在H系中的点
    if abs(X_m - X_k) < 1e-10:  # 垂直线
        X_aim = np.ones(100) * X_k
        Y_aim = np.linspace(min(Y_k, Y_m) - 1, max(Y_k, Y_m) + 1, 100)
    else:
        X_aim = np.linspace(min(X_k, X_m) - 1, max(X_k, X_m) + 5, 100)
        m = (Y_m - Y_k) / (X_m - X_k)
        Y_aim = Y_k + m * (X_aim - X_k)

    # 计算所有可能的交点
    # 特殊情况: 垂直线
    if abs(X_m - X_k) < 1e-10:
        t_intersect = (X_k - X_0) / (v0 * math.cos(theta))
        if t_intersect > 0:
            X_i = X_k
            Y_i = Y_0 + v0 * math.sin(theta) * \
                t_intersect - 0.5 * g * t_intersect**2
            plt.plot(X_i, Y_i, 'rx', markersize=10, label='交点')
    else:
        # 一般情况
        m = (Y_m - Y_k) / (X_m - X_k)
        A = -0.5 * g
        B = v0 * math.sin(theta) - m * v0 * math.cos(theta)
        C = Y_0 - Y_k - m * (X_0 - X_k)

        discriminant = B**2 - 4*A*C
        if discriminant >= 0:
            t1 = (-B + math.sqrt(discriminant)) / (2*A)
            t2 = (-B - math.sqrt(discriminant)) / (2*A)

            # 找出有效的交点
            valid_times = sorted([t for t in [t1, t2] if t > 0])

            # 绘制所有有效交点
            for i, t_val in enumerate(valid_times):
                X_i = X_0 + v0 * math.cos(theta) * t_val
                Y_i = Y_0 + v0 * math.sin(theta) * t_val - 0.5 * g * t_val**2
                plt.plot(X_i, Y_i, 'rx', markersize=10,
                         label=f'交点{i+1} (t={t_val:.2f}s)')

    # 绘图（在H系中）
    plt.figure(figsize=(10, 6))
    plt.plot(X_traj, Y_traj, 'r-', label='箭的轨迹')
    plt.plot(X_aim, Y_aim, 'b--', label='瞄准线')
    plt.plot(X_k, Y_k, 'go', label='窥孔位置')
    plt.plot(X_m, Y_m, 'mo', label='瞄镜位置')
    plt.plot(X_0, Y_0, 'ko', label='箭头初始位置')

    # 计算所有交点距离
    distances = calculate_intersection_distance(aim_x, aim_y, return_all=True)
    if distances:
        plt.title(
            f'箭的轨迹与瞄准线 (俯仰角: {math.degrees(theta):.1f}度, 交点距离: {", ".join([f"{d:.2f}米" for d in distances])})')
    else:
        plt.title(f'箭的轨迹与瞄准线 (俯仰角: {math.degrees(theta):.1f}度, 无交点)')

    plt.grid(True)
    plt.xlabel('真实水平距离 (米)')
    plt.ylabel('真实垂直高度 (米)')
    plt.legend()
    plt.axis('equal')
    plt.show()


def main():
    # 计算不同瞄镜位置的交点距离
    print("计算箭的轨迹与瞄准线的交点距离")
    print("=" * 40)

    # 获取窥孔的坐标作为参考
    xk = ARROW_CONFIG['xk']
    yk = ARROW_CONFIG['yk']
    theta = ARROW_CONFIG['theta']
    print("箭的俯仰角: %.1f度" % math.degrees(theta))
    print("窥孔位置 (S系): (%.2f, %.3f)" % (xk, yk))

    # 计算窥孔在H系中的坐标
    X_k = xk * math.cos(theta) - yk * math.sin(theta)
    Y_k = xk * math.sin(theta) + yk * math.cos(theta)
    print("窥孔位置 (H系): (%.2f, %.3f)" % (X_k, Y_k))

    aim_x = float(input("请输入瞄镜的X坐标 (米): "))
    aim_y = float(input("请输入瞄镜的Y坐标 (米): "))

    # 计算瞄镜在H系中的坐标
    X_m = aim_x * math.cos(theta) - aim_y * math.sin(theta)
    Y_m = aim_x * math.sin(theta) + aim_y * math.cos(theta)
    print("瞄镜位置 (H系): (%.2f, %.3f)" % (X_m, Y_m))

    # 计算瞄准线俯仰角
    if abs(X_m - X_k) < 1e-10:  # 垂直线
        aim_line_angle = 90.0
    else:
        m = (Y_m - Y_k) / (X_m - X_k)
        aim_line_angle = math.degrees(math.atan(m))
    print("瞄准线俯仰角: %.1f度" % aim_line_angle)
    print("箭的俯仰角: %.1f度" % math.degrees(theta))

    if aim_line_angle >= math.degrees(theta):
        print("警告：瞄准线俯仰角大于或等于箭的俯仰角，可能没有交点！")

    # 计算所有交点距离
    distances = calculate_intersection_distance(aim_x, aim_y, return_all=True)

    if distances:
        print("\n箭的轨迹与瞄准线的交点距离:")
        for i, distance in enumerate(distances):
            print(f"交点{i+1}距离: {distance:.2f}米")
    else:
        print("\n无交点。")

    # 绘制轨迹
    plot_trajectory(aim_x, aim_y)


if __name__ == "__main__":
    main()
