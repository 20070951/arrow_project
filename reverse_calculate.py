import math
import numpy as np
from scipy.optimize import minimize
from config import ARROW_CONFIG


def calculate_X_with_params(params, aim_x, aim_y, target_X, theta):
    """
    使用给定参数计算交点距离X

    参数:
        params: [v0, x0, y0] - 需要优化的参数
        aim_x, aim_y: 瞄镜坐标（在箭轴坐标系中）
        target_X: 目标中靶距离（真实水平距离）
        theta: 俯仰角（弧度）

    返回:
        与目标距离的误差平方
    """
    v0, x0, y0 = params

    # 从配置中获取固定参数
    xk = ARROW_CONFIG['xk']  # 窥孔x坐标 (S系)
    yk = ARROW_CONFIG['yk']  # 窥孔y坐标 (S系)
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
        # 求解 X(t) = X_k 时的t
        # X_0 + v0*cos(theta)*t = X_k
        t = (X_k - X_0) / (v0 * math.cos(theta))
        if t < 0:  # 时间不能为负
            return 1e6  # 返回大误差

        # 代入计算轨迹上的点
        Y_t = Y_0 + v0 * math.sin(theta) * t - 0.5 * g * t**2

        # 计算交点到原点的真实水平距离
        X_distance = X_k - X_0
        return (abs(X_distance) - target_X)**2
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
            return 1e6  # 返回大误差

        # 求解二次方程的两个根
        t1 = (-B + math.sqrt(discriminant)) / (2*A)
        t2 = (-B - math.sqrt(discriminant)) / (2*A)

        # 选择有效的时间（正值）
        valid_times = [t for t in [t1, t2] if t > 0]

        if not valid_times:
            # 没有正的时间解
            return 1e6  # 返回大误差

        # 取最小的正时间值（第一个交点）
        t = min(valid_times)

        # 计算交点的真实水平坐标
        X_t = X_0 + v0 * math.cos(theta) * t

        # 计算交点到原点的真实水平距离
        X_distance = X_t - X_0

        return (abs(X_distance) - target_X)**2


def reverse_calculate(aim_x, aim_y, target_X, theta, initial_guess=None):
    """
    根据中靶位置和瞄镜位置反向计算v0、x0和y0

    参数:
        aim_x, aim_y: 瞄镜坐标（在箭轴坐标系中）
        target_X: 实际中靶距离（真实水平距离）
        theta: 俯仰角（弧度）
        initial_guess: 初始猜测值 [v0, x0, y0]

    返回:
        优化后的参数 [v0, x0, y0]
    """
    if initial_guess is None:
        # 使用合理的初始猜测值
        initial_guess = [70.0, 0.0, 0.0]  # [v0, x0, y0]

    # 设置参数边界
    bounds = [
        (30.0, 120.0),  # v0 范围 (30-120 m/s)
        (-0.1, 0.1),    # x0 范围 (-0.1-0.1 m)
        (-0.1, 0.1)     # y0 范围 (-0.1-0.1 m)
    ]

    # 使用优化算法找到最佳参数
    result = minimize(
        calculate_X_with_params,
        initial_guess,
        args=(aim_x, aim_y, target_X, theta),
        bounds=bounds,
        method='L-BFGS-B'
    )

    if result.success:
        return result.x
    else:
        print("优化失败:", result.message)
        return None


def reverse_calculate_multiple_shots(shots_data, theta):
    """
    使用多个射击数据点进行反向计算，提高准确性

    参数:
        shots_data: 列表，每项包含 (aim_x, aim_y, target_X)
        theta: 俯仰角（弧度）

    返回:
        优化后的参数 [v0, x0, y0]
    """
    def objective_function(params):
        v0, x0, y0 = params
        total_error = 0

        for aim_x, aim_y, target_X in shots_data:
            error = calculate_X_with_params(
                [v0, x0, y0], aim_x, aim_y, target_X, theta)
            total_error += error

        return total_error / len(shots_data)

    # 使用合理的初始猜测值
    initial_guess = [70.0, 0.0, 0.0]  # [v0, x0, y0]

    # 设置参数边界
    bounds = [
        (30.0, 120.0),  # v0 范围 (30-120 m/s)
        (-0.1, 0.1),    # x0 范围 (-0.1-0.1 m)
        (-0.1, 0.1)     # y0 范围 (-0.1-0.1 m)
    ]

    # 使用优化算法找到最佳参数
    result = minimize(
        objective_function,
        initial_guess,
        bounds=bounds,
        method='L-BFGS-B'
    )

    if result.success:
        return result.x
    else:
        print("优化失败:", result.message)
        return None


def update_config_with_params(v0, x0, y0, theta=None):
    """
    使用计算出的参数更新配置文件

    参数:
        v0, x0, y0: 计算出的参数
        theta: 俯仰角（弧度），如果为None则不更新
    """
    from config import ARROW_CONFIG

    # 创建新的配置字典
    new_config = ARROW_CONFIG.copy()

    # 更新参数
    new_config['v0'] = v0
    new_config['x0'] = x0
    new_config['y0'] = y0
    if theta is not None:
        new_config['theta'] = theta

    # 打印新的配置
    print("\n计算得出的参数:")
    print("出弓速度 v0 = %.2f m/s" % v0)
    print("箭头初始x坐标 x0 = %.2f m" % x0)
    print("箭头初始y坐标 y0 = %.2f m" % y0)
    if theta is not None:
        print("俯仰角 theta = %.2f 度" % math.degrees(theta))

    # 返回新的配置
    return new_config


def main():
    print("射箭参数反向计算工具")
    print("=" * 40)
    print("请选择计算模式:")
    print("1. 单次射击数据计算")
    print("2. 多次射击数据计算（更准确）")

    mode = input("请输入选择 (1/2): ")

    theta = float(input("请输入俯仰角 (度): "))
    theta_rad = math.radians(theta)

    if mode == "1":
        # 单次射击数据计算
        aim_x = float(input("请输入瞄镜X坐标 (米): "))
        aim_y = float(input("请输入瞄镜Y坐标 (米): "))
        target_X = float(input("请输入实际中靶距离X (米): "))

        result = reverse_calculate(aim_x, aim_y, target_X, theta_rad)

        if result is not None:
            v0, x0, y0 = result
            update_config_with_params(v0, x0, y0, theta_rad)

    elif mode == "2":
        # 多次射击数据计算
        shots_data = []
        num_shots = int(input("请输入射击数据点数量: "))

        for i in range(num_shots):
            print("\n射击数据点 #%d:" % (i+1))
            aim_x = float(input("瞄镜X坐标 (米): "))
            aim_y = float(input("瞄镜Y坐标 (米): "))
            target_X = float(input("实际中靶距离X (米): "))
            shots_data.append((aim_x, aim_y, target_X))

        result = reverse_calculate_multiple_shots(shots_data, theta_rad)

        if result is not None:
            v0, x0, y0 = result
            update_config_with_params(v0, x0, y0, theta_rad)

    else:
        print("无效的选择!")


if __name__ == "__main__":
    main()
