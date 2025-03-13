"""
简化版弓箭校准模块

输入目标距离d，直接计算所需的瞄镜高度ym。
流程：d → 计算theta → 计算ym
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import config


def calculate_theta(distance):
    """
    计算给定距离下的俯仰角

    参数:
        distance: 目标距离，单位米

    返回:
        float: 俯仰角，单位弧度
    """
    # 使用简化的物理模型
    g = config.g  # 重力加速度
    v0 = config.v0  # 初速度
    h = config.h  # 靶心与箭头的高度差

    # 基本俯仰角（不考虑高度差）
    # 使用弹道方程求解: d = (v0^2 * sin(2*theta)) / g
    # 变形得到: sin(2*theta) = (g * d) / (v0^2)
    # 对于小角度: theta ≈ arcsin((g * d) / (2 * v0^2)) / 2
    # 进一步简化: theta ≈ arctan(g * d / (2 * v0^2))
    theta_base = math.atan(g * distance / (2 * v0**2))

    # 考虑高度差的修正
    # 高度差h导致的角度修正近似为arctan(h/d)
    theta_correction = math.atan(h / distance) if distance > 0 else 0

    # 总俯仰角
    theta = theta_base + theta_correction

    return theta


def calculate_ym(distance, theta):
    """
    计算给定距离和俯仰角下的瞄镜高度

    参数:
        distance: 目标距离，单位米
        theta: 俯仰角，单位弧度

    返回:
        float: 瞄镜高度，单位米
    """
    # 获取配置参数
    yk = config.yk  # 窥孔的纵坐标
    h = config.h  # 靶心与箭头的高度差
    arm_length = config.arm_length  # 臂展
    bow_a = config.bow_a  # 弓参数a
    bow_b = config.bow_b  # 弓参数b
    bow_length = config.bow_length  # 弓长

    # 计算窥孔的横坐标
    # 弓弦端点坐标
    x1, y1 = -arm_length, 0
    x2, y2 = -bow_a, bow_length/2 + bow_b

    # 根据yk计算xk
    xk = x1 + (yk - y1) * (x2 - x1) / (y2 - y1)

    # 计算S系中的靶心坐标
    xs_target = distance * math.cos(theta) + h * math.sin(theta)
    ys_target = -distance * math.sin(theta) + h * math.cos(theta)

    # 计算从窥孔到靶心的直线斜率
    # 如果xs_target和xk相等，斜率为无穷大，直接返回yk
    if abs(xs_target - xk) < 1e-10:
        return yk

    slope = (ys_target - yk) / (xs_target - xk)

    # 瞄准线通过靶心和窥孔，求其与y轴的交点（即x=0处的y值）
    # 使用点斜式方程: y - yk = slope * (x - xk)
    # 当x=0时: ym = yk - slope * xk
    ym = yk - slope * xk

    return ym


def calibrate(distance):
    """
    根据目标距离计算瞄镜高度

    参数:
        distance: 目标距离，单位米

    返回:
        dict: 包含俯仰角和瞄镜高度
    """
    # 计算俯仰角
    theta = calculate_theta(distance)

    # 计算瞄镜高度
    ym = calculate_ym(distance, theta)

    # 转换为度数，便于理解
    theta_degrees = math.degrees(theta)

    return {
        'distance': distance,
        'theta_radians': theta,
        'theta_degrees': theta_degrees,
        'ym': ym
    }


def create_calibration_table(distance_range=(10, 100), step=5, h=None, yk=None):
    """
    创建一个距离-瞄镜高度校准表格

    参数:
        distance_range: 距离范围，(最小距离, 最大距离)
        step: 距离步长
        h: 靶心与箭头的高度差，默认使用config.h
        yk: 窥孔的纵坐标，默认使用config.yk

    返回:
        list: 校准数据列表
    """
    # 保存原始参数值
    original_h = None
    original_yk = None

    # 临时修改参数，如果提供了自定义值
    if h is not None:
        original_h = config.h
        config.h = h

    if yk is not None:
        original_yk = config.yk
        config.yk = yk

    try:
        min_dist, max_dist = distance_range
        distances = np.arange(min_dist, max_dist + step, step)

        # 校准数据列表
        calibration_data = []

        for d in distances:
            result = calibrate(d)
            calibration_data.append(result)

        return calibration_data
    finally:
        # 恢复原始参数值
        if original_h is not None:
            config.h = original_h

        if original_yk is not None:
            config.yk = original_yk


def plot_calibration_chart(calibration_data):
    """
    绘制校准图表

    参数:
        calibration_data: 校准数据列表
    """
    distances = [data['distance'] for data in calibration_data]
    ym_values = [data['ym'] for data in calibration_data]

    plt.figure(figsize=(10, 6))

    # 绘制瞄镜高度曲线
    plt.subplot(2, 1, 1)
    plt.plot(distances, ym_values, 'b-', linewidth=2)
    plt.scatter(distances, ym_values, color='red', s=30)
    plt.title('Scope Height Calibration')
    plt.xlabel('Distance (m)')
    plt.ylabel('Scope Height (m)')
    plt.grid(True)

    # 为部分点添加标签
    for i, (d, ym) in enumerate(zip(distances, ym_values)):
        if i % 2 == 0:  # 每隔一个点添加标签
            plt.annotate(f"{ym:.4f}", (d, ym), textcoords="offset points",
                         xytext=(0, 10), ha='center')

    # 绘制俯仰角曲线
    theta_values = [math.degrees(data['theta_radians'])
                    for data in calibration_data]

    plt.subplot(2, 1, 2)
    plt.plot(distances, theta_values, 'g-', linewidth=2)
    plt.scatter(distances, theta_values, color='orange', s=30)
    plt.title('Elevation Angle Calibration')
    plt.xlabel('Distance (m)')
    plt.ylabel('Elevation Angle (degree)')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('calibration_charts.png')
    plt.show()


def print_calibration_table(calibration_data):
    """打印校准表格"""
    print("\n距离-瞄镜高度校准表:")
    print("-" * 60)
    print(f"{'距离(m)':^10} | {'俯仰角(度)':^12} | {'瞄镜高度(m)':^12}")
    print("-" * 60)

    for data in calibration_data:
        d = data['distance']
        theta_deg = data['theta_degrees']
        ym = data['ym']
        print(f"{d:^10.1f} | {theta_deg:^12.4f} | {ym:^12.4f}")

    print("-" * 60)


if __name__ == "__main__":
    # 示例：计算特定距离的瞄镜高度
    target_distance = 30  # 目标距离30米
    result = calibrate(target_distance)

    print(f"目标距离: {result['distance']}米")
    print(
        f"俯仰角: {result['theta_degrees']:.4f}度 ({result['theta_radians']:.6f}弧度)")
    print(f"瞄镜高度: {result['ym']:.4f}米")

    # 创建校准表格
    print("\n生成校准表格...")
    calibration_data = create_calibration_table((20, 100), 1)

    # 打印校准表格
    print_calibration_table(calibration_data)

    # 绘制校准图表
    plot_calibration_chart(calibration_data)
