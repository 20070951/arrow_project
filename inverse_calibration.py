"""
逆向标定模块

本模块通过两次标定实验，确定瞄准高度(ym)和距离(d)之间的线性关系。

流程：(ym1,d1) + (ym2,d2) → 确定线性关系 ym = a*d + b → 根据需要计算任意距离的ym
"""

import numpy as np
import matplotlib.pyplot as plt
import config
from simple_calibration import create_calibration_table, print_calibration_table


def linear_calibration(calibration_points):
    """
    使用两个标定点确定瞄准高度(ym)和距离(d)之间的线性关系。

    参数：
        calibration_points: 标定实验的元组列表 [(ym1, d1), (ym2, d2)]

    返回：
        dict: 包含线性关系系数和其他信息
    """
    # 此方法需要恰好两个点
    if len(calibration_points) != 2:
        raise ValueError("线性校准需要恰好两个标定点")

    # 提取标定数据
    (ym1, d1), (ym2, d2) = calibration_points

    # 计算线性关系参数：ym = a*d + b
    if d1 == d2:
        raise ValueError("两个标定点的距离不能相同")

    a = (ym2 - ym1) / (d2 - d1)  # 斜率
    b = ym1 - a * d1             # 截距

    print(f"线性关系: ym = {a:.6f} * d + {b:.6f}")

    # 返回结果
    return {
        'a': a,  # 斜率
        'b': b,  # 截距
        'formula': f"ym = {a:.6f} * d + {b:.6f}"
    }


def predict_ym(linear_params, distance):
    """
    根据线性关系预测给定距离下的瞄准高度。

    参数：
        linear_params: 线性关系参数字典
        distance: 目标距离

    返回：
        float: 预测的瞄准高度
    """
    a = linear_params['a']  # 斜率
    b = linear_params['b']  # 截距

    # 计算预测值：ym = a*d + b
    return a * distance + b


def verify_calibration(linear_params, calibration_points):
    """
    验证线性关系预测值与实际值的差异。

    参数：
        linear_params: 线性关系参数字典
        calibration_points: 标定实验的元组列表 (ym, d)

    返回：
        bool: 验证成功返回True
    """
    print("\n=== 校准验证 ===")
    print(f"线性关系: {linear_params['formula']}")

    print("\n标定点比较：")
    print(f"{'距离 (m)':^12} | {'实际ym (m)':^15} | {'预测ym (m)':^15} | {'误差 (mm)':^12} | {'误差 (%)':^10}")
    print("-" * 70)

    for i, (ym_actual, d_value) in enumerate(calibration_points):
        ym_predicted = predict_ym(linear_params, d_value)

        error = ym_predicted - ym_actual
        error_mm = error * 1000  # 转换为毫米
        error_percent = (error / ym_actual) * \
            100 if ym_actual != 0 else float('inf')

        print(f"{d_value:^12.1f} | {ym_actual:^15.6f} | {ym_predicted:^15.6f} | {error_mm:^12.2f} | {error_percent:^10.2f}")

    return True


def plot_calibration(linear_params, calibration_points):
    """
    绘制线性关系校准曲线。

    参数：
        linear_params: 线性关系参数字典
        calibration_points: 标定实验的元组列表 (ym, d)
    """
    # 使用线性关系生成校准曲线
    d_min = min(d for _, d in calibration_points)
    d_max = max(d for _, d in calibration_points)

    # 确保范围更广一些，便于可视化
    range_min = max(0, d_min - 10)
    range_max = d_max + 20

    # 生成数据点
    distances = np.linspace(range_min, range_max, 100)
    ym_values = [predict_ym(linear_params, d) for d in distances]

    # 提取标定点
    cal_distances = [d for _, d in calibration_points]
    cal_ym_actual = [ym for ym, _ in calibration_points]

    # 绘图
    plt.figure(figsize=(10, 6))

    # 绘制校准曲线
    plt.plot(distances, ym_values, 'b-', linewidth=2, label='Calibration Line')

    # 绘制标定点
    plt.scatter(cal_distances, cal_ym_actual, color='red',
                s=80, zorder=5, label='Calibration Points')

    # 添加标签和标题
    plt.title(f'Linear Calibration: {linear_params["formula"]}')
    plt.xlabel('Distance (m)')
    plt.ylabel('Scope Height (m)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # 为标定点添加注释
    for d, ym in zip(cal_distances, cal_ym_actual):
        plt.annotate(f"({d}, {ym:.4f})",
                     xy=(d, ym),
                     xytext=(5, 5),
                     textcoords='offset points')

    plt.tight_layout()
    plt.savefig('linear_calibration.png', dpi=300)
    plt.show()


def create_calibration_table_linear(linear_params, distance_range, step=10):
    """
    基于线性关系创建校准表。

    参数：
        linear_params: 线性关系参数字典
        distance_range: 距离范围元组 (min_distance, max_distance)
        step: 步长

    返回：
        list: 校准数据列表
    """
    min_dist, max_dist = distance_range
    distances = np.arange(min_dist, max_dist + step, step)

    calibration_data = []
    for d in distances:
        ym = predict_ym(linear_params, d)
        calibration_data.append({
            'distance': d,
            'ym': ym
        })

    return calibration_data


def print_calibration_table_linear(calibration_data):
    """
    打印校准表。

    参数：
        calibration_data: 校准数据列表
    """
    print("\n=== 校准表 ===")
    print(f"{'距离 (m)':^12} | {'瞄准高度 (m)':^15}")
    print("-" * 30)

    for data in calibration_data:
        print(f"{data['distance']:^12.0f} | {data['ym']:^15.6f}")


def main():
    """运行逆向标定过程的主函数。"""
    print("=== 线性校准过程 ===")
    print("将通过两次标定实验确定瞄准高度(ym)和距离(d)之间的线性关系。\n")

    # 选择使用样本数据或输入新数据
    use_sample = input("使用样本标定数据？(y/n): ").lower().strip() == 'y'

    if use_sample:
        # 样本标定数据
        calibration_points = [
            (0.087, 20),   # 距离20m时的瞄准高度0.087m
            (0.046, 100)   # 距离100m时的瞄准高度0.046m
        ]
    else:
        # 输入新标定数据
        calibration_points = []
        print("\n输入您的标定数据：")

        for i in range(2):
            while True:
                try:
                    d = float(input(f"输入距离 {i+1} (m): "))
                    ym = float(input(f"输入瞄准高度 {i+1} (m): "))
                    calibration_points.append((ym, d))
                    break
                except ValueError:
                    print("输入无效。请输入数字值。")

    print(f"\n标定点：")
    for i, (ym, d) in enumerate(calibration_points):
        print(f"点 {i+1}: 距离 = {d}m, 瞄准高度 = {ym}m")

    # 使用线性校准方法确定关系
    try:
        linear_params = linear_calibration(calibration_points)

        # 验证校准
        verify_calibration(linear_params, calibration_points)

        # 绘制校准曲线
        plot_calibration(linear_params, calibration_points)

        # 创建并打印校准表
        print("\n是否要创建校准表？(y/n): ", end="")
        if input().lower().strip() == 'y':
            # 获取距离范围
            min_dist = int(input("输入最小距离 (m): "))
            max_dist = int(input("输入最大距离 (m): "))
            step = int(input("输入步长 (m): "))

            # 生成校准表
            cal_table = create_calibration_table_linear(
                linear_params, (min_dist, max_dist), step)
            print_calibration_table_linear(cal_table)

        # 交互式距离查询
        print("\n是否要查询特定距离的瞄准高度？(y/n): ", end="")
        if input().lower().strip() == 'y':
            while True:
                try:
                    distance = float(input("\n输入目标距离 (m)，输入负数退出: "))
                    if distance < 0:
                        break

                    ym_pred = predict_ym(linear_params, distance)
                    print(
                        f"距离 {distance}m 的预测瞄准高度: {ym_pred:.6f}m ({ym_pred*1000:.1f}mm)")
                except ValueError:
                    print("输入无效。请输入数字值。")

    except Exception as e:
        print(f"线性校准过程中出错：{str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
