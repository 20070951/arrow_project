from reverse_calculate import reverse_calculate, update_config_with_params
import math
from config import ARROW_CONFIG
from arrow_trajectory import calculate_intersection_distance


def test_reverse_calculation():
    """
    测试反向计算功能

    使用已知参数生成模拟数据，然后尝试反向计算这些参数
    """
    # 保存原始配置
    original_v0 = ARROW_CONFIG['v0']
    original_x0 = ARROW_CONFIG['x0']
    original_y0 = ARROW_CONFIG['y0']
    original_theta = ARROW_CONFIG['theta']

    # 设置测试参数
    test_v0 = 65.0  # 出弓速度
    test_x0 = 0.0   # 箭头初始x坐标
    test_y0 = 0.0   # 箭头初始y坐标
    test_theta = math.radians(5.0)  # 俯仰角，与config.py保持一致

    # 临时修改配置
    ARROW_CONFIG['v0'] = test_v0
    ARROW_CONFIG['x0'] = test_x0
    ARROW_CONFIG['y0'] = test_y0
    ARROW_CONFIG['theta'] = test_theta

    # 获取窥孔的坐标作为参考
    xk = ARROW_CONFIG['xk']
    yk = ARROW_CONFIG['yk']

    # 箭的俯仰角（度数）
    arrow_angle = math.degrees(test_theta)
    print("箭的俯仰角: %.1f度" % arrow_angle)

    # 生成模拟数据 - 确保瞄准线俯仰角小于箭的俯仰角
    aim_points = [
        (0.05, -0.05),   # 低于窥孔 - 下射
        (0.05, 0.00),    # 与原点同高 - 接近水平
        (0.05, 0.02),    # 略高于原点 - 轻微上射
    ]

    print("生成模拟数据:")
    print("真实参数: v0=%.2f, x0=%.2f, y0=%.2f, theta=%.2f度" %
          (test_v0, test_x0, test_y0, arrow_angle))
    print("窥孔位置 (S系): (%.2f, %.3f)" % (xk, yk))

    # 转换到H系的坐标
    X_k = xk * math.cos(test_theta) - yk * math.sin(test_theta)
    Y_k = xk * math.sin(test_theta) + yk * math.cos(test_theta)
    print("窥孔位置 (H系): (%.2f, %.3f)" % (X_k, Y_k))

    print("\n瞄镜坐标、俯仰角和交点距离")
    print("-" * 60)
    print("箭俯仰角: %.1f度" % arrow_angle)
    print("-" * 60)

    shots_data = []
    for aim_x, aim_y in aim_points:
        # 计算瞄镜在H系中的坐标
        X_m = aim_x * math.cos(test_theta) - aim_y * math.sin(test_theta)
        Y_m = aim_x * math.sin(test_theta) + aim_y * math.cos(test_theta)

        # 计算窥孔到瞄镜连线在H系中的俯仰角
        if abs(X_m - X_k) < 1e-10:  # 垂直线
            aim_line_angle = 90.0
        else:
            # 计算连线斜率
            m = (Y_m - Y_k) / (X_m - X_k)
            aim_line_angle = math.degrees(math.atan(m))

        # 计算交点距离
        X = calculate_intersection_distance(aim_x, aim_y)

        angle_comparison = "< 箭角 (会相交)" if aim_line_angle < arrow_angle else "> 箭角 (不会相交)"
        result_str = "S系(%.2f, %.3f) -> 俯仰角: %.1f度 %s -> %s" % (
            aim_x, aim_y, aim_line_angle, angle_comparison,
            "距离: %.2f m" % X if X is not None else "无交点")

        print(result_str)

        if X is not None:
            shots_data.append((aim_x, aim_y, X))

    # 如果没有找到交点，无法继续测试
    if not shots_data:
        print("\n所有测试点都没有交点，无法进行反向计算！")
        return

    # 恢复原始配置
    ARROW_CONFIG['v0'] = original_v0
    ARROW_CONFIG['x0'] = original_x0
    ARROW_CONFIG['y0'] = original_y0
    ARROW_CONFIG['theta'] = original_theta

    print("\n开始反向计算...")

    # 使用单个数据点进行反向计算
    print("\n使用单个数据点:")
    aim_x, aim_y, target_X = shots_data[0]
    result = reverse_calculate(aim_x, aim_y, target_X, test_theta)

    if result is not None:
        v0, x0, y0 = result
        print("\n单点计算结果:")
        print("v0=%.2f (误差: %.2f%%)" % (v0, 100 * abs(v0 - test_v0) / test_v0))
        print("x0=%.2f (误差: %.2f米)" % (x0, abs(x0 - test_x0)))
        print("y0=%.2f (误差: %.2f米)" % (y0, abs(y0 - test_y0)))

    # 使用多个数据点进行反向计算
    from reverse_calculate import reverse_calculate_multiple_shots

    if len(shots_data) >= 2:
        print("\n使用多个数据点:")
        result = reverse_calculate_multiple_shots(shots_data, test_theta)

        if result is not None:
            v0, x0, y0 = result
            print("\n多点计算结果:")
            print("v0=%.2f (误差: %.2f%%)" %
                  (v0, 100 * abs(v0 - test_v0) / test_v0))
            print("x0=%.2f (误差: %.2f米)" % (x0, abs(x0 - test_x0)))
            print("y0=%.2f (误差: %.2f米)" % (y0, abs(y0 - test_y0)))
    else:
        print("\n没有足够的数据点进行多点计算")


if __name__ == "__main__":
    test_reverse_calculation()
