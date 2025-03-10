from arrow_trajectory import calculate_intersection_distance
import math
from config import ARROW_CONFIG


def get_all_intersection_points(aim_x, aim_y):
    """
    计算所有可能的交点及其距离

    返回值格式: [(时间, 距离), ...], 按时间排序
    """
    # 从配置中获取参数
    xk = ARROW_CONFIG['xk']  # 窥孔x坐标 (S系)
    yk = ARROW_CONFIG['yk']  # 窥孔y坐标 (S系)
    v0 = ARROW_CONFIG['v0']  # 出弓速度
    theta = ARROW_CONFIG['theta']  # S系相对于H系的旋转角度
    x0 = ARROW_CONFIG['x0']  # 箭头初始x坐标 (S系)
    y0 = ARROW_CONFIG['y0']  # 箭头初始y坐标 (S系)
    g = ARROW_CONFIG['g']    # 重力加速度

    # 坐标转换: S系 -> H系
    X_m = aim_x * math.cos(theta) - aim_y * math.sin(theta)
    Y_m = aim_x * math.sin(theta) + aim_y * math.cos(theta)
    X_k = xk * math.cos(theta) - yk * math.sin(theta)
    Y_k = xk * math.sin(theta) + yk * math.cos(theta)
    X_0 = x0 * math.cos(theta) - y0 * math.sin(theta)
    Y_0 = x0 * math.sin(theta) + y0 * math.cos(theta)

    # 瞄镜与窥孔的连线方程
    if abs(X_m - X_k) < 1e-10:  # 垂直线
        t = (X_k - X_0) / (v0 * math.cos(theta))
        if t < 0:
            return []

        Y_t = Y_0 + v0 * math.sin(theta) * t - 0.5 * g * t**2
        X_distance = X_k - X_0
        return [(t, X_distance)]  # 只有一个交点
    else:
        # 一般情况
        m = (Y_m - Y_k) / (X_m - X_k)
        A = -0.5 * g
        B = v0 * math.sin(theta) - m * v0 * math.cos(theta)
        C = Y_0 - Y_k - m * (X_0 - X_k)

        discriminant = B**2 - 4*A*C
        if discriminant < 0:
            return []

        t1 = (-B + math.sqrt(discriminant)) / (2*A)
        t2 = (-B - math.sqrt(discriminant)) / (2*A)

        points = []
        for t in [t1, t2]:
            if t > 0:
                X_t = X_0 + v0 * math.cos(theta) * t
                Y_t = Y_0 + v0 * math.sin(theta) * t - 0.5 * g * t**2
                X_distance = X_t - X_0
                points.append((t, X_distance))

        return sorted(points, key=lambda p: p[0])  # 按时间排序


def calculate_aim_angle(aim_x, aim_y):
    """
    计算瞄镜和窥孔连线与S系水平轴的夹角（度数）
    """
    xk = ARROW_CONFIG['xk']
    yk = ARROW_CONFIG['yk']

    # 计算连线在S系中的斜率
    if abs(aim_x - xk) < 1e-10:  # 垂直线
        return 90.0
    else:
        m = (aim_y - yk) / (aim_x - xk)
        return math.degrees(math.atan(m))


def get_aim_point_for_angle(angle_degrees):
    """
    根据给定的瞄准角度计算瞄镜位置
    """
    xk = ARROW_CONFIG['xk']
    yk = ARROW_CONFIG['yk']

    # 瞄镜x坐标固定为前方0.1米
    aim_x = 0.1

    # 根据角度和x距离计算y坐标
    angle_rad = math.radians(angle_degrees)
    aim_y = yk + math.tan(angle_rad) * (aim_x - xk)

    return aim_x, aim_y


def test_formula():
    """
    使用不同的瞄镜位置测试计算公式
    """
    # 获取窥孔的坐标作为参考
    xk = ARROW_CONFIG['xk']
    yk = ARROW_CONFIG['yk']
    theta = ARROW_CONFIG['theta']
    v0 = ARROW_CONFIG['v0']

    # 转换到H系的坐标
    X_k = xk * math.cos(theta) - yk * math.sin(theta)
    Y_k = xk * math.sin(theta) + yk * math.cos(theta)

    # 系统角度说明
    print("【坐标系统和角度说明】")
    print("S系: 箭轴坐标系，x轴为箭所在的轴线")
    print("H系: 真实水平坐标系，X轴为真实水平方向")
    print("- 箭在S系中的角度始终为0度（沿着x轴）")
    print("- S系相对于H系的旋转角度(theta): %.1f度" % math.degrees(theta))
    print("- 箭在H系中的俯仰角: %.1f度" % math.degrees(theta))
    print("-" * 80)

    print("窥孔位置 (S系): (%.2f, %.3f)" % (xk, yk))
    print("窥孔位置 (H系): (%.2f, %.3f)" % (X_k, Y_k))

    # 计算箭的初始速度分量
    vx = v0 * math.cos(theta)
    vy = v0 * math.sin(theta)
    print("箭的速度: %.1f m/s (水平分量: %.1f m/s, 垂直分量: %.1f m/s)" % (v0, vx, vy))

    # 计算箭可能达到的最大高度和最远距离
    max_height = vy**2 / (2 * ARROW_CONFIG['g'])
    max_distance = v0**2 * math.sin(2*theta) / ARROW_CONFIG['g']
    print("箭的最大高度: %.2f m, 最大水平距离: %.2f m" % (max_height, max_distance))

    print("\n【交点条件说明】")
    print("- 箭在S系中沿x轴运动（0度）")
    print("- 要有交点，瞄准线在S系中的角度应小于0度")
    # print("- 只有在存在两个交点时，才取第一个交点作为有效结果")

    # 聚焦在-0.75度到-0.70度之间的更细致的测试
    print("\n【-3度到-0.50度之间的精细测试】")
    fine_angles = [x/100 for x in range(-300, -50, 10)]
    run_test_for_angles(fine_angles)

    # 更加精细的测试，找到临界值
    # critical_angles = []
    # for i in range(750, 700, -1):
    #     critical_angles.append(-i/1000)

    # print("\n【临界角度精确搜索 (精确到0.001度)】")
    # run_test_for_angles(critical_angles)


def run_test_for_angles(test_angles):
    """针对给定的角度集合运行测试"""
    # 根据角度生成瞄镜位置
    test_points = [get_aim_point_for_angle(angle) for angle in test_angles]

    print("-" * 80)
    print("序号 | 瞄镜坐标(S系) | S系瞄准角度 | 交点情况")
    print("-" * 80)

    for i, ((aim_x, aim_y), target_angle) in enumerate(zip(test_points, test_angles)):
        # 计算实际角度（由于取整可能与目标角度略有差异）
        actual_angle = calculate_aim_angle(aim_x, aim_y)

        # 判断是否满足交点条件
        angle_condition = "满足" if actual_angle < 0 else "不满足"

        # 计算所有交点
        intersections = get_all_intersection_points(aim_x, aim_y)

        # 交点数量
        if len(intersections) == 0:
            intersection_status = "无交点"
        elif len(intersections) == 1:
            intersection_status = "1个交点"
        else:
            intersection_status = "2个交点"

        # 输出信息
        print("#%d | (%.2f, %.2f) | %.3f度 | %s, %s" %
              (i+1, aim_x, aim_y, actual_angle, angle_condition, intersection_status))

        # 详细输出交点信息
        if intersections:
            for j, (t, distance) in enumerate(intersections):
                print("    交点%d: 时间 = %.2f秒, H系距离 = %.2f米" %
                      (j+1, t, distance))

            # 正式计算交点距离，只有两个交点时才返回第一个交点距离
            X = calculate_intersection_distance(aim_x, aim_y)
            if X is not None:
                print("    选择的有效交点距离: %.2f米" % X)
            else:
                print("    无有效交点 (只有在有两个交点时才有效)")


if __name__ == "__main__":
    test_formula()
