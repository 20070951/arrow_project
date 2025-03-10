import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import math
from config import ARROW_CONFIG
import io


def calculate_intersection_points(aim_x, aim_y, config):
    """
    计算箭的轨迹与窥孔和瞄镜连线的所有交点

    返回：交点列表[(时间, 距离), ...]，按时间排序
    """
    # 从配置中获取参数
    xk = config['xk']  # 窥孔x坐标 (S系)
    yk = config['yk']  # 窥孔y坐标 (S系)
    v0 = config['v0']  # 出弓速度
    theta = config['theta']  # S系相对于H系的旋转角度
    x0 = config['x0']  # 箭头初始x坐标 (S系)
    y0 = config['y0']  # 箭头初始y坐标 (S系)
    g = config['g']    # 重力加速度

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


def calculate_aim_angle(aim_x, aim_y, config):
    """计算瞄准线在S系中的角度（度数）"""
    xk = config['xk']
    yk = config['yk']

    dx = aim_x - xk
    dy = aim_y - yk

    if abs(dx) < 1e-10:  # 避免除以零
        return 90.0 if dy > 0 else -90.0

    angle_rad = math.atan2(dy, dx)
    return math.degrees(angle_rad)


def plot_trajectory(aim_x, aim_y, config):
    """
    绘制箭的轨迹和瞄准线，返回图像对象
    """
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei',
                                       'Microsoft YaHei', 'Arial Unicode MS']  # 优先使用的中文字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 从配置中获取参数
    xk = config['xk']
    yk = config['yk']
    v0 = config['v0']
    theta = config['theta']
    x0 = config['x0']
    y0 = config['y0']
    g = config['g']

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

    # 计算瞄准线在H系中的角度
    if abs(X_m - X_k) < 1e-10:  # 垂直线
        aim_angle_h = 90.0 if Y_m > Y_k else -90.0
    else:
        m = (Y_m - Y_k) / (X_m - X_k)
        aim_angle_h = math.degrees(math.atan(m))

    # 估计飞行时间
    # 粗略估计：如果箭水平飞行时能到达的最远距离
    t_flight = 2 * v0 * math.sin(theta) / g if v0 * math.sin(theta) > 0 else 20

    # 创建时间点
    t = np.linspace(0, t_flight * 1.2, 1000)

    # 计算箭在H系中的轨迹
    X_traj = X_0 + v0 * math.cos(theta) * t
    Y_traj = Y_0 + v0 * math.sin(theta) * t - 0.5 * g * t**2

    # 计算瞄准线在H系中的点（从瞄镜到窥孔）
    if abs(X_m - X_k) < 1e-10:  # 垂直线
        X_aim = np.ones(100) * X_k
        Y_aim = np.linspace(min(Y_k, Y_m) - 1, max(Y_k, Y_m) + 1, 100)
    else:
        X_aim = np.linspace(min(X_k, X_m) - 1, max(X_k, X_m) + 5, 100)
        m = (Y_m - Y_k) / (X_m - X_k)
        Y_aim = Y_k + m * (X_aim - X_k)

    # 计算瞄准线的延伸射线（从窥孔延伸出去）
    if abs(X_m - X_k) < 1e-10:  # 垂直线
        X_ray = np.ones(100) * X_k
        Y_ray = np.linspace(min(Y_k, Y_m) - 20, max(Y_k, Y_m) + 20, 100)
    else:
        m = (Y_m - Y_k) / (X_m - X_k)
        # 计算射线延伸的最大距离
        max_distance = max(150, max(X_traj) * 1.2)

        # 计算从窥孔延伸出去的点
        X_ray = np.linspace(X_k, X_k + max_distance, 100)
        Y_ray = Y_k + m * (X_ray - X_k)

    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制轨迹
    ax.plot(X_traj, Y_traj, 'r-', label='箭的轨迹')
    ax.plot(X_aim, Y_aim, 'b--', label='瞄准线')
    ax.plot(X_ray, Y_ray, 'g:', linewidth=1.5, label='瞄准射线延伸')
    ax.plot(X_k, Y_k, 'go', label='窥孔位置')
    ax.plot(X_m, Y_m, 'mo', label='瞄镜位置')
    ax.plot(X_0, Y_0, 'ko', label='箭头初始位置')

    # 计算交点并绘制
    intersections = calculate_intersection_points(aim_x, aim_y, config)

    # 绘制所有交点
    for i, (t_val, _) in enumerate(intersections):
        X_i = X_0 + v0 * math.cos(theta) * t_val
        Y_i = Y_0 + v0 * math.sin(theta) * t_val - 0.5 * g * t_val**2
        ax.plot(X_i, Y_i, 'rx', markersize=10,
                label=f'交点{i+1} (t={t_val:.2f}s)')

    # 设置图表属性
    aim_angle = calculate_aim_angle(aim_x, aim_y, config)  # S系中的瞄准角度
    if intersections:
        ax.set_title(
            f'箭的轨迹与瞄准线 (俯仰角: {math.degrees(theta):.1f}度, H系瞄准角度: {aim_angle_h:.2f}度)')
    else:
        ax.set_title(
            f'箭的轨迹与瞄准线 (俯仰角: {math.degrees(theta):.1f}度, H系瞄准角度: {aim_angle_h:.2f}度, 无交点)')

    # 计算轨迹的最高点
    max_height = max(Y_traj)

    # 调整y轴比例，使得轨迹更明显
    y_max = max(max_height + 5, 10)  # 至少10米高
    y_min = min(min(Y_traj) - 5, -2)  # 至少有-2米的下限

    # 为了保持更好的视觉效果，设置y轴最小范围为12米
    if y_max - y_min < 12:
        y_center = (y_max + y_min) / 2
        y_max = y_center + 6
        y_min = y_center - 6

    ax.set_ylim(y_min, y_max)

    # 如果箭射得很远，限制x轴范围
    max_x = max(X_traj) * 1.1
    if max_x > 150:
        ax.set_xlim(-5, 150)
    else:
        ax.set_xlim(-5, max_x)

    ax.grid(True)
    ax.set_xlabel('真实水平距离 (米)')
    ax.set_ylabel('真实垂直高度 (米)')
    ax.legend()

    # 返回图像对象
    return fig


def main():
    # 设置页面配置
    st.set_page_config(page_title="箭的轨迹计算器", layout="wide")

    # 设置全局中文字体
    st.markdown("""
    <style>
        body {
            font-family: "Microsoft YaHei", SimHei, sans-serif !important;
        }
        .st-ae, .st-af, .st-ag, .st-ah, .st-ai, .st-aj, .st-ak, .st-al, .st-am, .st-an {
            font-family: "Microsoft YaHei", SimHei, sans-serif !important;
        }
    </style>
    """, unsafe_allow_html=True)

    st.title("箭的轨迹计算与交点分析")

    # 添加坐标系说明
    st.markdown("""
    ### 坐标系说明
    - **S系 (箭轴坐标系)**: 以箭为参考系的坐标系，箭的飞行方向始终沿着S系的x轴正方向（0度角）。
    - **H系 (真实世界坐标系)**: 以地面为参考的真实世界坐标系，重力方向沿着y轴负方向。
    - **θ (theta)**: S系相对于H系的旋转角度，同时也是箭在H系中的俯仰角。
    """)

    # 创建两列布局
    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("配置区")

        # 复制默认配置并提供修改界面
        config = ARROW_CONFIG.copy()

        with st.expander("箭的参数设置", expanded=True):
            config['v0'] = st.slider(
                "箭的出弓速度 (m/s)", 10.0, 100.0, float(config['v0']), 0.5)
            config['theta'] = math.radians(
                st.slider("箭的俯仰角 (度)", 0.0, 20.0, float(math.degrees(config['theta'])), 0.5))
            config['g'] = st.slider(
                "重力加速度 (m/s²)", 9.0, 10.0, float(config['g']), 0.1)

        with st.expander("坐标设置", expanded=True):
            config['xk'] = st.number_input(
                "窥孔X坐标 (S系)", -1.0, 1.0, float(config['xk']), 0.01)
            config['yk'] = st.number_input(
                "窥孔Y坐标 (S系)", -1.0, 1.0, float(config['yk']), 0.01)
            config['x0'] = st.number_input(
                "箭头初始X坐标 (S系)", -1.0, 1.0, float(config['x0']), 0.01)
            config['y0'] = st.number_input(
                "箭头初始Y坐标 (S系)", -1.0, 1.0, float(config['y0']), 0.01)

        st.subheader("瞄镜设置")

        # 直接设置瞄镜的位置
        aim_x = st.number_input("瞄镜X坐标 (S系)", -1.0, 1.0, -0.3, 0.01)
        aim_y = st.number_input("瞄镜Y坐标 (S系)", -1.0, 1.0, 0.1, 0.01)

        # 计算并显示瞄准线角度
        aim_angle = calculate_aim_angle(aim_x, aim_y, config)
        st.write(f"当前瞄准线角度 (S系): {aim_angle:.2f}度")

        # 允许通过角度调整瞄镜位置
        st.subheader("通过瞄准线角度调整")
        new_aim_angle = st.slider(
            "瞄准线角度 (度)", -3.0, 0.0, float(aim_angle), 0.01)

        # 如果用户调整了角度滑块，重新计算瞄镜位置
        if abs(new_aim_angle - aim_angle) > 0.001:
            # 保持到窥孔的距离不变，仅改变角度
            distance = math.sqrt(
                (aim_x - config['xk'])**2 + (aim_y - config['yk'])**2)
            aim_x = config['xk'] + distance * \
                math.cos(math.radians(new_aim_angle))
            aim_y = config['yk'] + distance * \
                math.sin(math.radians(new_aim_angle))
            st.write(f"更新后的瞄镜位置 (S系): ({aim_x:.3f}, {aim_y:.3f})")

        # 计算按钮
        calculate_button = st.button("计算交点")

    with col2:
        st.header("计算结果区")

        if calculate_button or 'last_config' in st.session_state:
            # 保存当前配置到会话状态
            st.session_state.last_config = config
            st.session_state.last_aim_x = aim_x
            st.session_state.last_aim_y = aim_y

            # 绘制图表
            fig = plot_trajectory(aim_x, aim_y, config)
            st.pyplot(fig)

            # 计算交点
            intersections = calculate_intersection_points(aim_x, aim_y, config)

            # 显示交点信息
            st.subheader("交点信息")

            if not intersections:
                st.warning("无交点")
            else:
                data = []
                for i, (t, distance) in enumerate(intersections):
                    data.append({
                        "交点序号": i + 1,
                        "时间 (秒)": f"{t:.2f}",
                        "H系距离 (米)": f"{distance:.2f}"
                    })

                st.table(data)

            # 显示各种坐标系中的角度信息
            st.subheader("角度信息")
            st.write(f"箭的俯仰角 (H系): {math.degrees(config['theta']):.2f}度")
            st.write(f"瞄准线角度 (S系): {aim_angle:.2f}度")

            # 计算瞄准线在H系中的角度
            xk_h = config['xk'] * math.cos(config['theta']) - \
                config['yk'] * math.sin(config['theta'])
            yk_h = config['xk'] * math.sin(config['theta']) + \
                config['yk'] * math.cos(config['theta'])

            aim_x_h = aim_x * \
                math.cos(config['theta']) - aim_y * math.sin(config['theta'])
            aim_y_h = aim_x * \
                math.sin(config['theta']) + aim_y * math.cos(config['theta'])

            dx_h = aim_x_h - xk_h
            dy_h = aim_y_h - yk_h

            if abs(dx_h) < 1e-10:
                aim_angle_h = 90.0 if dy_h > 0 else -90.0
            else:
                aim_angle_h = math.degrees(math.atan2(dy_h, dx_h))

            st.write(f"瞄准线角度 (H系): {aim_angle_h:.2f}度")

            # 显示交点条件分析
            st.subheader("交点条件分析")
            condition_met = aim_angle < 0
            st.write("- 箭在S系中沿x轴运动（0度）")
            st.write(
                f"- 瞄准线在S系中的角度: {aim_angle:.2f}度 {'< 0度，满足条件' if condition_met else '≥ 0度，不满足条件'}")

            if not condition_met and not intersections:
                st.error("因瞄准线角度不满足条件，无交点")
            elif not intersections:
                st.warning("虽然瞄准线角度满足条件，但仍然无交点，可能是因为箭的速度不足或其他参数设置问题")


if __name__ == "__main__":
    main()
