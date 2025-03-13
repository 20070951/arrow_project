"""
弓箭瞄镜高度计算器
基于simple_calibration算法的Streamlit应用程序
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 添加项目根目录到路径以导入config和simple_calibration
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 导入配置和校准函数
try:
    import config
    from simple_calibration import calculate_theta, calculate_ym, create_calibration_table
except ImportError as e:
    st.error(f"导入模块失败: {e}")
    st.info("请确保config.py和simple_calibration.py在同一目录下")
    st.stop()

# 设置页面标题
st.set_page_config(
    page_title="弓箭瞄镜高度计算器",
    page_icon="🎯",
    layout="wide"
)

# 标题和介绍
st.title("🎯 弓箭瞄镜高度计算器")
st.markdown("""
此应用根据您的弓箭参数和两次实验数据，计算出不同距离的瞄镜高度表。
使用基于物理模型的校准算法，帮助提高您的射箭精度。
""")

# 主布局
col1, col2 = st.columns([1, 3])  # 更改列比例，左侧占比减小

# 侧边栏：用户参数输入
with col1:
    st.header("参数设置")

    # 用户参数模块
    with st.expander("射手和设备参数", expanded=True):
        # 使用config中的默认值，并确保都是float类型
        arm_span = st.number_input("臂展 (m)",
                                   min_value=0.5, max_value=2.5,
                                   value=float(
                                       getattr(config, 'arm_length', 1.7)),
                                   step=0.01,
                                   format="%.2f")

        bow_length = st.number_input("弓长 (m)",
                                     min_value=0.5, max_value=2.0,
                                     value=float(
                                         getattr(config, 'bow_length', 1.5)),
                                     step=0.01,
                                     format="%.2f")

        arrow_speed = st.number_input("箭的初速度 (m/s)",
                                      min_value=10.0, max_value=100.0,
                                      value=float(getattr(config, 'v0', 50.0)),
                                      step=0.5,
                                      format="%.1f")

        # 弓固定参数
        st.subheader("弓的固定参数")
        bow_a = st.number_input("弓参数 a",
                                min_value=0.0, max_value=10.0,
                                value=float(getattr(config, 'bow_a', 1.0)),
                                step=0.01,
                                format="%.2f")

        bow_b = st.number_input("弓参数 b",
                                min_value=0.0, max_value=10.0,
                                value=float(getattr(config, 'bow_b', 1.0)),
                                step=0.01,
                                format="%.2f")


# 逆向校准函数，使用两个实验点来估计h和yk
def inverse_calibration(d1, ym1, d2, ym2, v0, arm_length, bow_length, bow_a, bow_b):
    """
    通过两个实验点估计h和yk参数

    参数:
        d1, ym1: 第一个实验点的距离和瞄镜高度
        d2, ym2: 第二个实验点的瞄镜高度
        其他参数: 弓箭配置参数

    返回:
        估计的h和yk值
    """
    # 简单线性近似法
    # 假设瞄镜高度(ym)与距离(d)近似为线性关系
    slope = (ym2 - ym1) / (d2 - d1)

    # 起始猜测值
    yk_guess = 0.15
    h_guess = abs(slope) * (d1 * d2) / (d1 - d2) * 2

    # 更新配置参数，以便后续计算使用
    config.arm_length = float(arm_length)
    config.bow_length = float(bow_length)
    config.v0 = float(v0)
    config.bow_a = float(bow_a)
    config.bow_b = float(bow_b)

    # 微调h和yk，使得计算的ym更接近实验值
    best_error = float('inf')
    best_h = h_guess
    best_yk = yk_guess

    # 网格搜索最佳参数
    for h_mult in np.linspace(0.5, 1.5, 10):
        for yk_mult in np.linspace(0.5, 1.5, 10):
            h_test = h_guess * h_mult
            yk_test = yk_guess * yk_mult

            # 计算这组参数下的预测值
            config.h = float(h_test)
            config.yk = float(yk_test)

            # 注意：calculate_ym需要先计算theta
            theta1 = calculate_theta(d1)
            theta2 = calculate_theta(d2)

            ym1_calc = calculate_ym(d1, theta1)
            ym2_calc = calculate_ym(d2, theta2)

            # 计算误差
            error = (ym1 - ym1_calc)**2 + (ym2 - ym2_calc)**2

            if error < best_error:
                best_error = error
                best_h = h_test
                best_yk = yk_test

    # 细化搜索
    for _ in range(5):
        h_range = np.linspace(best_h * 0.9, best_h * 1.1, 10)
        yk_range = np.linspace(best_yk * 0.9, best_yk * 1.1, 10)

        for h_test in h_range:
            for yk_test in yk_range:
                config.h = float(h_test)
                config.yk = float(yk_test)

                theta1 = calculate_theta(d1)
                theta2 = calculate_theta(d2)

                ym1_calc = calculate_ym(d1, theta1)
                ym2_calc = calculate_ym(d2, theta2)

                error = (ym1 - ym1_calc)**2 + (ym2 - ym2_calc)**2

                if error < best_error:
                    best_error = error
                    best_h = h_test
                    best_yk = yk_test

    return best_h, best_yk


# 结果显示部分
with col2:
    st.header("瞄镜高度计算结果")

    # 实验数据输入 - 横向排列两次实验
    st.subheader("实验数据输入")
    with st.container():
        st.info("请输入两次实验的测量数据")

        # 将实验1和实验2并排放置
        exp_col1, exp_col2 = st.columns(2)

        # 实验1
        with exp_col1:
            st.markdown("**实验 1**")
            d1 = st.number_input("距离 d1 (m)",
                                 min_value=10.0, max_value=90.0,
                                 value=20.0,
                                 step=1.0,
                                 key="d1")

            ym1_mm = st.number_input("瞄镜高度 ym1 (mm)",
                                     min_value=0.0, max_value=1000.0,
                                     value=150.0,  # 默认0.15m = 150mm
                                     step=1.0,
                                     format="%.1f",
                                     key="ym1_mm")
            # 转换为米单位用于计算
            ym1 = ym1_mm / 1000.0

        # 实验2
        with exp_col2:
            st.markdown("**实验 2**")
            d2 = st.number_input("距离 d2 (m)",
                                 min_value=10.0, max_value=90.0,
                                 value=50.0,
                                 step=1.0,
                                 key="d2")

            ym2_mm = st.number_input("瞄镜高度 ym2 (mm)",
                                     min_value=0.0, max_value=1000.0,
                                     value=80.0,  # 默认0.08m = 80mm
                                     step=1.0,
                                     format="%.1f",
                                     key="ym2_mm")
            # 转换为米单位用于计算
            ym2 = ym2_mm / 1000.0

    # 计算设置 - 横向放在计算按钮上方
    st.subheader("计算设置")
    with st.container():
        # 使用三列布局
        setting_col1, setting_col2, setting_col3 = st.columns(3)

        with setting_col1:
            st.markdown("**最近靶位 (m)**")
            min_distance = st.number_input("最近靶位输入",
                                           min_value=10.0, max_value=50.0,
                                           value=10.0,
                                           step=1.0,
                                           label_visibility="collapsed")

        with setting_col2:
            st.markdown("**最远靶位 (m)**")
            max_distance = st.number_input("最远靶位输入",
                                           min_value=50.0, max_value=200.0,
                                           value=100.0,
                                           step=1.0,
                                           label_visibility="collapsed")

        with setting_col3:
            st.markdown("**靶位间隔 (m)**")
            step_size = st.number_input("靶位间隔输入",
                                        min_value=1.0, max_value=10.0,
                                        value=1.0,
                                        step=1.0,
                                        label_visibility="collapsed")

    # 计算按钮
    calculate_button = st.button(
        "计算瞄镜高度表", type="primary", use_container_width=True)

    if calculate_button:
        st.info("正在根据输入参数和实验数据计算...")

        # 检查两个实验点是否相同
        if abs(d1 - d2) < 1e-6:
            st.error("两次实验距离必须不同!")
        else:
            try:
                # 使用逆向校准估计参数
                h_estimated, yk_estimated = inverse_calibration(
                    d1, ym1, d2, ym2,
                    arrow_speed, arm_span, bow_length, bow_a, bow_b
                )

                st.success(
                    f"成功估计物理参数: h = {h_estimated:.3f} m, yk = {yk_estimated:.3f} m")

                # 生成校准表
                table_data = create_calibration_table(
                    (min_distance, max_distance),
                    step_size,
                    h_estimated,
                    yk_estimated
                )

                # 创建pandas DataFrame - 使用英文列名并将瞄镜高度转换为mm
                df = pd.DataFrame([
                    {
                        'Distance (m)': float(data['distance']),
                        # 转换为mm
                        'Scope Height (mm)': float(data['ym']) * 1000,
                        'Elevation Angle (deg)': float(data['theta_degrees']),
                        'Flight Time (s)': float(d / arrow_speed)  # 简化计算飞行时间
                    } for d, data in [(data['distance'], data) for data in table_data]
                ])

                # 显示表格 - 创建中文到英文列名的映射用于显示
                column_name_map = {
                    'Distance (m)': '距离 (m)',
                    'Scope Height (mm)': '瞄镜高度 (mm)',
                    'Elevation Angle (deg)': '俯仰角 (度)',
                    'Flight Time (s)': '飞行时间 (s)'
                }

                # 在界面上显示表格时用中文列名
                display_df = df.copy()
                display_df.columns = [column_name_map[col]
                                      for col in df.columns]

                st.subheader("瞄镜高度表")
                st.dataframe(display_df.style.format({
                    '距离 (m)': '{:.1f}',
                    '瞄镜高度 (mm)': '{:.1f}',  # 毫米显示一位小数
                    '俯仰角 (度)': '{:.2f}',
                    '飞行时间 (s)': '{:.3f}'
                }), use_container_width=True, height=600)

                # 下载按钮 - 导出的CSV使用英文列名
                csv = df.to_csv(index=False)
                st.download_button(
                    label="下载瞄镜高度表为CSV",
                    data=csv,
                    file_name="scope_height_table.csv",
                    mime="text/csv",
                )

            except Exception as e:
                st.error(f"计算过程中出现错误: {str(e)}")
                st.info("请检查输入参数和实验数据是否合理")
    else:
        st.info("请输入参数和实验数据，然后点击'计算瞄镜高度表'按钮")

# 应用说明
st.markdown("---")
st.header("使用说明")
st.markdown("""
### 如何使用此应用:
1. 在左侧输入您的个人参数（臂展、弓长等）
2. 输入两次射箭实验的结果（距离和瞄镜高度）
3. 设置想要生成的瞄镜高度表范围和间隔
4. 点击"计算瞄镜高度表"按钮
5. 查看并下载生成的瞄镜高度表

### 注意事项:
- 两次实验的距离必须不同，且相差越大越好
- 确保所有输入的参数都在合理的物理范围内
- 如有任何问题，可以尝试使用不同的实验数据点
""")

# 页脚
st.markdown("---")
st.markdown("*©2023 弓箭瞄镜高度计算器 | 基于物理模型的校准工具*")
