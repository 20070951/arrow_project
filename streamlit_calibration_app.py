"""
弓箭瞄镜高度计算器
基于两点校准的Streamlit应用程序
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 添加项目根目录到路径以导入config
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 导入配置
try:
    import config
except ImportError as e:
    st.error(f"导入模块失败: {e}")
    st.info("请确保config.py在同一目录下")
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
此应用根据您的两次实验数据，计算出不同距离的瞄镜高度表。
帮助提高您的射箭精度。
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
                                       getattr(config, 'arm_length', 0.68)),
                                   step=0.01,
                                   format="%.2f")

        bow_length = st.number_input("弓长 (m)",
                                     min_value=0.5, max_value=2.0,
                                     value=float(
                                         getattr(config, 'bow_length', 0.7)),
                                     step=0.01,
                                     format="%.2f")

        arrow_speed = st.number_input("箭的初速度 (m/s)",
                                      min_value=10.0, max_value=100.0,
                                      value=float(getattr(config, 'v0', 50.0)),
                                      step=0.5,
                                      format="%.1f")

        # 弓的固定参数
        st.subheader("弓的固定参数")
        bow_a = st.number_input("弓参数 a",
                                min_value=0.01, max_value=0.5,
                                value=float(getattr(config, 'bow_a', 0.12)),
                                step=0.01,
                                format="%.2f")

        bow_b = st.number_input("弓参数 b",
                                min_value=0.01, max_value=0.5,
                                value=float(getattr(config, 'bow_b', 0.04)),
                                step=0.01,
                                format="%.2f")


# 简单线性校准函数
def linear_calibration(d1, ym1, d2, ym2):
    """
    通过两个实验点计算简单线性关系

    参数:
        d1, ym1: 第一个实验点的距离和瞄镜高度
        d2, ym2: 第二个实验点的瞄镜高度

    返回:
        斜率和截距
    """
    # 计算线性关系: ym = slope * distance + intercept
    slope = (ym2 - ym1) / (d2 - d1)
    intercept = ym1 - slope * d1

    return slope, intercept


# 生成线性校准表
def create_linear_calibration_table(distance_range, step, slope, intercept):
    """
    使用线性关系生成校准表

    参数:
        distance_range: (min_distance, max_distance)
        step: 距离步长
        slope, intercept: 线性关系参数

    返回:
        包含距离、瞄镜高度的数据列表
    """
    min_dist, max_dist = distance_range
    distances = np.arange(min_dist, max_dist + step, step)

    table_data = []
    for d in distances:
        # 使用线性方程计算瞄镜高度
        ym = slope * d + intercept

        # 使用简单近似公式计算俯仰角(度)
        elevation_angle = np.arctan(ym / d) * 180 / np.pi

        table_data.append({
            'distance': d,
            'ym': ym,
            'theta_degrees': elevation_angle
        })

    return table_data


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
                                        min_value=0.5, max_value=10.0,
                                        value=1.0,
                                        step=0.5,
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
                # 使用简单线性校准
                slope, intercept = linear_calibration(d1, ym1, d2, ym2)

                # 移除显示公式的部分，只显示简单成功消息
                st.success("计算完成！")

                # 生成校准表
                table_data = create_linear_calibration_table(
                    (min_distance, max_distance),
                    step_size,
                    slope,
                    intercept
                )

                # 创建pandas DataFrame - 使用英文列名并将瞄镜高度转换为mm
                df = pd.DataFrame([
                    {
                        'Distance (m)': float(data['distance']),
                        # 转换为mm
                        'Scope Height (mm)': float(data['ym']) * 1000,
                        'Elevation Angle (deg)': float(data['theta_degrees']),
                        # 简化计算飞行时间
                        'Flight Time (s)': float(data['distance'] / arrow_speed)
                    } for data in table_data
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

# 页脚
st.markdown("---")
st.markdown("*©2023 弓箭瞄镜高度计算器 | 基于两点校准的工具*")
