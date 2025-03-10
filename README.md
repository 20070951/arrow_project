# 箭轨迹计算器

这是一个用于计算和可视化弓箭轨迹的交互式Web应用程序。该应用程序可以模拟箭的飞行轨迹，计算箭与瞄准线的交点，并提供详细的分析结果。

## 功能特点

- 可视化箭的飞行轨迹和瞄准线
- 计算箭轨迹与瞄准线的所有交点
- 支持调整箭的参数（出弓速度、俯仰角等）
- 支持直接调整瞄镜在S系中的位置
- 提供S系和H系两种坐标系下的角度分析
- 显示交点条件分析结果

## 坐标系统说明

本程序使用两个坐标系来描述箭的飞行：

- **S系 (箭轴坐标系)**: 以箭为参考系的坐标系，箭的飞行方向始终沿着S系的x轴正方向（0度角）。
- **H系 (真实世界坐标系)**: 以地面为参考的真实世界坐标系，重力方向沿着y轴负方向。
- **θ (theta)**: S系相对于H系的旋转角度，同时也是箭在H系中的俯仰角。

## 安装说明

1. 确保已安装Python 3.7或更高版本
2. 安装必要的依赖：

```bash
pip install -r requirements.txt
```

## 使用说明

1. 启动应用程序：

```bash
streamlit run arrow_gui.py
```

2. 在Web浏览器中打开显示的URL（通常是http://localhost:8501）
3. 使用左侧配置区域调整参数：
   - 调整箭的参数（出弓速度、俯仰角、重力加速度）
   - 设置坐标系中的位置（窥孔位置、箭头初始位置）
   - 调整瞄镜位置或瞄准线角度
4. 点击"计算交点"按钮查看结果
5. 在右侧结果区域查看：
   - 箭轨迹与瞄准线的可视化图表
   - 交点信息（时间和距离）
   - 角度信息（S系和H系）
   - 交点条件分析

## 参数说明

- **出弓速度 (v0)**: 箭离开弓时的初始速度，单位为米/秒
- **俯仰角 (theta)**: 箭在H系中的发射角度，单位为度
- **重力加速度 (g)**: 通常设置为9.8米/秒²
- **窥孔位置**: 瞄准器中窥孔在S系中的坐标
- **箭头初始位置**: 箭头在S系中的初始坐标
- **瞄镜位置**: 瞄准器中瞄镜在S系中的坐标
- **瞄准线角度**: 从窥孔到瞄镜连线与水平线的夹角，单位为度

## 交点条件

要形成交点，需要满足以下条件：
- 箭在S系中沿x轴运动（0度）
- 瞄准线在S系中的角度小于0度

## 程序文件

- **arrow_gui.py**: 主程序，包含Streamlit界面和计算逻辑
- **config.py**: 配置文件，包含默认参数设置
- **arrow_trajectory.py**: 箭轨迹计算模块 