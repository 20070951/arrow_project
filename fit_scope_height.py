"""
瞄镜高度与距离关系拟合工具

本工具尝试多种两参数函数形式拟合距离(d)与瞄镜高度(ym)之间的关系，
并对比分析找出最佳拟合模型。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import sys
import os

# 创建结果文件夹
RESULT_DIR = "result"
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)
    print(f"已创建结果文件夹: {RESULT_DIR}/")

# 导入标定表生成函数
try:
    from simple_calibration import create_calibration_table
    HAVE_CALIBRATION_MODULE = True
except ImportError:
    print("警告: 无法导入simple_calibration模块，将无法使用自动生成标定数据功能")
    HAVE_CALIBRATION_MODULE = False

# 预设参数列表 - 多组h和yk值
# 格式: [(h1, yk1), (h2, yk2), ...]
PARAMETER_SETS = [
    (0.3, 0.2),   # 例: h=0.15m, yk=0.05m
    (0.1, 0.35),   # 例: h=0.20m, yk=0.10m
    (0.25, 0.15),   # 例: h=0.25m, yk=0.15m
]

# 标定距离范围
DEFAULT_DISTANCE_RANGE = (20, 100)
DEFAULT_STEP = 1

# 定义各种两参数拟合模型


def linear_model(d, a, b):
    """线性模型: ym = a*d + b"""
    return a * d + b


def power_model(d, a, b):
    """幂函数模型: ym = a * d^(-b)"""
    return a * d**(-b)


def exponential_model(d, a, b):
    """指数模型: ym = a * exp(-b*d)"""
    # 添加溢出保护
    with np.errstate(over='ignore', invalid='ignore'):
        return a * np.exp(-b * d)


def hyperbolic_model(d, a, b):
    """双曲线模型: ym = a / (d + b)"""
    return a / (d + b)


def logarithmic_model(d, a, b):
    """对数模型: ym = a - b * log(d)"""
    return a - b * np.log(d)


def rational_model(d, a, b):
    """有理函数模型: ym = a / (1 + b*d)"""
    return a / (1 + b * d)


def inverse_model(d, a, b):
    """反比例变形模型: ym = a * (1 - b/d)"""
    # 注意：此模型在d接近0时不稳定
    return a * (1 - b/d)


def quadratic_model(d, a, b):
    """二次模型: ym = a*d^2 + b"""
    return a * d**2 + b


def sqrt_model(d, a, b):
    """平方根模型: ym = a*sqrt(d) + b"""
    return a * np.sqrt(d) + b


def inverse_square_model(d, a, b):
    """倒数平方模型: ym = a/(d^2) + b"""
    return a / (d**2) + b


def log_square_model(d, a, b):
    """对数平方模型: ym = a*(log(d))^2 + b"""
    return a * (np.log(d))**2 + b


def exponential_saturation_model(d, a, b):
    """指数饱和模型: ym = a*(1-exp(-b*d))"""
    # 增加稳定性
    with np.errstate(over='ignore', invalid='ignore'):
        return a * (1 - np.exp(-b * d))


def reciprocal_log_model(d, a, b):
    """倒数对数模型: ym = a/(1 + b*log(d))"""
    return a / (1 + b * np.log(d))


def compound_model(d, a, b):
    """混合模型: ym = a*d + b/d"""
    return a * d + b / d


def load_data_from_calibration_table(distance_range=(20, 100), step=1, h=None, yk=None):
    """
    从create_calibration_table函数获取标定数据

    参数：
        distance_range: 距离范围元组 (min_distance, max_distance)
        step: 步长
        h: 靶心与箭头高度差 (m)
        yk: 窥孔高度 (m)

    返回：
        tuple: (distances, heights) - 距离数组和对应的瞄镜高度数组
    """
    if not HAVE_CALIBRATION_MODULE:
        print("错误: 未导入simple_calibration模块，无法使用此功能")
        return None, None

    try:
        # 直接调用修改后的create_calibration_table函数，并传入h和yk参数
        calibration_data = create_calibration_table(
            distance_range, step, h, yk)

        # 提取距离和瞄镜高度数据
        distances = np.array([data['distance'] for data in calibration_data])
        heights = np.array([data['ym'] for data in calibration_data])

        print(f"成功从标定表生成了 {len(distances)} 个数据点")
        print(f"距离范围: {min(distances)}m - {max(distances)}m, 步长: {step}m")
        if h is not None and yk is not None:
            print(f"使用参数: h={h}m, yk={yk}m")

        return distances, heights
    except Exception as e:
        print(f"从标定表获取数据失败: {str(e)}")
        return None, None


def load_data_from_clipboard():
    """从剪贴板加载数据，假设已复制包含距离和瞄镜高度的表格数据"""
    try:
        df = pd.read_clipboard(sep='\s+', header=None)
        # 尝试自动识别数据列
        if df.shape[1] >= 3:  # 假设至少有3列
            # 通常第1列是距离，第3列是瞄镜高度
            distances = df.iloc[:, 0].astype(float).values
            heights = df.iloc[:, 2].astype(float).values
            return distances, heights
        else:
            print("剪贴板数据格式不正确，请按正确格式输入数据")
            return None, None
    except Exception as e:
        print(f"从剪贴板读取数据失败: {e}")
        return None, None


def load_data_from_input():
    """手动输入数据点"""
    print("请输入一系列距离和瞄镜高度数据点")
    print("每行输入一对 '距离 瞄镜高度'，输入空行结束")

    distances = []
    heights = []

    while True:
        line = input("距离 瞄镜高度 (空行结束): ").strip()
        if not line:
            break

        try:
            values = line.split()
            if len(values) >= 2:
                d = float(values[0])
                ym = float(values[1])
                distances.append(d)
                heights.append(ym)
            else:
                print("输入格式错误，请按'距离 瞄镜高度'格式输入")
        except ValueError:
            print("输入的不是有效数字，请重试")

    return np.array(distances), np.array(heights)


def load_data_from_file(filename):
    """从文件加载数据"""
    try:
        data = np.loadtxt(filename, delimiter=',', skiprows=1)
        distances = data[:, 0]
        heights = data[:, 1]
        return distances, heights
    except Exception as e:
        print(f"从文件加载数据失败: {e}")
        return None, None


def fit_models(distances, heights):
    """使用不同模型拟合数据并计算拟合优度"""
    models = {
        "线性模型 (ym = a*d + b)": linear_model,
        "幂函数模型 (ym = a * d^(-b))": power_model,
        "双曲线模型 (ym = a / (d + b))": hyperbolic_model,
        "对数模型 (ym = a - b * log(d))": logarithmic_model,
        "有理函数模型 (ym = a / (1 + b*d))": rational_model,
        "二次模型 (ym = a*d^2 + b)": quadratic_model,
        "平方根模型 (ym = a*sqrt(d) + b)": sqrt_model,
        "倒数平方模型 (ym = a/(d^2) + b)": inverse_square_model,
        "对数平方模型 (ym = a*(log(d))^2 + b)": log_square_model,
        "指数饱和模型 (ym = a*(1-exp(-b*d)))": exponential_saturation_model,
        "倒数对数模型 (ym = a/(1 + b*log(d)))": reciprocal_log_model,
        "混合模型 (ym = a*d + b/d)": compound_model,
    }

    if np.min(distances) > 10:  # 确保距离足够大，避免反比例模型在小距离时不稳定
        models["反比例变形模型 (ym = a * (1 - b/d))"] = inverse_model

    results = {}

    for name, model_func in models.items():
        try:
            # 尝试拟合模型
            params, covariance = curve_fit(model_func, distances, heights)

            # 计算预测值
            y_pred = model_func(distances, *params)

            # 计算R方值(拟合优度)
            ss_tot = np.sum((heights - np.mean(heights))**2)
            ss_res = np.sum((heights - y_pred)**2)
            r_squared = 1 - (ss_res / ss_tot)

            # 计算均方根误差(RMSE)
            rmse = np.sqrt(np.mean((y_pred - heights)**2))

            # 保存结果
            results[name] = {
                'params': params,
                'r_squared': r_squared,
                'rmse': rmse,
                'model_func': model_func
            }

        except Exception as e:
            print(f"{name} 拟合失败: {e}")

    return results


def plot_results(distances, heights, results, param_suffix=""):
    """
    绘制拟合结果

    param_suffix: 附加到文件名和标题的参数后缀
    """
    # 配置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei',
                                       'KaiTi', 'STSong', 'NSimSun'] + plt.rcParams['font.sans-serif']
    plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题

    plt.figure(figsize=(12, 8))

    # 绘制原始数据点
    plt.scatter(distances, heights, color='black', label='原始数据')

    # 生成用于绘制拟合曲线的x值(比数据范围更广一些)
    d_min, d_max = np.min(distances), np.max(distances)
    x_fit = np.linspace(d_min * 0.9, d_max * 1.1, 1000)

    # 按R²值排序
    sorted_results = sorted(
        results.items(), key=lambda x: x[1]['r_squared'], reverse=True)

    # 获取最佳模型的R²值
    if sorted_results:
        best_r_squared = sorted_results[0][1]['r_squared']
    else:
        best_r_squared = 0

    # 设置R²过滤阈值 - 只显示R²高于此阈值的模型
    # 如果最佳模型的R²非常高(>0.99)，则使用更严格的阈值，否则使用较宽松的阈值
    r_squared_threshold = 0.98 if best_r_squared > 0.99 else 0.9

    # 过滤掉R²值低于阈值的模型
    filtered_results = [(name, result) for name, result in sorted_results
                        if result['r_squared'] >= r_squared_threshold]

    # 如果过滤后没有足够的模型，至少保留前3个最佳模型
    if len(filtered_results) < min(3, len(sorted_results)):
        filtered_results = sorted_results[:min(3, len(sorted_results))]

    # 颜色列表
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']

    # 绘制每个模型的拟合曲线
    for i, (name, result) in enumerate(filtered_results):
        model_func = result['model_func']
        params = result['params']
        r_squared = result['r_squared']

        # 计算拟合曲线
        try:
            y_fit = model_func(x_fit, *params)
            plt.plot(x_fit, y_fit, color=colors[i % len(colors)],
                     label=f"{name} (R^2={r_squared:.4f})")
        except Exception as e:
            print(f"绘制 {name} 曲线失败: {e}")

    plt.xlabel('距离 (m)')
    plt.ylabel('瞄镜高度 (m)')

    title = '距离与瞄镜高度关系的不同模型拟合'
    if param_suffix:
        title += f" {param_suffix}"

    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()

    # 保存图像到结果文件夹
    filename = 'scope_height_fitting'
    if param_suffix:
        filename += f"_{param_suffix.replace(' ', '_').replace('=', '')}"
    save_path = os.path.join(RESULT_DIR, f'{filename}.png')
    plt.savefig(save_path, dpi=300)
    plt.close()  # 关闭图表，避免显示
    print(f"图表已保存为: {save_path}")


def generate_table(best_model, distances, min_dist=20, max_dist=100, step=1):
    """使用最佳模型生成瞄镜高度表"""
    model_func = best_model['model_func']
    params = best_model['params']

    # 生成距离范围
    dist_range = np.arange(min_dist, max_dist + step, step)

    # 计算对应的瞄镜高度
    heights = model_func(dist_range, *params)

    # 返回表格数据
    return list(zip(dist_range, heights))


def save_table_to_file(table_data, filename="fitted_scope_heights.csv"):
    """将表格保存到文件"""
    # 确保文件名包含路径
    if not filename.startswith(RESULT_DIR):
        filename = os.path.join(RESULT_DIR, filename)

    with open(filename, 'w', encoding='utf-8') as f:
        f.write("距离(m),瞄镜高度(m)\n")
        for dist, height in table_data:
            f.write(f"{dist:.1f},{height:.6f}\n")

    print(f"表格已保存到 {filename}")


def predict_height(best_model, distance):
    """使用最佳模型预测特定距离的瞄镜高度"""
    model_func = best_model['model_func']
    params = best_model['params']

    height = model_func(distance, *params)
    return height


def process_parameter_set(h, yk, distance_range=DEFAULT_DISTANCE_RANGE, step=DEFAULT_STEP):
    """处理单组参数(h, yk)的完整流程"""
    param_suffix = f"h={h}_yk={yk}"
    print(f"\n===== 处理参数集: h={h}, yk={yk} =====")

    # 从标定函数生成数据
    distances, heights = load_data_from_calibration_table(
        distance_range, step, h, yk)

    if distances is None or len(distances) < 2:
        print(f"参数集 {param_suffix} 数据生成失败或数据点不足")
        return None  # 返回None表示处理失败

    print(f"成功加载 {len(distances)} 个数据点")

    # 拟合模型
    print("\n开始拟合模型...\n")
    results = fit_models(distances, heights)

    if not results:
        print(f"参数集 {param_suffix} 所有模型拟合失败")
        return None  # 返回None表示处理失败

    # 选择最佳模型(R²最高的)
    best_model_name, best_model = max(
        results.items(), key=lambda x: x[1]['r_squared'])

    # 只输出最佳模型信息
    print(f"\n最佳拟合模型: {best_model_name}")
    print(f"参数: {best_model['params']}")
    print(f"R^2: {best_model['r_squared']:.6f}")

    # 绘制结果
    plot_results(distances, heights, results, param_suffix)

    # 生成表格并保存
    table_data = generate_table(best_model, distances,
                                min_dist=distance_range[0],
                                max_dist=distance_range[1],
                                step=step)

    # 保存表格
    filename = f"fitted_scope_heights_{param_suffix.replace(' ', '_').replace('=', '')}.csv"
    save_table_to_file(table_data, filename)

    # 返回最佳模型信息
    return {
        'h': h,
        'yk': yk,
        'best_model_name': best_model_name,
        'r_squared': best_model['r_squared'],
        'params': best_model['params']
    }


def batch_process():
    """批量处理所有预设参数集"""
    if not HAVE_CALIBRATION_MODULE:
        print("错误: 未能导入simple_calibration模块，无法使用自动生成标定数据功能")
        return

    print(f"===== 批量处理 {len(PARAMETER_SETS)} 组参数 =====")

    # 收集所有参数集的处理结果
    results_summary = []

    for i, (h, yk) in enumerate(PARAMETER_SETS):
        print(f"\n[{i+1}/{len(PARAMETER_SETS)}] 处理参数集: h={h}, yk={yk}")
        result = process_parameter_set(h, yk)
        if result:
            results_summary.append(result)

    print("\n===== 批量处理完成 =====")

    # 显示总结信息
    if results_summary:
        print("\n===== 参数拟合总结 =====")
        print(f"{'h值(m)':^10} | {'yk值(m)':^10} | {'最佳拟合模型':^40} | {'R²值':^10}")
        print("-" * 75)

        for result in results_summary:
            h = result['h']
            yk = result['yk']
            model_name = result['best_model_name'].split(
                ' (')[0]  # 只取模型名称，不要公式部分
            r_squared = result['r_squared']

            print(f"{h:^10.3f} | {yk:^10.3f} | {model_name:^40} | {r_squared:^10.6f}")
    else:
        print("\n没有成功拟合的参数组。")


def main():
    print("===== 瞄镜高度与距离关系拟合工具 =====")
    print("本工具将尝试多种两参数函数形式拟合距离与瞄镜高度的关系\n")

    # 批量处理预设参数
    batch_process()


if __name__ == "__main__":
    main()
