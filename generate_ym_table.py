"""
瞄准高度表生成工具

本工具通过两个标定点，计算任意距离的瞄准高度(ym)，并生成表格。
使用线性关系：ym = a*d + b
"""

import numpy as np


def calculate_linear_params(ym1, d1, ym2, d2):
    """
    根据两个标定点计算线性关系参数

    参数：
        ym1: 第一个标定点的瞄准高度(m)
        d1: 第一个标定点的距离(m)
        ym2: 第二个标定点的瞄准高度(m)
        d2: 第二个标定点的距离(m)

    返回：
        dict: 包含线性关系参数
    """
    # 检查输入
    if d1 == d2:
        raise ValueError("两个标定点的距离不能相同")

    # 计算线性关系: ym = a*d + b
    a = (ym2 - ym1) / (d2 - d1)  # 斜率
    b = ym1 - a * d1             # 截距

    return {'a': a, 'b': b}


def predict_ym(linear_params, distance):
    """
    计算给定距离的瞄准高度

    参数：
        linear_params: 线性关系参数
        distance: 目标距离(m)

    返回：
        float: 瞄准高度(m)
    """
    return linear_params['a'] * distance + linear_params['b']


def generate_ym_table(linear_params, min_dist, max_dist, step):
    """
    生成距离范围内的瞄准高度表

    参数：
        linear_params: 线性关系参数
        min_dist: 最小距离(m)
        max_dist: 最大距离(m)
        step: 步长(m)

    返回：
        list: 距离-瞄准高度对应表
    """
    table = []
    current_dist = min_dist

    while current_dist <= max_dist:
        ym = predict_ym(linear_params, current_dist)
        table.append((current_dist, ym))
        current_dist += step

    return table


def print_ym_table(table):
    """
    打印瞄准高度表

    参数：
        table: 距离-瞄准高度对应表
    """
    print("\n===== 瞄准高度表 =====")
    print(f"{'距离 (m)':^12} | {'瞄准高度 (m)':^15} | {'瞄准高度 (mm)':^15}")
    print("-" * 48)

    for dist, ym in table:
        print(f"{dist:^12.0f} | {ym:^15.6f} | {ym*1000:^15.1f}")


def main():
    print("===== 瞄准高度表生成工具 =====")
    print("请输入两个标定点的数据：\n")

    # 获取标定点数据
    try:
        d1 = float(input("输入第1个标定点的距离 (m): "))
        ym1 = float(input("输入第1个标定点的瞄准高度 (m): "))

        d2 = float(input("\n输入第2个标定点的距离 (m): "))
        ym2 = float(input("输入第2个标定点的瞄准高度 (m): "))

        # 计算线性关系
        linear_params = calculate_linear_params(ym1, d1, ym2, d2)
        a, b = linear_params['a'], linear_params['b']

        print(f"\n计算得到的线性关系: ym = {a:.6f} * d + {b:.6f}")

        # 获取表格范围
        min_dist = int(input("\n输入表格最小距离 (m): "))
        max_dist = int(input("输入表格最大距离 (m): "))
        step = int(input("输入距离步长 (m): "))

        # 生成并打印表格
        table = generate_ym_table(linear_params, min_dist, max_dist, step)
        print_ym_table(table)

        # 保存到文件
        save = input("\n是否将表格保存到文件？(y/n): ").lower().strip() == 'y'
        if save:
            filename = input(
                "输入文件名 (默认: ym_table.txt): ").strip() or "ym_table.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("距离 (m),瞄准高度 (m),瞄准高度 (mm)\n")
                for dist, ym in table:
                    f.write(f"{int(dist)},{ym:.6f},{ym*1000:.1f}\n")
            print(f"表格已保存到 {filename}")

        # 交互式查询
        print("\n是否需要查询特定距离的瞄准高度？(y/n): ", end="")
        if input().lower().strip() == 'y':
            while True:
                try:
                    query_dist = float(input("\n输入目标距离 (m)，输入负数退出: "))
                    if query_dist < 0:
                        break

                    ym = predict_ym(linear_params, query_dist)
                    print(
                        f"距离 {query_dist}m 的瞄准高度: {ym:.6f}m ({ym*1000:.1f}mm)")
                except ValueError:
                    print("输入无效，请输入数字")

    except ValueError as e:
        print(f"输入错误: {str(e)}")
    except Exception as e:
        print(f"程序出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
