# 弓箭瞄镜高度计算器

## 项目简介

弓箭瞄镜高度计算器是一个基于物理模型的弓箭瞄准辅助工具，可以根据实验数据自动生成不同距离下的瞄镜高度表。此工具使用Streamlit框架开发，提供简洁直观的用户界面，帮助射箭爱好者提高瞄准精度。

![应用截图](docs/screenshot.png)

## 主要功能

- **参数化物理模型**：基于物理原理，考虑箭的飞行轨迹、弓的特性等因素
- **逆向校准算法**：通过两次实验数据，自动估算关键物理参数
- **自定义计算范围**：可设置目标距离范围和间隔，生成个性化瞄镜高度表
- **数据可视化展示**：清晰展示计算结果，包括距离、瞄镜高度、俯仰角和飞行时间
- **CSV导出功能**：一键导出瞄镜高度表，方便在实地使用

## 安装指南

### 环境要求

- Python 3.8+
- pip包管理器

### 安装步骤

1. 克隆代码仓库

```bash
git clone https://github.com/yourusername/archery-scope-calculator.git
cd archery-scope-calculator
```

2. 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

1. 启动应用

```bash
streamlit run streamlit_calibration_app.py
```

2. 在浏览器中访问应用（通常为 http://localhost:8501）

3. 输入参数和实验数据：
   - 在左侧设置面板中输入弓箭参数（臂展、弓长、箭速等）
   - 在主界面输入两次实验的距离和瞄镜高度数据
   - 设置计算范围（最近距离、最远距离和间隔）

4. 点击"计算瞄镜高度表"按钮

5. 查看结果并下载CSV文件

## 项目结构

```
archery-scope-calculator/
├── streamlit_calibration_app.py  # Streamlit应用主程序
├── simple_calibration.py         # 校准算法实现
├── config.py                     # 配置参数
├── requirements.txt              # 项目依赖
├── docs/                         # 文档和图片
└── README.md                     # 项目说明
```

## 技术栈

- **Streamlit**：用于构建Web应用界面
- **Pandas**：数据处理和表格生成
- **NumPy**：科学计算和数值优化
- **Matplotlib**：（可选）数据可视化

## 物理模型说明

应用基于抛物线物理模型，考虑以下关键参数：

- **h**：目标中心与箭尖的高度差
- **yk**：与瞄准系统相关的物理参数
- **v0**：箭的初速度
- **arm_length**：射手臂展
- **bow_length**：弓的长度

通过两点校准法，系统能够估算出最佳的h和yk值，进而计算出不同距离的瞄镜高度。

## 许可证

本项目采用MIT许可证。详情请参阅LICENSE文件。

## 贡献指南

欢迎提交问题报告和功能建议。如果您想贡献代码，请遵循以下步骤：

1. Fork本仓库
2. 创建您的特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交您的更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 提交Pull Request

## 联系方式

如有问题或建议，请联系项目维护者：[your-email@example.com](mailto:your-email@example.com)

---

*©2023 弓箭瞄镜高度计算器 | 基于物理模型的校准工具* 