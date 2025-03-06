import os
import json
import glob
import argparse
import matplotlib.pyplot as plt
import numpy as np


def compare_json_files(output_file="comparison.png", x_key="step", y_key="mean_return", smooth_factor=0):
    """
    比较文件夹中所有JSON文件的step和mean_return数据

    参数:
        folder: 包含JSON文件的文件夹
        output_file: 输出图像文件路径
        x_key: 用于X轴的JSON键名
        y_key: 用于Y轴的JSON键名
        smooth_factor: 平滑因子 (0表示不平滑)
    """
    # 查找所有JSON文件
    json_files = glob.glob(os.path.join("", "*.json"))

    # 按字母顺序排序文件
    json_files.sort()

    plt.figure(figsize=(12, 8))

    # 使用不同的颜色和标记
    colors = plt.cm.tab10(np.linspace(0, 1, len(json_files)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

    for i, json_file in enumerate(json_files):
        file_name = os.path.basename(json_file).replace('.json', '')

        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # 提取数据
            steps = []
            returns = []

            # 处理不同的JSON结构
            if isinstance(data, list):
                # 如果是列表格式，遍历每个元素
                for item in data:
                    if x_key in item and y_key in item:
                        steps.append(item[x_key])
                        returns.append(item[y_key])
            elif isinstance(data, dict):
                # 如果是字典格式
                if x_key in data and y_key in data:
                    steps.append(data[x_key])
                    returns.append(data[y_key])

            # 确保有数据
            if not steps or not returns:
                print(f"警告: 在 {json_file} 中未找到 {x_key} 或 {y_key} 数据")
                continue

            # 排序数据点
            if len(steps) > 1:
                sorted_pairs = sorted(zip(steps, returns))
                steps, returns = zip(*sorted_pairs)

            # 应用平滑处理
            if smooth_factor > 0 and len(returns) > 2:
                smoothed_returns = []
                window = max(2, int(len(returns) * smooth_factor))
                for j in range(len(returns)):
                    start = max(0, j - window // 2)
                    end = min(len(returns), j + window // 2 + 1)
                    smoothed_returns.append(np.mean(returns[start:end]))

                # 绘制原始数据（半透明）和平滑数据
                plt.plot(steps, returns, alpha=0.3, color=colors[i])
                plt.plot(steps, smoothed_returns, label=file_name,
                         marker=markers[i % len(markers)], markersize=8,
                         markevery=max(1, len(steps) // 10),
                         linewidth=2, color=colors[i])
            else:
                # 直接绘制原始数据
                plt.plot(steps, returns, label=file_name,
                         marker=markers[i % len(markers)], markersize=8,
                         markevery=max(1, len(steps) // 10),
                         linewidth=2, color=colors[i])

        except Exception as e:
            print(f"处理 {json_file} 时出错: {e}")

    # 添加图表元素
    plt.title(f"{y_key} vs {x_key}", fontsize=16)
    plt.xlabel(x_key.capitalize(), fontsize=14)
    plt.ylabel(y_key.capitalize(), fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best', fontsize=12)

    # 保存图表
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"图表已保存至 {output_file}")

    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="比较文件夹中所有JSON文件的数据")
    parser.add_argument("--output", type=str, default="comparison.png", help="输出文件路径")
    parser.add_argument("--x-key", type=str, default="step", help="用于X轴的JSON键")
    parser.add_argument("--y-key", type=str, default="mean_return", help="用于Y轴的JSON键")
    parser.add_argument("--smooth", type=float, default=0.0, help="平滑因子 (0-1)")

    args = parser.parse_args()

    compare_json_files(args.output, args.x_key, args.y_key, args.smooth)