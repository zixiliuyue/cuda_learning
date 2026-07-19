"""
负载测试结果可视化工具
生成响应时间分布、QPS曲线、分位数对比等图表

功能:
    - 从 JSON 结果文件生成可视化图表
    - 从 Locust CSV 结果文件生成可视化图表
    - 支持多个结果文件对比
    - 生成交互式 HTML 报告

使用方法:
    python benchmark_plot.py --json result1.json result2.json --output report.html
    python benchmark_plot.py --csv results_stats.csv --output report.html
"""

import argparse
import json
import csv
import os
from datetime import datetime
from typing import List, Dict, Any
import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("警告: plotly 未安装，将使用 matplotlib")
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        MATPLOTLIB_AVAILABLE = True
    except ImportError:
        MATPLOTLIB_AVAILABLE = False
        print("错误: 请安装 plotly 或 matplotlib")
        print("  pip install plotly")
        print("  或")
        print("  pip install matplotlib seaborn")


class BenchmarkData:
    """基准测试数据容器"""

    def __init__(self, name: str):
        self.name = name
        self.response_times = []  # 响应时间列表
        self.qps = 0.0
        self.avg_rt = 0.0
        self.tp50 = 0.0
        self.tp90 = 0.0
        self.tp95 = 0.0
        self.tp99 = 0.0
        self.tp999 = 0.0
        self.success_rate = 0.0
        self.concurrency = 0


def load_from_json(filepath: str) -> BenchmarkData:
    """从 JSON 文件加载数据"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 处理 simple_loadtest 格式
    if 'response_times' in data:
        name = os.path.basename(filepath).replace('.json', '')
        result = BenchmarkData(name)
        rt_data = data['response_times']
        result.response_times = rt_data.get('raw', [])
        result.qps = data.get('qps', 0)
        result.avg_rt = rt_data.get('avg', 0)
        result.tp50 = rt_data.get('tp50', 0)
        result.tp90 = rt_data.get('tp90', 0)
        result.tp95 = rt_data.get('tp95', 0)
        result.tp99 = rt_data.get('tp99', 0)
        result.tp999 = rt_data.get('tp999', 0)
        result.success_rate = data.get('success_rate', 0)
        result.concurrency = data.get('concurrency', 0)
        return result

    # 处理对比格式 (包含 service1 和 service2)
    results = []
    for key in ['service1', 'service2']:
        if key in data:
            name = data[key].get('url', key)
            result = BenchmarkData(name)
            rt_data = data[key]['response_times']
            result.response_times = rt_data.get('raw', [])
            result.qps = data[key].get('qps', 0)
            result.avg_rt = rt_data.get('avg', 0)
            result.tp50 = rt_data.get('tp50', 0)
            result.tp90 = rt_data.get('tp90', 0)
            result.tp95 = rt_data.get('tp95', 0)
            result.tp99 = rt_data.get('tp99', 0)
            result.tp999 = rt_data.get('tp999', 0)
            result.success_rate = data[key].get('success_rate', 0)
            result.concurrency = data[key].get('concurrency', 0)
            results.append(result)

    return results if results else None


def load_from_csv(filepath: str) -> BenchmarkData:
    """从 Locust CSV 文件加载数据"""
    name = os.path.basename(filepath).replace('_stats.csv', '').replace('.csv', '')
    result = BenchmarkData(name)

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        response_times = []
        for row in reader:
            try:
                # Locust CSV 格式通常包含这些列
                if '50%' in row:
                    result.tp50 = float(row['50%'])
                if '90%' in row:
                    result.tp90 = float(row['90%'])
                if '95%' in row:
                    result.tp95 = float(row['95%'])
                if '99%' in row:
                    result.tp99 = float(row['99%'])
                if '99.9%' in row:
                    result.tp999 = float(row['99.9%'])
                if 'Average Response Time' in row:
                    result.avg_rt = float(row['Average Response Time'])
                if 'Requests/s' in row:
                    result.qps = float(row['Requests/s'])
                if 'Total Request Count' in row:
                    pass
            except (ValueError, KeyError):
                continue

    return result


def load_data(filepath: str) -> List[BenchmarkData]:
    """根据文件扩展名自动加载数据"""
    if filepath.endswith('.json'):
        data = load_from_json(filepath)
        if isinstance(data, list):
            return data
        return [data] if data else []
    elif filepath.endswith('.csv'):
        data = load_from_csv(filepath)
        return [data] if data else []
    return []


def generate_plotly_report(results: List[BenchmarkData], title: str = "负载测试性能报告") -> str:
    """使用 Plotly 生成交互式 HTML 报告"""

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            '响应时间分布',
            '分位数对比',
            'QPS 对比',
            '性能指标概览'
        ),
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "table"}]
        ]
    )

    # 1. 响应时间分布图 (左上)
    for i, result in enumerate(results):
        if result.response_times:
            fig.add_trace(
                go.Histogram(
                    x=result.response_times,
                    name=result.name,
                    opacity=0.6,
                    nbinsx=50,
                    marker_color=colors[i % len(colors)]
                ),
                row=1, col=1
            )

    fig.update_xaxes(title_text="响应时间 (ms)", row=1, col=1)
    fig.update_yaxes(title_text="请求数量", row=1, col=1)

    # 2. 分位数对比图 (右上)
    metrics = ['TP50', 'TP90', 'TP95', 'TP99', 'TP999']
    for i, result in enumerate(results):
        values = [result.tp50, result.tp90, result.tp95, result.tp99, result.tp999]
        fig.add_trace(
            go.Bar(
                x=metrics,
                y=values,
                name=result.name,
                marker_color=colors[i % len(colors)]
            ),
            row=1, col=2
        )

    fig.update_yaxes(title_text="响应时间 (ms)", row=1, col=2)

    # 3. QPS 对比图 (左下)
    names = [r.name for r in results]
    qps_values = [r.qps for r in results]

    fig.add_trace(
        go.Bar(
            x=names,
            y=qps_values,
            marker_color=colors[:len(results)]
        ),
        row=2, col=1
    )

    fig.update_yaxes(title_text="QPS (请求/秒)", row=2, col=1)

    # 4. 性能指标概览表格 (右下)
    header = ['指标'] + [r.name for r in results]
    cells = [
        ['QPS', '平均响应时间(ms)', 'TP50(ms)', 'TP90(ms)', 'TP95(ms)', 'TP99(ms)', 'TP999(ms)', '成功率(%)']
    ]

    for result in results:
        cells.append([
            f"{result.qps:.2f}",
            f"{result.avg_rt:.2f}",
            f"{result.tp50:.2f}",
            f"{result.tp90:.2f}",
            f"{result.tp95:.2f}",
            f"{result.tp99:.2f}",
            f"{result.tp999:.2f}",
            f"{result.success_rate:.2f}"
        ])

    cells = list(map(list, zip(*cells)))

    fig.add_trace(
        go.Table(
            header=dict(values=header, fill_color='#444', font=dict(color='white', size=12)),
            cells=dict(values=cells, fill_color='#f0f0f0', font=dict(size=11))
        ),
        row=2, col=2
    )

    fig.update_layout(
        height=800,
        title_text=title,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig.to_html(full_html=True, include_plotlyjs='cdn')


def generate_matplotlib_report(results: List[BenchmarkData], output_path: str, title: str):
    """使用 Matplotlib 生成静态图表"""
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig = plt.figure(figsize=(16, 12))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # 1. 响应时间分布
    ax1 = plt.subplot(2, 2, 1)
    for i, result in enumerate(results):
        if result.response_times:
            ax1.hist(result.response_times, bins=50, alpha=0.6, label=result.name, color=colors[i])
    ax1.set_xlabel('响应时间 (ms)')
    ax1.set_ylabel('请求数量')
    ax1.set_title('响应时间分布')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 分位数对比
    ax2 = plt.subplot(2, 2, 2)
    metrics = ['TP50', 'TP90', 'TP95', 'TP99', 'TP999']
    x = np.arange(len(metrics))
    width = 0.8 / len(results)

    for i, result in enumerate(results):
        values = [result.tp50, result.tp90, result.tp95, result.tp99, result.tp999]
        ax2.bar(x + i * width, values, width, label=result.name, color=colors[i], alpha=0.8)

    ax2.set_xlabel('分位指标')
    ax2.set_ylabel('响应时间 (ms)')
    ax2.set_title('分位数对比')
    ax2.set_xticks(x + width * (len(results) - 1) / 2)
    ax2.set_xticklabels(metrics)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. QPS 对比
    ax3 = plt.subplot(2, 2, 3)
    names = [r.name for r in results]
    qps_values = [r.qps for r in results]
    bars = ax3.bar(names, qps_values, color=colors[:len(results)], alpha=0.8)
    ax3.set_ylabel('QPS (请求/秒)')
    ax3.set_title('QPS 对比')
    ax3.grid(True, alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom')

    # 4. 性能指标表格
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('tight')
    ax4.axis('off')

    table_data = [
        ['指标'] + [r.name for r in results],
        ['QPS'] + [f'{r.qps:.2f}' for r in results],
        ['平均RT(ms)'] + [f'{r.avg_rt:.2f}' for r in results],
        ['TP50(ms)'] + [f'{r.tp50:.2f}' for r in results],
        ['TP90(ms)'] + [f'{r.tp90:.2f}' for r in results],
        ['TP95(ms)'] + [f'{r.tp95:.2f}' for r in results],
        ['TP99(ms)'] + [f'{r.tp99:.2f}' for r in results],
        ['TP999(ms)'] + [f'{r.tp999:.2f}' for r in results],
        ['成功率(%)'] + [f'{r.success_rate:.2f}' for r in results],
    ]

    table = ax4.table(cellText=table_data[1:], colLabels=table_data[0], loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    plt.suptitle(title, fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"图表已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="负载测试结果可视化工具")
    parser.add_argument('--json', nargs='+', help='一个或多个JSON结果文件')
    parser.add_argument('--csv', nargs='+', help='一个或多个CSV结果文件')
    parser.add_argument('--output', default='benchmark_report.html', help='输出文件路径')
    parser.add_argument('--title', default='负载测试性能报告', help='报告标题')

    args = parser.parse_args()

    if not args.json and not args.csv:
        parser.error("必须指定 --json 或 --csv 参数")

    all_results = []

    if args.json:
        for filepath in args.json:
            if os.path.exists(filepath):
                results = load_data(filepath)
                all_results.extend(results)
                print(f"已加载: {filepath} ({len(results)} 个数据集")
            else:
                print(f"警告: 文件不存在 - {filepath}")

    if args.csv:
        for filepath in args.csv:
            if os.path.exists(filepath):
                results = load_data(filepath)
                all_results.extend(results)
                print(f"已加载: {filepath} ({len(results)} 个数据集")
            else:
                print(f"警告: 文件不存在 - {filepath}")

    if not all_results:
        print("错误: 没有加载到有效数据")
        return

    print(f"\n总共加载了 {len(all_results)} 个数据集")

    if PLOTLY_AVAILABLE and args.output.endswith('.html'):
        html_content = generate_plotly_report(all_results, args.title)
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"\n交互式报告已生成: {args.output}")
        print(f"使用浏览器打开即可查看图表")
    elif MATPLOTLIB_AVAILABLE:
        if args.output.endswith('.html'):
            args.output = args.output.replace('.html', '.png')
        generate_matplotlib_report(all_results, args.output, args.title)
    else:
        print("\n错误: 没有可用的绘图库")
        print("请安装:")
        print("  pip install plotly  (推荐，支持交互式图表)")
        print("  或")
        print("  pip install matplotlib seaborn")


if __name__ == "__main__":
    main()
