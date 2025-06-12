import json
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']  # 优先使用的中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 测站信息
stations = {
    "PPPC": 9.773,
    "SHAO": 31.100,
    "CHAN": 43.791,
    "IRKJ": 52.219,
    "WHIT": 60.751,
    "SCOR": 70.485
}

# 按纬度从低到高排序测站
sorted_stations = dict(sorted(stations.items(), key=lambda item: item[1]))


# 读取VTEC数据
def load_vtec_data(station, data_dir="./"):
    """
    读取VTEC JSON数据
    参数:
        station: 测站名称
        data_dir: 数据文件目录
    返回:
        vtec_data: 包含各卫星VTEC数据的字典
    """
    # 查找匹配的文件
    station_files = []
    for file in os.listdir(data_dir):
        # 检查文件名是否包含测站名称且为JSON文件
        if station.lower() in file.lower() and file.endswith(".vtec.json"):
            station_files.append(os.path.join(data_dir, file))

    if not station_files:
        print(f"警告: 未找到测站 {station} 的VTEC数据文件")
        # 如果找不到文件，返回空字典或使用模拟数据
        return {}

    # 使用找到的第一个文件
    file_path = station_files[0]
    print(f"读取文件: {file_path}")

    try:
        with open(file_path, "r") as file:
            vtec_data = json.load(file)
        return vtec_data
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        return {}


# 分析VTEC数据统计特征
def analyze_vtec_statistics(vtec_data):
    """分析VTEC数据的统计特征"""
    all_values = []
    for prn, values in vtec_data.items():
        # 过滤掉0值
        valid_values = [v for v in values if abs(v) > 1.0e-6]
        if valid_values:
            all_values.extend(valid_values)

    if not all_values:
        return {
            "mean": 0,
            "max": 0,
            "min": 0,
            "std": 0,
            "range": 0
        }

    return {
        "mean": np.mean(all_values),
        "max": np.max(all_values),
        "min": np.min(all_values),
        "std": np.std(all_values),
        "range": np.max(all_values) - np.min(all_values)
    }


# 计算每个时间点的平均VTEC
def calculate_hourly_vtec(vtec_data):
    # 获取第一颗卫星的数据长度
    prns = list(vtec_data.keys())
    if not prns:
        return []

    epoch_count = len(vtec_data[prns[0]])

    hourly_vtec = []

    for epoch in range(epoch_count):
        values = []
        for prn, data in vtec_data.items():
            # 检查数据有效性并过滤零值和异常值
            if epoch < len(data) and abs(data[epoch]) > 1.0e-6 and data[epoch] < 200:
                values.append(data[epoch])

        # 只有当有足够有效值时才计算平均值
        if len(values) >= 3:  # 至少需要3颗卫星的有效数据
            hourly_vtec.append(np.mean(values))
        else:
            # 使用插值而非0值
            if hourly_vtec:
                hourly_vtec.append(hourly_vtec[-1])  # 使用前一个有效值
            else:
                hourly_vtec.append(np.nan)  # 标记为缺失值

    # 对NaN值进行插值处理
    hourly_vtec = np.array(hourly_vtec)
    mask = np.isnan(hourly_vtec)
    hourly_vtec[mask] = np.interp(
        np.flatnonzero(mask),
        np.flatnonzero(~mask),
        hourly_vtec[~mask]
    )

    return hourly_vtec.tolist()


# 分析所有测站
def analyze_all_stations():
    """分析所有测站的VTEC数据"""
    station_stats = {}
    hourly_data = {}

    for station in sorted_stations:
        vtec_data = load_vtec_data(station)
        stats = analyze_vtec_statistics(vtec_data)
        hourly_vtec = calculate_hourly_vtec(vtec_data)

        station_stats[station] = stats
        hourly_data[station] = hourly_vtec

    return station_stats, hourly_data


# 绘制结果
def plot_results(station_stats, hourly_data):
    """绘制分析结果"""
    # 1. 不同测站VTEC统计值
    stations = list(station_stats.keys())
    latitudes = [sorted_stations[s] for s in stations]
    means = [station_stats[s]["mean"] for s in stations]
    maxes = [station_stats[s]["max"] for s in stations]
    mins = [station_stats[s]["min"] for s in stations]
    ranges = [station_stats[s]["range"] for s in stations]

    # 创建图形
    plt.figure(figsize=(10, 8))

    # VTEC随纬度变化
    plt.plot(latitudes, means, 'o-', label='平均VTEC')
    plt.plot(latitudes, maxes, '^-', label='最大VTEC')
    plt.plot(latitudes, mins, 'v-', label='最小VTEC')
    plt.plot(latitudes, ranges, 's-', label='VTEC变化范围')

    plt.xlabel('纬度 (°N)')
    plt.ylabel('VTEC (TECU)')
    plt.title('不同纬度测站的VTEC统计特征')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('./result/vtec_latitude_analysis.png', dpi=300)
    plt.close()

    # 3. 生成统计表格
    stats_df = pd.DataFrame({
        '测站': stations,
        '纬度(°N)': latitudes,
        '平均VTEC(TECU)': means,
        '最大VTEC(TECU)': maxes,
        '最小VTEC(TECU)': mins,
        'VTEC变化幅度(TECU)': ranges
    })

    print(stats_df)
    try:
        os.makedirs('./result', exist_ok=True)
        stats_df.to_csv('./result/vtec_statistics.csv', index=False, encoding='utf-8-sig')
    except Exception as e:
        print(f"保存UTF-8-sig格式失败: {e}")
    return stats_df


# 主分析函数
def main_analysis(data_dir="./"):
    """
    主分析函数
    参数:
        data_dir: 数据文件目录
    """
    # 检查数据文件
    print(f"检查目录 {data_dir} 中的数据文件...")
    all_files = os.listdir(data_dir)
    vtec_files = [f for f in all_files if f.endswith(".vtec.json")]
    print(f"找到 {len(vtec_files)} 个VTEC数据文件: {vtec_files}")

    # 分析所有测站
    station_stats = {}
    hourly_data = {}

    for station in sorted_stations:
        print(f"\n处理测站: {station}，纬度: {sorted_stations[station]}°N")
        vtec_data = load_vtec_data(station, data_dir)

        if not vtec_data:
            print(f"未找到测站 {station} 的有效数据，跳过此测站")
            continue

        print(f"成功读取 {station} 测站数据，包含 {len(vtec_data)} 颗卫星的VTEC记录")
        stats = analyze_vtec_statistics(vtec_data)
        hourly_vtec = calculate_hourly_vtec(vtec_data)

        station_stats[station] = stats
        hourly_data[station] = hourly_vtec

        print(f"测站 {station} 统计数据: 平均VTEC={stats['mean']:.2f} TECU, 最大值={stats['max']:.2f} TECU")

    if not station_stats:
        print("未找到任何测站的有效数据，无法进行分析")
        return

    stats_df = plot_results(station_stats, hourly_data)

    # 分析结果
    print("\n不同纬度测站电离层VTEC差异分析结果:")
    print("-" * 60)

    # 找出最大VTEC的测站和纬度
    max_vtec_station = stats_df.loc[stats_df['最大VTEC(TECU)'].idxmax()]
    print(f"最大VTEC值出现在 {max_vtec_station['测站']} 测站 (纬度: {max_vtec_station['纬度(°N)']}°N)")

    # 低纬度和高纬度的差异
    low_lat_df = stats_df[stats_df['纬度(°N)'] < 30]
    high_lat_df = stats_df[stats_df['纬度(°N)'] > 55]

    if not low_lat_df.empty and not high_lat_df.empty:
        low_lat_mean = low_lat_df['平均VTEC(TECU)'].mean()
        high_lat_mean = high_lat_df['平均VTEC(TECU)'].mean()
        diff_percent = (low_lat_mean - high_lat_mean) / high_lat_mean * 100 if high_lat_mean > 0 else 0
        print(f"低纬度地区平均VTEC比高纬度地区高约 {diff_percent:.1f}%")

    # 电离层赤道异常特征
    eq_anomaly_stations = stats_df[(stats_df['纬度(°N)'] >= 10) & (stats_df['纬度(°N)'] <= 20)]
    if not eq_anomaly_stations.empty:
        print(f"观察到电离层赤道异常现象，赤道附近±15°纬度区域VTEC值较高")

    # 尝试计算纬度相关性
    try:
        from scipy.stats import pearsonr
        if len(stats_df) >= 3:  # 至少需要3个点才能计算有意义的相关性
            corr, p_value = pearsonr(stats_df['纬度(°N)'], stats_df['平均VTEC(TECU)'])
            print(f"VTEC与纬度的相关系数为：{corr:.6f}")
            if p_value < 0.05:
                print(f"   相关性显著：(p-value={p_value:.6f})")
            else:
                print(f"   相关性不显著：(p-value={p_value:.6f})")
        else:
            print("测站数量不足，无法计算VTEC与纬度的相关性")
    except Exception as e:
        print(f"计算相关性时出错: {e}")

    return stats_df


# 执行分析
if __name__ == "__main__":
    data_dir = "vtecs"
    main_analysis(data_dir)