import pandas as pd
import xml.etree.ElementTree as et
import os

# 确保这里的路径与你在 .sumocfg 文件中配置的输出路径一致
# 假设 SUMO 输出到 ./scenario/results/tripinfo.xml
TRIPINFO_FILE = "./scenario/results/sample.tripinfo.xml"


def get_average_travel_time():
    """
    读取 tripinfo.xml 文件，计算所有完成行程的车辆的平均行程时间。
    如果文件不存在或没有完成行程的车辆，返回一个惩罚值（如 9999.0）。
    """

    # 1. 检查文件是否存在
    if not os.path.exists(TRIPINFO_FILE):
        # 车辆可能尚未完成行程，返回一个大惩罚值，确保 AverageTravelTime 会变化
        print(f"Warning: Tripinfo file not found at {TRIPINFO_FILE}. Returning MAX PENALTY.")
        return 9999.0

    try:
        # 2. 解析 XML 文件 (修正了错误的文件名)
        xtree = et.parse(TRIPINFO_FILE)
    except et.ParseError:
        print(f"Error: Could not parse XML file {TRIPINFO_FILE}. Returning MAX PENALTY.")
        return 9999.0

    xroot = xtree.getroot()
    rows = []

    # 3. 提取所有 tripinfo 标签中的 duration 属性
    for node in xroot.findall('tripinfo'):  # 使用 findall('tripinfo') 代替直接遍历 xroot
        travel_time = node.attrib.get("duration")
        if travel_time is not None:
            rows.append({"travel_time": travel_time})

    # 4. 检查是否有有效数据
    if not rows:
        # 如果没有车辆完成行程 (rows为空)，返回一个大惩罚值
        # 995.117 不再是硬编码的返回值！
        print(f"Warning: No completed trips found in {TRIPINFO_FILE}. Returning MAX PENALTY.")
        return 9999.0

    # 5. 使用 Pandas 计算平均值
    columns = ["travel_time"]
    travel_time_df = pd.DataFrame(rows, columns=columns).astype("float64")

    # 返回平均行程时间
    return travel_time_df["travel_time"].mean()
