import matplotlib.pyplot as plt
import numpy as np

# ===============================
# 1. 定义原始数据
# ===============================
# points = {
#     "1": (129, 168),
#     "2": (233.5, 96.5),
#     "3": (112.5, 105.5),
#     "4": (70.5, 117.5),
#     "5": (50.5, 214.5),
#     "6": (144.5, 251.5),
# }

points = {
     "1": (152, 157),
      "2": (146.5, 108.5),
       "3": (88.5, 131.5),
    "4": (185.5, 198.5),  
}

# ===============================
# 2. 设置中心点（胞体中心点1）
# ===============================
center = np.array(points["1"])

# ===============================
# 3. 计算每个点的角度和半径
# ===============================
angles, radii, labels = [], [], []

for name, (x, y) in points.items():
    if name == "胞体中心点1":
        continue
    vec = np.array([x, y]) - center
    r = np.linalg.norm(vec)                   # 半径（距离）
    print(r)
    theta = np.arctan2(vec[1], vec[0])        # 弧度角
    angles.append(theta)
    radii.append(r)
    labels.append(name)

angles = np.array(angles)
radii = np.array(radii)

# ===============================
# 4. 绘制极坐标柱状图（分成12份）
# ===============================
fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(111, polar=True)

rotate_deg = 0
ax.set_theta_offset(np.deg2rad(rotate_deg))
ax.set_theta_direction(-1)   # 1 表示逆时针方向，-1 表示顺时针

# 分区数
num_bins = 12

# 计算每个点所属的扇区
bin_edges = np.linspace(-np.pi, np.pi, num_bins+1)   # 扇区边界
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2   # 扇区中心

# 初始化每个扇区的半径
sector_radii = np.zeros(num_bins)

# 将每个点分配到对应扇区
for theta, r in zip(angles, radii):
    bin_idx = np.digitize(theta, bin_edges) - 1
    if bin_idx == num_bins:
        bin_idx = 0
    sector_radii[bin_idx] = max(sector_radii[bin_idx], r)

# 绘制极坐标柱状图
ax.bar(
    bin_centers, sector_radii, width=2*np.pi/num_bins, 
    bottom=0.0, color="skyblue", edgecolor="black", linewidth=7, alpha=1, align="center",zorder=5  
)

ax.set_ylim(0, 140)

# ===============================
# 隐藏半径刻度（不显示最外层数字）
# ===============================
# ax.set_yticks([])         # 移除所有半径刻度
ax.set_yticklabels([])    # 移除对应的标签

ax.set_thetagrids(range(0, 360, 30), labels=[""] * 12)
# ax.tick_params(axis="x", labelsize=12, width=2)  # x = θ轴
# ax.tick_params(axis="y", labelsize=12, width=2)  # y = r轴
# ax.set_xticks([])  # 去掉角度刻度线和数字
# 🔧 关键 3：角度刻度线加粗
ax.xaxis.grid(True, linewidth=3)
# 🔧 关键 2：半径刻度线加粗
ax.yaxis.grid(True, linewidth=3)
# 🔧 关键 4：最外圈圆（polar spine）加粗
ax.spines["polar"].set_linewidth(3)


# ===============================
# 5. 标注每个点的名称（仍保留）
# ===============================
# for theta, r, label in zip(angles, radii, labels):
#     ax.text(theta, r+10, label, fontsize=10, ha="center", va="center")

# 设置角度刻度为 0, 30, 60 ... 330
ax.set_thetagrids(range(0, 360, 30))

# 获取半径刻度线对象
rgridlines = ax.yaxis.get_gridlines()

# 设置线条粗细和颜色
for line in rgridlines:
    line.set_linewidth(3)   # 加粗
    line.set_color("0")   # 可选：颜色

# ===============================
# 6. 标题 & 显示
# ===============================
ax.set_title("", va="bottom")
plt.savefig("/Users/longzhicheng/Downloads/2.png", dpi=300)  # 输出 2100x2100 px PNG
