import os
import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from scipy.spatial import ConvexHull, Voronoi, voronoi_plot_2d
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import ElasticNet
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 基本参数
RNX_DIR  = Path(r'hkdata')   # RINEX数据目录
VTEC_DIR = Path(r'HKVTEC')  # VTEC数据目录

DEGREE     = 5        # 多项式阶次
ALPHA      = 0.0009  # ElasticNet正则化强度
L1_RATIO   = 0.5    # ElasticNet的L1比例
SAMPLE_SEC = 8*3600  # 选取 8:00
INTVL_SEC  = 30     # 观测采样间隔（秒）

def ecef2geodetic(x: float, y: float, z: float):
    """WGS-84 椭球下 ECEF ➜ (lat, lon)  (度)"""
    a  = 6378137.0            # 长半轴
    e2 = 6.69437999014e-3     # 第一偏心率平方
    b  = a * math.sqrt(1 - e2)

    lon = math.atan2(y, x)
    p   = math.hypot(x, y)
    theta = math.atan2(z * a, p * b)
    lat = math.atan2(z + e2 * b * math.sin(theta)**3,
                     p - e2 * a * math.cos(theta)**3)
    return math.degrees(lat), math.degrees(lon)

def read_rinex_positions(folder: Path):
    """返回 {站名: (lon, lat)}"""
    pos = {}
    for file in folder.glob('*.rnx'):
        sta = file.stem[:4].lower()
        with file.open('r', errors='ignore') as fh:
            for line in fh:
                if 'APPROX POSITION XYZ' in line:
                    x, y, z = map(float, line.split()[:3])
                    lat, lon = ecef2geodetic(x, y, z)
                    pos[sta] = (lon, lat)
                    break
    return pos

def read_vtec_at_epoch(folder: Path, epoch_idx: int):
    """返回 {站名: vtec_value}"""
    vtec = {}
    for file in folder.glob('*.json'):
        sta = file.stem[:4].lower()
        with file.open('r', encoding='utf-8') as fh:
            data = json.load(fh)
        vals = []
        for prn, series in data.items():
            if epoch_idx < len(series):
                val = series[epoch_idx]
                if val is not None and not (isinstance(val, float) and math.isnan(val)):
                    vals.append(val)
        if vals:
            vtec[sta] = float(np.mean(vals))
    return vtec

def normalize(arr):
    """线性映射到 [-1, 1]"""
    mn, mx = float(np.min(arr)), float(np.max(arr))
    return 2*(arr - mn)/(mx - mn) - 1, mn, mx

# 读取数据
stations_ll = read_rinex_positions(RNX_DIR)
epoch_idx   = SAMPLE_SEC // INTVL_SEC
stations_vt = read_vtec_at_epoch(VTEC_DIR, epoch_idx)

common_keys = sorted(set(stations_ll) & set(stations_vt))
if len(common_keys) < 4:
    raise RuntimeError('有效测站不足，无法拟合多项式')

lon_arr = np.array([stations_ll[k][0] for k in common_keys])
lat_arr = np.array([stations_ll[k][1] for k in common_keys])
vtec_arr = np.array([stations_vt[k]    for k in common_keys])

# 多项式拟合
lon_n, lon_min, lon_max = normalize(lon_arr)
lat_n, lat_min, lat_max = normalize(lat_arr)
X_norm = np.column_stack((lon_n, lat_n))

poly = PolynomialFeatures(degree=DEGREE, include_bias=True)
X_poly = poly.fit_transform(X_norm)

model = ElasticNet(alpha=ALPHA, l1_ratio=L1_RATIO, fit_intercept=False, tol=1e-8,max_iter=10000)
model.fit(X_poly, vtec_arr)
r2 = model.score(X_poly, vtec_arr)

print(f'>>> 已拟合 {DEGREE} 阶多项式，R² = {r2:.3f}，样本数 = {len(common_keys)}')

# 生成网格
lon_lin = np.linspace(lon_min, lon_max, 200)
lat_lin = np.linspace(lat_min, lat_max, 200)
grid_lon, grid_lat = np.meshgrid(lon_lin, lat_lin)
grid_pts = np.column_stack((grid_lon.ravel(), grid_lat.ravel()))

# 利用测站凸包裁剪，避免外插
hull = ConvexHull(np.column_stack((lon_arr, lat_arr)))
hull_path = MplPath(np.column_stack((lon_arr, lat_arr))[hull.vertices])
inside = hull_path.contains_points(grid_pts)

# 归一化 + 预测
lon_n_g = 2*(grid_pts[:, 0] - lon_min)/(lon_max - lon_min) - 1
lat_n_g = 2*(grid_pts[:, 1] - lat_min)/(lat_max - lat_min) - 1
Z_pred  = model.predict(poly.transform(np.column_stack((lon_n_g, lat_n_g))))

Z_grid = np.full(grid_lon.shape, np.nan)
Z_grid.ravel()[inside] = Z_pred[inside]

# 计算残差
station_pred = model.predict(X_poly)
residuals = vtec_arr - station_pred
abs_residuals = np.abs(residuals)

# 图1: 残差热力图
plt.figure(figsize=(8, 6))
plt.title('残差空间分布热力图', fontsize=15)

grid_x, grid_y = np.mgrid[lon_min:lon_max:100j, lat_min:lat_max:100j]
grid_residuals = griddata((lon_arr, lat_arr), residuals, (grid_x, grid_y), method='cubic')

hull_mask = np.zeros_like(grid_residuals, dtype=bool)
for i in range(grid_x.shape[0]):
    for j in range(grid_x.shape[1]):
        if not hull_path.contains_point((grid_x[i, j], grid_y[i, j])):
            hull_mask[i, j] = True
grid_residuals = np.ma.array(grid_residuals, mask=hull_mask)

cmap = plt.cm.coolwarm
im = plt.pcolormesh(grid_x, grid_y, grid_residuals, cmap=cmap, shading='auto',
                    vmin=-np.max(abs_residuals), vmax=np.max(abs_residuals))
plt.colorbar(im, label='残差 (TECU)')

plt.scatter(lon_arr, lat_arr, c='k', s=30, marker='^', label='测站位置')
for i, txt in enumerate(common_keys):
    plt.annotate(f'{txt}\n({residuals[i]:.2f})', (lon_arr[i], lat_arr[i]),
                 fontsize=9, ha='center', va='bottom', color='k')

plt.xlabel('经度 (°E)')
plt.ylabel('纬度 (°N)')
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(loc='best')

rmse = np.sqrt(np.mean(residuals**2))
plt.figtext(0.5, 0.01,
            f"均方根误差(RMSE): {rmse:.3f} TECU | "
            f"最大残差: {np.max(abs_residuals):.3f} TECU | "
            f"平均绝对残差: {np.mean(abs_residuals):.3f} TECU",
            ha='center', fontsize=11, bbox=dict(boxstyle="round,pad=0.5",
                                              fc="white", ec="gray", alpha=0.8))

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig('./result/残差热力图.png', dpi=300, bbox_inches='tight')
plt.show()

# 图2: 3D表面
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

vtec_cmap = plt.cm.plasma
surf = ax.plot_surface(grid_lon, grid_lat, Z_grid,
                       rstride=4, cstride=4,
                       cmap=vtec_cmap, edgecolor='none', alpha=0.90)
ax.scatter(lon_arr, lat_arr, vtec_arr, c='white', s=35, edgecolor='black', label='测站')

ax.set_xlabel('经度 (°E)')
ax.set_ylabel('纬度 (°N)')
ax.set_zlabel('VTEC (TECU)')
ax.set_title(f'香港 VTEC 3D表面，使用 {DEGREE} 阶多项式拟合\n'
             f't = {SAMPLE_SEC//3600:02d}:{SAMPLE_SEC%3600//60:02d}')
fig.colorbar(surf, shrink=0.55, aspect=12, label='VTEC (TECU)')
plt.legend()
plt.tight_layout()
plt.savefig('./result/VTEC_3D表面.png', dpi=300, bbox_inches='tight')
plt.show()

# 图3: 2D空间分布图
plt.figure(figsize=(8, 6))
plt.title('香港地区 VTEC 2D空间分布图', fontsize=15)

contour_filled = plt.contourf(grid_lon, grid_lat, Z_grid, levels=15, cmap='jet', alpha=0.8)
contour_lines = plt.contour(grid_lon, grid_lat, Z_grid, levels=8, colors='k', linewidths=0.5, alpha=0.7)
plt.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f')

plt.scatter(lon_arr, lat_arr, c='white', s=40, edgecolor='black', marker='^', label='测站位置')
for i, txt in enumerate(common_keys):
    plt.annotate(txt, (lon_arr[i], lat_arr[i]), fontsize=9, ha='center', va='bottom')

cbar = plt.colorbar(contour_filled, label='VTEC (TECU)')
cbar.set_label('VTEC (TECU)', fontsize=12)

plt.xlabel('经度 (°E)', fontsize=12)
plt.ylabel('纬度 (°N)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(loc='best')
plt.axis('equal')
plt.tight_layout()
plt.savefig('./result/VTEC_2D空间分布图.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n图表生成完成！共生成3张图：")
print("1. 残差热力图")
print("2. VTEC_3D表面图")
print("3. VTEC_2D空间分布图")