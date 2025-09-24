import os
import numpy as np
import xarray as xr
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rasterio
from rasterio.transform import from_origin
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import ndimage
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

file_path = r"C:\Users\anura\OneDrive\Desktop\S5P_NRTI_L2__NO2____20250902T080829_20250902T081329_40873_03_020800_20250902T084905.nc"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

ds = xr.open_dataset(file_path, group='PRODUCT')

required = ['nitrogendioxide_tropospheric_column',
            'qa_value', 'latitude', 'longitude']
for var in required:
    if var not in ds:
        raise ValueError(f"Variable '{var}' not found in PRODUCT group")

no2 = ds['nitrogendioxide_tropospheric_column'].isel(time=0)
qa = ds['qa_value'].isel(time=0)
lat = ds['latitude'].isel(time=0).values
lon = ds['longitude'].isel(time=0).values

no2_masked = no2.where(qa > 0.75)
no2_vals = np.nan_to_num(no2_masked.values, nan=0.0)

X = np.column_stack((lat.flatten(), lon.flatten()))
y = no2_vals.flatten()
mask = y > 0
X, y = X[mask], y[mask]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(
    n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_val)

rmse = np.sqrt(mean_squared_error(y_val, y_pred))
r2 = r2_score(y_val, y_pred)
print(f"Validation RMSE: {rmse:.6f}, R²: {r2:.4f}")

joblib.dump(rf, "rf_no2_downscaler.pkl")

factor = 5
new_shape = (lat.shape[0]*factor, lat.shape[1]*factor)
lat_fine = resize(lat, new_shape, order=3, mode='reflect', anti_aliasing=True)
lon_fine = resize(lon, new_shape, order=3, mode='reflect', anti_aliasing=True)

XY_fine = np.column_stack((lat_fine.flatten(), lon_fine.flatten()))
y_fine = rf.predict(XY_fine)
no2_fine = y_fine.reshape(new_shape)
no2_fine_smooth = ndimage.gaussian_filter(no2_fine, sigma=1.0)
no2_fine_smooth = np.maximum(no2_fine_smooth, 0)

pixel_h = (lat.max() - lat.min()) / new_shape[0]
pixel_w = (lon.max() - lon.min()) / new_shape[1]
transform = from_origin(lon.min(), lat.max(), pixel_w, pixel_h)

with rasterio.open(
    "fine_NO2_map.tif", "w",
    driver="GTiff",
    height=new_shape[0],
    width=new_shape[1],
    count=1,
    dtype=no2_fine_smooth.dtype,
    crs="EPSG:4326",
    transform=transform
) as dst:
    dst.write(no2_fine_smooth, 1)

fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={
                         'projection': ccrs.PlateCarree()})
cmap = plt.get_cmap("YlGnBu_r")

# Cities to plot
cities = {
    "New Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Bengaluru": (12.9716, 77.5946),
    "Kolkata": (22.5726, 88.3639),
    "Chennai": (13.0827, 80.2707),
    "Hyderabad": (17.3850, 78.4867),
    "Ahmedabad": (23.0225, 72.5714),
    "Pune": (18.5204, 73.8567),
    "Jaipur": (26.9124, 75.7873)
}


def add_cities(ax):
    for name, (lat_c, lon_c) in cities.items():
        ax.plot(lon_c, lat_c, marker='o', color='red',
                markersize=5, transform=ccrs.PlateCarree())
        ax.text(lon_c + 0.2, lat_c + 0.2, name, fontsize=8,
                color='black', transform=ccrs.PlateCarree())


# Original map
ax0 = axes[0]
ax0.set_title("Original Sentinel-5P NO₂")
ax0.add_feature(cfeature.COASTLINE)
ax0.add_feature(cfeature.BORDERS, linestyle=':')
pcm0 = ax0.pcolormesh(lon, lat, no2_vals, cmap=cmap,
                      transform=ccrs.PlateCarree())
fig.colorbar(pcm0, ax=ax0, orientation='vertical', label='NO₂ (mol/m²)')
add_cities(ax0)

# Downscaled map
ax1 = axes[1]
ax1.set_title(f"Downscaled NO₂ (×{factor})")
ax1.add_feature(cfeature.COASTLINE)
ax1.add_feature(cfeature.BORDERS, linestyle=':')
pcm1 = ax1.pcolormesh(lon_fine, lat_fine, no2_fine_smooth,
                      cmap=cmap, transform=ccrs.PlateCarree())
fig.colorbar(pcm1, ax=ax1, orientation='vertical', label='NO₂ (mol/m²)')
add_cities(ax1)

# Difference map
no2_orig_resized = resize(no2_vals, new_shape, order=1,
                          mode='reflect', anti_aliasing=True)
diff = no2_fine_smooth - no2_orig_resized
ax2 = axes[2]
ax2.set_title("Difference (Downscaled - Original)")
ax2.add_feature(cfeature.COASTLINE)
ax2.add_feature(cfeature.BORDERS, linestyle=':')
pcm2 = ax2.pcolormesh(lon_fine, lat_fine, diff,
                      cmap='RdBu_r', transform=ccrs.PlateCarree())
fig.colorbar(pcm2, ax=ax2, orientation='vertical', label='Difference (mol/m²)')

plt.tight_layout()
plt.savefig("downscaled_NO2_with_cities.png", dpi=300)
plt.show()

# Histogram
plt.figure(figsize=(10, 6))
orig_values = no2_vals[no2_vals > 0].flatten()
down_values = no2_fine_smooth[no2_fine_smooth > 0].flatten()
plt.hist(orig_values, bins=50, alpha=0.6,
         label='Original', density=True, color='blue')
plt.hist(down_values, bins=50, alpha=0.6,
         label='Downscaled', density=True, color='green')
plt.xlabel('NO₂ (mol/m²)')
plt.ylabel('Density')
plt.title('Histogram: Original vs Downscaled NO₂')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("no2_histogram_comparison.png", dpi=300)
plt.show()

print("\nPROCESSING COMPLETE")
