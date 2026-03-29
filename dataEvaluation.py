import argopy
from argopy import IndexFetcher
import matplotlib.pyplot as plt

# Florida Keys  Zone
box = [-83.5, -79.5, 23.5, 26.0, '2010-01', '2023-12']

print(f"Region: {box[2]}°N-{box[3]}°N, {abs(box[0])}°W-{abs(box[1])}°W")

print(f"Period:  {box[4]}m to {box[5]}m")

idx = IndexFetcher().region(box)
df = idx.to_dataframe()

total_profiles = len(df)
print(f"\nTotal profiles found: {total_profiles}")

df['year'] = df['date'].dt.year
yearly = df.groupby('year').size()
print("\nProfiles per year:")
print(yearly.to_string())

unique_floats = df['wmo'].nunique()
print(f"\nUnique floats: {unique_floats}")
idx.plot('trajectory')
plt.savefig('argo_map.png')
print("\n" + "=" * 50)
if total_profiles >= 500:
    print(f"SUFFICIENT DATA: {total_profiles} profiles — good to proceed")
elif total_profiles >= 300:
    print(f"MARGINAL DATA: {total_profiles} profiles — consider widening region slightly")
else:
    print(f"INSUFFICIENT DATA: {total_profiles} profiles — widen region or reconsider")

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

lon_min, lon_max, lat_min, lat_max = -83.5, -79.5, 23.5, 26.0
years = sorted(df['year'].unique())
n_years = len(years)

cols = 4
rows_for_years = int(np.ceil(n_years / cols))
total_rows = 1 + rows_for_years

fig = plt.figure(figsize=(18, 4 + rows_for_years * 4))
gs = fig.add_gridspec(total_rows, cols, hspace=0.35, wspace=0.3)

ax_main = fig.add_subplot(gs[0, :], projection=ccrs.PlateCarree())
ax_main.add_feature(cfeature.LAND, facecolor='lightgray')
ax_main.add_feature(cfeature.COASTLINE)
ax_main.add_feature(cfeature.BORDERS, linestyle=':')

scatter = ax_main.scatter(df['longitude'], df['latitude'], 
                         c=df['date'].dt.year, cmap='viridis', 
                         s=15, alpha=0.7, transform=ccrs.PlateCarree())

ax_main.plot([lon_min, lon_max, lon_max, lon_min, lon_min],
            [lat_min, lat_min, lat_max, lat_max, lat_min],
            color='red', linestyle='--', transform=ccrs.PlateCarree(), linewidth=2, label='Study Area')

ax_main.set_extent([-85, -78, 22, 28])
ax_main.set_title(f"All Years Combined - {len(df)} Total Profiles (2010-2023)", fontsize=14, fontweight='bold')
ax_main.legend(loc='upper right')
cbar = plt.colorbar(scatter, ax=ax_main, label='Year', shrink=0.8)

for idx, year in enumerate(years):
    row = 1 + idx // cols
    col = idx % cols
    
    ax_year = fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree())
    
    df_year = df[df['year'] == year]
    year_count = len(df_year)
    
    ax_year.add_feature(cfeature.LAND, facecolor='lightgray')
    ax_year.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax_year.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    
    ax_year.scatter(df_year['longitude'], df_year['latitude'], 
                   c='steelblue', s=20, alpha=0.7, 
                   transform=ccrs.PlateCarree(), edgecolors='darkblue', linewidth=0.3)
    
    ax_year.plot([lon_min, lon_max, lon_max, lon_min, lon_min],
               [lat_min, lat_min, lat_max, lat_max, lat_min],
               color='red', linestyle='--', transform=ccrs.PlateCarree(), linewidth=1)
    
    ax_year.set_extent([-85, -78, 22, 28])
    ax_year.set_title(f"{year}: {year_count} Profiles", fontsize=11, fontweight='bold')
    ax_year.set_xlabel('Longitude', fontsize=9)
    ax_year.set_ylabel('Latitude', fontsize=9)

for idx in range(n_years, rows_for_years * cols):
    row = 1 + idx // cols
    col = idx % cols
    ax_unused = fig.add_subplot(gs[row, col])
    ax_unused.axis('off')

plt.suptitle(f"Float Profile Trajectories - Argo Data (Florida Keys, 2010-2023)", 
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig('argo_trajectories_by_year.png', dpi=300, bbox_inches='tight')
print(f"\nVisualization saved to 'argo_trajectories_by_year.png'")
print(f"Total subplots: 1 (all years) + {n_years} (individual years)")
plt.show()