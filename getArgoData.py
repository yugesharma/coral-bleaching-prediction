import argopy
from argopy import DataFetcher
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def downloadArgoData(box):
    fetcher = DataFetcher(src='erddap').region(box)
    ds = fetcher.to_xarray()

    # Save raw download locally — do this once, don't re-download every run
    ds.to_netcdf('argo_florida_keys_raw_2000.nc')
    print('Argo data downloaded')

def checkData(ds):
    df = ds.to_dataframe().reset_index()
    print(len(df))
    # 1. How many unique profiles do you actually have
    n_profiles = df.groupby(['PLATFORM_NUMBER', 'CYCLE_NUMBER']).ngroups
    print(f"Unique profiles: {n_profiles}")

    # 2. Depth coverage — what's the max pressure per profile
    max_depth = df.groupby(['PLATFORM_NUMBER', 'CYCLE_NUMBER'])['PRES'].max()
    print(max_depth.describe())
    # You want most profiles reaching at least 200m
    print(f"Profiles reaching 200m+: {(max_depth >= 200).sum()}")
    # df['month'] = df['TIME'].dt.month
    # df['year'] = df['TIME'].dt.year

    # print(df.groupby(['year', 'month']).size().unstack())

    # profile_depths = df.groupby(['PLATFORM_NUMBER', 'CYCLE_NUMBER'])['PRES'].count()

    # print("Depth levels per profile:")
    # print(profile_depths.describe())
    # print(f"\nMedian depth levels: {profile_depths.median():.0f}")
    # print(f"Mode depth levels: {profile_depths.mode()[0]:.0f}")

    # # Distribution to see if it's consistent
    # print("\nDistribution:")
    # print(profile_depths.value_counts().sort_index())

    plt.figure(figsize=(10, 6))

    # Use a KDE (Kernel Density Estimate) to see the smooth curve
    sns.histplot(df['PSAL'], kde=True, color='teal', bins=100)

    plt.title('Distribution of Practical Salinity (PSAL)')
    plt.xlabel('Salinity (psu)')
    plt.ylabel('Frequency')

    # Optional: Zoom in on the realistic range to ignore extreme "junk" outliers
    # plt.xlim(30, 38) 

    plt.grid(axis='y', alpha=0.3)
    plt.savefig('PSAL_distribution.png')
    plt.clf()
    sns.histplot(df['TEMP'], kde=True, color='teal', bins=100)

    plt.title('Distribution of temperature (TEMP)')
    plt.xlabel('Temp (C)')
    plt.ylabel('Frequency')

    # Optional: Zoom in on the realistic range to ignore extreme "junk" outliers
    # plt.xlim(30, 38) 

    plt.grid(axis='y', alpha=0.3)
    plt.savefig('temp_distribution.png')

    temp_stats = df['TEMP'].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99])
    print(temp_stats)

def prepareData(ds):
    df = ds.to_dataframe().reset_index()

    # Limit to upper 300m
    df = df[df['PRES'] <= 300]

    # Drop profiles with too few depth levels
    MIN_DEPTHS = 25
    profile_counts = df.groupby(['PLATFORM_NUMBER', 'CYCLE_NUMBER'])['PRES'].count()
    valid_idx = profile_counts[profile_counts >= MIN_DEPTHS].index
    df = df.set_index(['PLATFORM_NUMBER', 'CYCLE_NUMBER'])
    df = df[df.index.isin(valid_idx)].reset_index()

    # Drop profiles not reaching 200m
    profile_max_depth = df.groupby(['PLATFORM_NUMBER', 'CYCLE_NUMBER'])['PRES'].max()
    deep_idx = profile_max_depth[profile_max_depth >= 200].index
    df = df.set_index(['PLATFORM_NUMBER', 'CYCLE_NUMBER'])
    df = df[df.index.isin(deep_idx)].reset_index()

    # Drop physical outliers
    df = df[(df['TEMP'] > -2) & (df['TEMP'] < 35)]
    df = df[(df['PSAL'] > 30) & (df['PSAL'] < 38)]

    print(f"Profiles after filtering: {df.groupby(['PLATFORM_NUMBER', 'CYCLE_NUMBER']).ngroups}")
    
    return df

if __name__ == "__main__":
    box = [-83.5, -79.5, 23.5, 26.0, 0, 300, '2010-01', '2023-12']
    # downloadArgoData(box)
    filePath='argo_florida_keys_raw.nc'
    ds=xr.open_dataset(filePath)
    # print(ds)
    # checkData(ds)
    df=prepareData(ds)