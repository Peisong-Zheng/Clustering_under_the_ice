# the pipeline to prepare the training data
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
import pandas as pd
import numpy as np
import plotly.graph_objs as go

def prepare_training_data(
    lat_step: float,
    lon_step: float,
    quantile4T: float,
    ds_cleaned: xr.Dataset,
    ratio_monthly_sampled: float = 1
) -> xr.Dataset:
    """
    Prepare the training data by spatially and temporally sample the cleaned data.

    Args:
        lat_step (float): The latitude step size.
        lon_step (float): The longitude step size.
        quantile4T (float): The quantile value for threshold calculation.
        ds_cleaned (xr.Dataset): The cleaned dataset.
        ratio_monthly_sampled (float, optional): The ratio for monthly sampling. Defaults to 1.

    Returns:
        xr.Dataset: The prepared training data.
    """
    threshold = int(cal_threshold(ds_cleaned, lat_step, lon_step, quantile=quantile4T))
    ds_cleaned_spatially_sampled = drop_profiles(ds_cleaned, lat_step, lon_step, threshold)
    ds_cleaned_monthly_sampled = ramdon_sample_by_month(ds_cleaned_spatially_sampled, ratio_monthly_sampled, random_seed=0)
    ds_cleaned_monthly_sampled['lat_step'] = lat_step
    ds_cleaned_monthly_sampled['lon_step'] = lon_step
    ds_cleaned_monthly_sampled['quantile4T'] = quantile4T
    print(f'threshold={threshold}, training dataset size={len(ds_cleaned_monthly_sampled.nprof)}')
    return ds_cleaned_monthly_sampled



#############################################################################################################

def cal_threshold(
    ds: xr.Dataset,
    lat_step: float,
    lon_step: float,
    quantile: float
) -> int:
    """
    Calculate the threshold based on the difference between adjacent elements in the sorted latitude-longitude bins.

    Args:
        ds (xr.Dataset): The dataset.
        lat_step (float): The latitude step size for binning.
        lon_step (float): The longitude step size for binning.
        quantile (float): The quantile value for threshold calculation.

    Returns:
        int: The calculated threshold.
    """
    df_latlon = ds[['lat', 'lon']].to_dataframe()

    lon_bins = pd.cut(df_latlon['lon'], bins=np.arange(-180, 185, lon_step))
    lat_bins = pd.cut(df_latlon['lat'], bins=np.arange(-90, 95, lat_step))

    latlon_groups = df_latlon.groupby([lon_bins, lat_bins])

    g_len = []
    for group_key in latlon_groups.groups:
        try:
            group = latlon_groups.get_group(group_key)
            # print(f"Group {group_key}: {len(group)} rows")
            g_len.append(len(group))
        except KeyError:
            # Handle the KeyError gracefully
            print(f"No data points for group {group_key}")

    g_len_sorted = sorted(g_len)
    transition_point = None

    g_len_diff = np.diff(g_len_sorted)

    threshold = np.quantile(g_len_diff, quantile)  # set a threshold for the difference between adjacent elements

    for i in range(1, len(g_len_sorted)):
        if g_len_sorted[i] - g_len_sorted[i-1] > threshold:
            transition_point = i - 1
            break

    return g_len_sorted[transition_point]


####################################################################################
import numpy as np
import xarray as xr
import pandas as pd

def drop_profiles(
    ds: xr.Dataset,
    lat_step: float,
    lon_step: float,
    threshold: int
) -> xr.Dataset:
    """
    Drop the profiles in the bins with sample size larger than the threshold.

    Args:
        ds (xr.Dataset): The dataset.
        lat_step (float): The latitude step size for binning.
        lon_step (float): The longitude step size for binning.
        threshold (int): The threshold for dropping profiles.

    Returns:
        xr.Dataset: The dataset with profiles dropped.
    """
    df_latlon = ds[['lat', 'lon']].to_dataframe()

    lon_bins = pd.cut(df_latlon['lon'], bins=np.arange(-180, 185, lon_step))
    lat_bins = pd.cut(df_latlon['lat'], bins=np.arange(-90, 95, lat_step))

    latlon_groups = df_latlon.groupby([lon_bins, lat_bins])

    latlon_groups_list = []

    for group_key in latlon_groups.groups:
        try:
            group = latlon_groups.get_group(group_key)
            if len(group) > threshold:
                group_drop = group.sample(n=len(group)-threshold, random_state=0)
                latlon_groups_list.append(group_drop)
        except KeyError:
            print(f"No data points for group {group_key}")

    ds_spatially_sampled = ds.copy()

    nprof_dropped = []

    for group in latlon_groups_list:
        temp_index = group.index.values
        nprof_dropped.append(temp_index.tolist())

    nprof_dropped = [item for sublist in nprof_dropped for item in sublist]

    ds_spatially_sampled = ds_spatially_sampled.drop_sel(nprof=nprof_dropped)

    return ds_spatially_sampled


########################################################################################################################


def ramdon_sample_by_month(
    ds: xr.Dataset,
    ratio: float,
    random_seed: int = 0
) -> xr.Dataset:
    """
    Sample profiles by month based on a given ratio.

    Args:
        ds (xr.Dataset): The dataset.
        ratio (float): The ratio of profiles to sample (between 0 and 1).
        random_seed (int, optional): The random seed for reproducibility. Defaults to 0.

    Returns:
        xr.Dataset: The dataset with sampled profiles.
    """
    if ratio < 0 or ratio > 1:
        raise ValueError("Ratio must be between 0 and 1.")

    if 'date' not in ds:
        raise ValueError("Input dataset must have a 'date' variable.")

    if 'nprof' not in ds.dims:
        raise ValueError("Input dataset must have a 'nprof' dimension.")

    ds_monthly = ds.groupby('date.month')
    months_with_data = [month for month in ds_monthly.groups.keys() if len(ds_monthly[month]) > 0]

    if not months_with_data:
        raise ValueError("Input dataset does not contain any data.")

    smallest_month_size = np.min([ds_monthly[month].nprof.size for month in months_with_data])
    selected_size_monthly = ds.nprof.size * ratio / 12

    if smallest_month_size < selected_size_monthly:
        selected_size_monthly = smallest_month_size
        print(f"Ratio is too large, selected_size_monthly is reset to the smallest_month_size: {smallest_month_size}")

    ds_selected = xr.concat([ds_monthly[month].isel(nprof=np.random.default_rng(random_seed).choice(ds_monthly[month].nprof.size, int(selected_size_monthly), replace=False)) for month in months_with_data], dim='nprof')

    ds_selected['nprof'] = np.arange(ds_selected.sizes['nprof'])

    return ds_selected



# ########################################################################################################################
# import xarray as xr

# def add_info(
#     ds_train: xr.Dataset,
#     ds_fit: xr.Dataset
# ) -> xr.Dataset:
#     """
#     Add additional information from the training dataset to the fit dataset.

#     Args:
#         ds_train (xr.Dataset): The training dataset.
#         ds_fit (xr.Dataset): The fit dataset.

#     Returns:
#         xr.Dataset: The fit dataset with additional information added.
#     """
#     ds_fit['lat_step'] = ds_train['lat_step']
#     ds_fit['lon_step'] = ds_train['lon_step']
#     ds_fit['quantile4T'] = ds_train['quantile4T']
#     ds_fit['nprof_of_train'] = ds_train.nprof.size
    
#     return ds_fit

