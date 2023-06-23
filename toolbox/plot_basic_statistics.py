import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr

def plot_month_latlon_dis(ds: xr.Dataset) -> None:
    """
    Plot three histograms: month distribution, longitude distribution, and latitude distribution.

    Args:
        ds (xr.Dataset): The dataset to plot.

    Returns:
        None
    """
    months = ds.date.dt.month.values

    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
    fig, ax = plt.subplots(1, 3, figsize=(20, 5), dpi=300)
    sns.histplot(months, ax=ax[0])
    sns.histplot(ds.lon, ax=ax[1])
    sns.histplot(ds.lat, ax=ax[2])
    # set the x and y label
    ax[0].set_xlabel('Month')
    ax[0].set_ylabel('Count')
    ax[1].set_xlabel('Longitude')
    ax[1].set_ylabel('Count')
    ax[2].set_xlabel('Latitude')
    ax[2].set_ylabel('Count')
    # set the title
    ax[0].set_title('Month distribution')
    ax[1].set_title('Longitude distribution')
    ax[2].set_title('Latitude distribution')
    plt.show()


###############################################################################################################

import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
import pandas as pd
import numpy as np
import plotly.graph_objs as go

def plot_month_distribution(ds: xr.Dataset) -> None:
    """
    Plot the distribution of months in a dataset.

    Args:
        ds (xr.Dataset): The dataset to plot.

    Returns:
        None
    """
    months = ds.date.dt.month.values

    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300)
    sns.histplot(months, ax=ax)
    # set the x and y label
    ax.set_xlabel('Month')
    ax.set_ylabel('Count')
    # set the title
    # ax.set_title('Month distribution')
    plt.show()

    ###############################################################################################################
    # function to plot the heatmap of the profiles distribution
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

def plot_heatmap(ds_cleaned, bin_max=None):
    '''
    ds_cleaned: A dataset containing cleaned data.
    bin_max (optional): The maximum bin value for the heatmap. If not provided, it is calculated as the maximum value of the histogram.
    '''

    # Define the map bounds and resolution
    lon_min, lon_max = -180, 180
    lat_min, lat_max = 70, 90
    latlon_step = 5
    resolution = 'l'  # l for low resolution, h for high resolution

    # Define bin edges for longitude, latitude, and pressure
    lon_bins = np.arange(lon_min, lon_max+1, latlon_step)
    lat_bins = np.arange(lat_min, lat_max+1, latlon_step)

    # Use the number of profiles in each bin as the height
    H, xedges, yedges = np.histogram2d(ds_cleaned.lon, ds_cleaned.lat, bins=[lon_bins, lat_bins])

    # Set up the figure and the Basemap instance
    fig = plt.figure(figsize=(8,8),dpi=300)
    m = Basemap(projection='npstere', boundinglat=lat_min, lon_0=0, resolution=resolution)

    m.drawparallels(np.arange(lat_min, lat_max+1, 5), labels=[], linewidth=0.1)
    m.drawmeridians(np.arange(lon_min, lon_max+1, 5), labels=[], linewidth=0.1)

    # Create the X and Y grids for the histogram
    X, Y = np.meshgrid(xedges[:-1], yedges[:-1])

    new_col = np.arange(lat_min, lat_max, latlon_step)
    Y = np.concatenate((Y, new_col[:, np.newaxis]), axis=1)

    new_col = lon_max*np.ones(len(lat_bins)-1)
    X = np.concatenate((X, new_col[:, np.newaxis]), axis=1)

    new_row = X[-1]
    X=np.vstack([X,new_row])

    new_row = lat_max*np.ones(np.shape(Y)[1])
    Y=np.vstack([Y,new_row])

    # Create the colored grid using pcolormesh
    cmap = plt.get_cmap('Wistia').copy()
    cmap.set_under(color='white', alpha=1.0)
    cmap.set_over(color='white', alpha=1.0)

    if bin_max is None:
        bin_max = H.max()
    

    m.pcolormesh(X, Y, H.T, cmap=cmap, vmin=1, vmax=bin_max, latlon=True)

    m.drawcoastlines()
    m.fillcontinents(color='grey',lake_color='lightblue')

    # Add a colorbar
    cbar = m.colorbar(location='bottom', pad="5%")
    cbar.set_label('Number of profiles')

    # Set the axis labels and title
    plt.xticks([])
    plt.yticks([])
    plt.title('Profiles distribution')

    # Show the plot
    plt.show()
    print(f'bin max={bin_max}')