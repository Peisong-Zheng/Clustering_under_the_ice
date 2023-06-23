import numpy as np
import xarray as xr

def relabel_pcm_labels(ds: xr.Dataset, label_mapping: dict = {0: 0, 1: 1, 2: 2, 3: 3}) -> xr.Dataset:
    """
    Relabels the PCM_LABELS and updates PCM_POST values in the input dataset.

    Parameters:
        - ds (xr.Dataset): Input dataset containing 'PCM_LABELS' and 'PCM_POST' variables.
        - label_mapping (dict): Mapping of original labels to new labels. Default is {0: 0, 1: 3, 2: 1, 3: 2}.

    Returns:
        - ds (xr.Dataset): Updated dataset with relabeled 'PCM_LABELS' and 'PCM_POST' variables.
    """
    ds=ds.copy()
    unique_labels = np.unique(ds.PCM_LABELS.values)
    num_classes = len(unique_labels)
    # if num_classes > 4:
    #     print('The number of unique labels is not 4, please check the input ds')
    #     return ds

    # Find the indices where the label is 0, 1, 2, and 3
    idx_0 = np.where(ds['PCM_LABELS'] == 0)
    idx_1 = np.where(ds['PCM_LABELS'] == 1)
    idx_2 = np.where(ds['PCM_LABELS'] == 2)
    idx_3 = np.where(ds['PCM_LABELS'] == 3)

    # Set the values in idx_0 to 1, idx_1 to 3, idx_2 to 1, and idx_3 to 2
    ds['PCM_LABELS'][idx_0] = label_mapping[0]
    ds['PCM_LABELS'][idx_1] = label_mapping[1]
    ds['PCM_LABELS'][idx_2] = label_mapping[2]
    ds['PCM_LABELS'][idx_3] = label_mapping[3]

    # Copy the PCM_POST values
    p0 = ds['PCM_POST'].values[0].copy()
    p1 = ds['PCM_POST'].values[1].copy()
    p2 = ds['PCM_POST'].values[2].copy()
    p3 = ds['PCM_POST'].values[3].copy()

    # Update the PCM_POST values
    ds['PCM_POST'].values[label_mapping[0]] = p0
    ds['PCM_POST'].values[label_mapping[2]] = p2
    ds['PCM_POST'].values[label_mapping[3]] = p3
    ds['PCM_POST'].values[label_mapping[1]] = p1

    return ds

##############################################################################################################
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import ListedColormap

def compare_labels(ds0, ds1, ds2, show_title=False):
    """
    Plot labels on a map.

    Args:
        ds0: The first input dataset.
        ds1: The second input dataset.
        ds2: The third input dataset.
        show_title (bool): Whether to show the title. Default is False.
    """

    fig, axs = plt.subplots(1,3,figsize=(18, 6), dpi=300)
    
    lon_min, lon_max = -180, 180
    lat_min, lat_max = 70, 90
    lat_step = 10
    lon_step = 20

    # Define your own color list
    color_list = ['red', 'saddlebrown', 'deepskyblue', 'purple','black','darkorange','forestgreen'] 
    cmap = ListedColormap(color_list)

    for i in range(3):
        if i==0:
            ds=ds0
        if i==1:
            ds=ds1
        if i==2:
            ds=ds2
            
        # Get the unique labels
        unique_labels = np.unique(ds.PCM_LABELS.values)
        num_labels = len(unique_labels)
        ax=axs[i]

        m = Basemap(projection='npstere', boundinglat=lat_min, lon_0=0, resolution='l', ax=ax)

        m.drawparallels(np.arange(lat_min, lat_max + 1, lat_step), labels=[], linewidth=0.1)
        m.drawcoastlines()
        m.fillcontinents(color='grey', lake_color='lightblue')
        m.drawmeridians(np.arange(lon_min, lon_max + 1, lon_step), labels=[1, 0, 0, 1], linewidth=0.1)


        x, y = m(ds['lon'], ds['lat'])
        sc = m.scatter(x, y, s=3, c=ds.PCM_LABELS.values, cmap=cmap, vmin=0, vmax=num_labels-1)

        cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
        cbar.ax.set_ylabel('class')
        cbar.set_ticks(np.arange(0, num_labels, 1))

        # Add a title and set the font size
        if show_title:
            ax.set_title(f'lat_step={ds.lat_step.values}, lon_step={ds.lon_step.values}, quantile4T={ds.quantile4T.values},nprof_of_train={ds.nprof_of_train.values}')

        for lat in np.arange(lat_min, lat_max + 1, lat_step):
            if lat < 80:
                x, y = m(0, lat+1)
            else:
                x, y = m(0, lat)

            ax.text(x, y, f'{lat}°N', fontsize=8, ha='center', va='center', color='k',
                    bbox=dict(facecolor='w', edgecolor='None', alpha=1))

    plt.show()

