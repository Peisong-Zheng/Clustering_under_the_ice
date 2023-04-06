import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_data_location(ds):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=100, subplot_kw={'projection':ccrs.AzimuthalEquidistant(central_latitude=90)})

    ax.add_feature(cfeature.COASTLINE.with_scale('10m'),linewidth=0.5)
    ax.add_feature(cfeature.LAND.with_scale('10m'))
    ax.add_feature(cfeature.OCEAN.with_scale('10m'))

    # add gridlines and set latitude labels
    gl = ax.gridlines(draw_labels=True)
    gl.left_labels = True
    gl.bottom_labels = True

    ax.scatter(ds['lon'].values, ds['lat'].values, c='r', transform=ccrs.PlateCarree(), zorder=4)
    plt.show()
    

def plot_pred_map(ds, m):
    proj = ccrs.NorthPolarStereo()
    subplot_kw={'projection': proj}
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5), dpi=120, facecolor='w', edgecolor='k', subplot_kw=subplot_kw)

    kmap = m.plot.cmap()
    sc = ax.scatter(ds['lon'], ds['lat'], s=3, c=ds['PCM_LABELS'], cmap=kmap, transform=ccrs.PlateCarree(), vmin=0, vmax=m.K)
    cl = m.plot.colorbar(ax=ax)

    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.COASTLINE)
    ax.set_title('LABELS of the training set')
    plt.show()
    

def plot_quantile(ds_group):
    fig, axs = plt.subplots(1, 3, figsize=(12, 12),dpi=200)

    for i in range(3):
        qua5=ds_group[0]['temperature_Q'][i][0].values
        qua50=ds_group[0]['temperature_Q'][i][1].values
        qua95=ds_group[0]['temperature_Q'][i][2].values
        ax=axs[i]
        ax.plot(qua5,ds_group[0]['pressure'].values,c='b',label=0.05)
        ax.plot(qua50,ds_group[0]['pressure'].values,c='r',label=0.50)
        ax.plot(qua95,ds_group[0]['pressure'].values,c='g',label=0.95)
        ax.grid(True)
        ax.legend()
        ax.set_title(f'Component {i+1}')
        ax.set_ylabel('Pressure')
        ax.set_xlabel('Temperature')

    plt.subplots_adjust(wspace=0.3, hspace=0.25)
    plt.show()