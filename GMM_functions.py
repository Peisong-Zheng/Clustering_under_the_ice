from pyxpcm.models import pcm
import numpy as np

def GMM(ds_train,ds_fit,K):  
    if np.max(ds_train.pressure) >=0:
        print('Warning, the pressure is not negative!!!')
    pstart=np.max(ds_train.pressure)
    pend=np.min(ds_train.pressure)
    z = np.arange(pstart,pend,-10.)

    pcm_features = {'temperature': z, 'salinity':z}
    m = pcm(K=K, features=pcm_features)
    
    features_in_ds = {'temperature': 'temperature', 'salinity': 'salinity'}

    m.fit(ds_train,features=features_in_ds,dim='pressure')
    ds_fit=m.predict(ds_fit,features=features_in_ds,dim='pressure',inplace=True)
    return ds_fit,m

##################################################################################
# did some post processing to the ds_fit
def post_processing(ds_fit,ds_train,m):
    ds_fit['lat_step']=ds_train['lat_step']
    ds_fit['lon_step']=ds_train['lon_step']
    ds_fit['quantile4T']=ds_train['quantile4T']
    ds_fit['nprof_of_train']=ds_train.nprof.size

    # # Calculate the quantile and plot it
    # for vname in ['temperature', 'salinity']:
    #     ds_fit = ds_fit.pyxpcm.quantile(m, q=[0.05, 0.5, 0.95], of=vname, outname=vname + '_Q', keep_attrs=True, inplace=True)

    features_in_ds = {'temperature': 'temperature', 'salinity': 'salinity'}
    m.predict_proba(ds_fit, features=features_in_ds, dim='pressure',inplace=True)

    for vname in ds_fit.data_vars:
        try:
            del ds_fit[vname].attrs['_pyXpcm_cleanable']
        except:
            print(f'{vname} has no _pyXpcm_cleanable attribute, skip it')



    return ds_fit


import prepare_training_data_functions as ptd

def do_gmm(ds,lat_step,lon_step,quantile4T,K=4):
    ds_train_latlonT = ptd.prepare_training_data(lat_step, lon_step, quantile4T, ds)
    ds_fit_latlonT, m_latlonT = GMM(ds_train_latlonT, ds.copy(), K)
    ds_fit_latlonT = post_processing(ds_fit_latlonT, ds_train_latlonT, m_latlonT)
    return ds_fit_latlonT

##################################################################################


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

def plot_labels(ds, show_title=False):
    # Get the unique labels
    unique_labels = np.unique(ds.PCM_LABELS.values)
    num_labels = len(unique_labels)

    fig, ax = plt.subplots(figsize=(16, 8), dpi=300)

    lon_min, lon_max = -180, 180
    lat_min, lat_max = 70, 90
    lat_step = 10
    lon_step = 20

    m = Basemap(projection='npstere', boundinglat=lat_min, lon_0=0, resolution='l', ax=ax)

    m.drawparallels(np.arange(lat_min, lat_max + 1, lat_step), labels=[], linewidth=0.1)
    m.drawcoastlines()
    m.fillcontinents(color='grey', lake_color='lightblue')
    m.drawmeridians(np.arange(lon_min, lon_max + 1, lon_step), labels=[1, 0, 0, 1], linewidth=0.1)

    # Create a custom colormap
    cmap = plt.cm.get_cmap('tab20', num_labels)
    x, y = m(ds['lon'], ds['lat'])
    sc = m.scatter(x, y, s=3, c=ds.PCM_LABELS.values, cmap=cmap, vmin=0, vmax=num_labels-1)

    cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cbar.ax.set_ylabel('class')
    cbar.set_ticks(np.arange(0, num_labels, 1))

    # Add a title and set the font size
    if show_title:
        plt.title(f'lat_step={ds.lat_step.values}, lon_step={ds.lon_step.values}, quantile4T={ds.quantile4T.values},nprof_of_train={ds.nprof_of_train.values}')


    for lat in np.arange(lat_min, lat_max + 1, lat_step):
        if lat < 80:
            x, y = m(0, lat+1)
        else:
            x, y = m(0, lat)

        ax.text(x, y, f'{lat}°N', fontsize=8, ha='center', va='center', color='k',
                bbox=dict(facecolor='w', edgecolor='None', alpha=1))

    plt.show()

##########################################################


import matplotlib.pyplot as plt
import numpy as np

def plot_mean_profile(ds_fit):

    unique_labels = np.unique(ds_fit.PCM_LABELS.values)
    num_classes = len(unique_labels)
    num_rows = 2  # Calculate the number of rows
    num_cols = num_classes  

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 5 * num_rows), dpi=300)

    for i in range(num_classes):

        ds_class = ds_fit.where(ds_fit.PCM_LABELS == i, drop=True)

        ax=axs[0,i]
        qua5 = np.quantile(ds_class['temperature'], 0.05, axis=0)
        qua95 = np.quantile(ds_class['temperature'], 0.95, axis=0)
        qua50 = np.quantile(ds_class['temperature'], 0.50, axis=0)

        ax.plot(qua5, ds_fit['pressure'].values, c='b', label=0.05)
        ax.plot(qua50, ds_fit['pressure'].values, c='r', label=0.50)
        ax.plot(qua95, ds_fit['pressure'].values, c='g', label=0.95)
        ax.grid(True)
        ax.legend()
        ax.set_title(f'Class {i}')

        ax.set_ylabel('Pressure')
        ax.set_xlabel('Temperature')

        # plot the salinity
        ax=axs[1,i]

        qua5 = np.quantile(ds_class['salinity'], 0.05, axis=0)
        qua95 = np.quantile(ds_class['salinity'], 0.95, axis=0)
        qua50 = np.quantile(ds_class['salinity'], 0.50, axis=0)

        ax.plot(qua5, ds_fit['pressure'].values, c='b', label=0.05)
        ax.plot(qua50, ds_fit['pressure'].values, c='r', label=0.50)
        ax.plot(qua95, ds_fit['pressure'].values, c='g', label=0.95)

        ax.grid(True)
        ax.legend()
        ax.set_title(f'Class {i}')

        ax.set_ylabel('Pressure')
        ax.set_xlabel('Salinity')

    plt.subplots_adjust(wspace=0.5, hspace=0.25)
    plt.show()
###############################################################################
# # plot the likelihood of each class
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.basemap import Basemap

# def plot_single_class_likelihood(ds):
#     nrows = np.ceil(ds.PCM_POST.shape[0] / 2).astype(int)
#     fig, axs = plt.subplots(nrows, 2, figsize=(16, 8 * nrows), dpi=300)

#     lon_min, lon_max = -180, 180
#     lat_min, lat_max = 70, 90
#     lat_step = 10
#     lon_step = 20

#     fig.subplots_adjust(hspace=0)  # Adjust the vertical spacing between subplots

#     for i in range(ds.PCM_POST.values.shape[0]):
#         ax = axs[i // 2, i % 2]
#         m = Basemap(projection='npstere', boundinglat=lat_min, lon_0=0, resolution='l', ax=ax)

#         m.drawparallels(np.arange(lat_min, lat_max + 1, lat_step), labels=[], linewidth=0.1)
#         m.drawcoastlines()
#         m.fillcontinents(color='grey', lake_color='lightblue')
#         m.drawmeridians(np.arange(lon_min, lon_max + 1, lon_step), labels=[1, 0, 0, 1], linewidth=0.1)

#         pcm_post = ds.PCM_POST.values[i, :]
#         x, y = m(ds['lon'], ds['lat'])
#         pcm_post[pcm_post < 0.4] = 0.05

#         sc = m.scatter(x, y, s=3, c=np.array([pcm_post]), cmap='Blues', vmin=0, vmax=np.max(pcm_post))

#         cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
#         cbar.ax.set_ylabel('likelihood')
#         # add a title and set the font size
#         ax.set_title(f'Likelihood of class {i}', fontsize=12)

#         for lat in np.arange(lat_min, lat_max + 1, lat_step):
#             if lat < 80:
#                 x, y = m(0, lat+1)
#             else:
#                 x, y = m(0, lat)
                
#             ax.text(x, y, f'{lat}°N', fontsize=8, ha='center', va='center', color='k',
#                     bbox=dict(facecolor='w', edgecolor='None', alpha=1))

#     plt.show()

###############################################################################
# plot the likelihood of each class
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

def plot_single_class_likelihood(ds):
    nrows = np.ceil(ds.PCM_POST.shape[0] / 2).astype(int)
    fig, axs = plt.subplots(nrows, 2, figsize=(16, 8 * nrows), dpi=300)

    lon_min, lon_max = -180, 180
    lat_min, lat_max = 70, 90
    lat_step = 10
    lon_step = 20

    fig.subplots_adjust(hspace=0)  # Adjust the vertical spacing between subplots

    for i in range(ds.PCM_POST.values.shape[0]):
        ax = axs[i // 2, i % 2]
        m = Basemap(projection='npstere', boundinglat=lat_min, lon_0=0, resolution='l', ax=ax)

        m.drawparallels(np.arange(lat_min, lat_max + 1, lat_step), labels=[], linewidth=0.1)
        m.drawcoastlines()
        m.fillcontinents(color='grey', lake_color='lightblue')
        m.drawmeridians(np.arange(lon_min, lon_max + 1, lon_step), labels=[1, 0, 0, 1], linewidth=0.1)

        pcm_post = ds.PCM_POST.values[i, :]
        x, y = m(ds['lon'], ds['lat'])
        pcm_post[pcm_post < 0.4] = 0.05

        sc = m.scatter(x, y, s=3, c=np.array([pcm_post]), cmap='Blues', vmin=0, vmax=np.max(pcm_post))
        # plot the pcm_post larger than 0.5
        m.scatter(x[pcm_post > 0.5], y[pcm_post > 0.5], s=3, c=np.array([pcm_post[pcm_post > 0.5]]), cmap='Blues', vmin=0, vmax=np.max(pcm_post))

        cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
        cbar.ax.set_ylabel('likelihood')
        # add a title and set the font size
        ax.set_title(f'Likelihood of class {i}', fontsize=12)

        for lat in np.arange(lat_min, lat_max + 1, lat_step):
            if lat < 80:
                x, y = m(0, lat+1)
            else:
                x, y = m(0, lat)
                
            ax.text(x, y, f'{lat}°N', fontsize=8, ha='center', va='center', color='k',
                    bbox=dict(facecolor='w', edgecolor='None', alpha=1))

    plt.show()
##################################################################################
# '''group the ds and plot the salinity-temperature profiles of classes in a plot'''
# def plot_san_temp_profiles(ds):

#     ds_group = ds.groupby('PCM_LABELS')
    
#     unique_labels = np.unique(ds.PCM_LABELS.values)
#     num_labels = len(unique_labels)
#     cmap = plt.cm.get_cmap('tab20', num_labels)

#     # create a figure
#     fig=plt.figure(figsize=(10,10),dpi=300)
#     # loop over the classes
#     for i in range(len(ds_group)):
#         # get the data of each class
#         ds_class=ds_group[i]
#         # plot the salinity-temperature profiles of each class
#         plt.plot(ds_class['salinity'],ds_class['temperature'],c=cmap(i),alpha=0.3)
#         plt.plot(ds_class['salinity'].values[0],ds_class['temperature'].values[0],c=cmap(i),label=f'class {i}',alpha=0.5)

#     plt.legend()
#     plt.xlabel('Salinity')
#     plt.ylabel('Temperature (°C)')


#     plt.show()

def plot_san_temp_profiles_onebyone(ds):
    ds_group = ds.groupby('PCM_LABELS')
    
    unique_labels = np.unique(ds.PCM_LABELS.values)
    num_labels = len(unique_labels)
    cmap = plt.cm.get_cmap('tab20', num_labels)

    # Calculate the number of rows and columns for the subplots
    num_plots = len(ds_group)
    nrows = (num_plots + 1) // 2
    ncols = 2

    # Create a figure with subplots
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 6 * nrows), dpi=300)
    fig.subplots_adjust(hspace=0.3)

    # Loop over the classes and plot the subplots
    for i in range(len(ds_group)):
        ds_class=ds_group[i]
        # Calculate the subplot index
        row = i // ncols
        col = i % ncols

        # Get the corresponding axis for the subplot
        if nrows > 1:
            ax = axs[row, col]
        else:
            ax = axs[col]

        # Plot the salinity-temperature profiles of each class
        ax.plot(ds_class['salinity'], ds_class['temperature'], c=cmap(i), alpha=0.3)
        ax.plot(ds_class['salinity'].values[0], ds_class['temperature'].values[0], c=cmap(i), label=f'class {i}', alpha=0.5)

        # plot the mean profile
        ax.plot(ds_class['salinity'].mean(dim='nprof'), ds_class['temperature'].mean(dim='nprof'), c='k', label=f'mean of class {i}', alpha=1)

        ax.legend()
        ax.set_xlabel('Salinity')
        ax.set_ylabel('Temperature (°C)')

    plt.show()


##################################################################################
import numpy as np

def relabel_pcm_labels(ds,label_mapping = {0: 0, 1: 3, 2: 1, 3: 2}):
    unique_labels = np.unique(ds.PCM_LABELS.values)
    num_classes = len(unique_labels)
    if num_classes>4:
        print('The number of unique labels is not 4, please check the input ds')
        return
              
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

##################################################################################
'''group the ds and plot the salinity-temperature profiles of classes in a plot'''
import matplotlib.pyplot as plt

def plot_san_temp_profiles(ds):
    
    unique_labels = np.unique(ds.PCM_LABELS.values)
    num_labels = len(unique_labels)

    if num_labels>4:
        print('The number of unique labels is not 4, please check the input ds')
        return
              

    cmap = plt.cm.get_cmap('tab20', num_labels)

    # create a figure
    fig=plt.figure(figsize=(10,10),dpi=300)

    sequence = [3,1,0,2]
    symbols = ['o','s','^','d']
    # loop over the classes
    for i in sequence:
        # get the data of each class
        ds_class = ds.where(ds.PCM_LABELS == i, drop=True)
        # plot the salinity-temperature profiles of each class
        plt.plot(ds_class['salinity'],ds_class['temperature'],c=cmap(i),alpha=1)
        plt.plot(ds_class['salinity'].values[0],ds_class['temperature'].values[0],c=cmap(i),label=f'class {i}',alpha=1)
    
    for i in sequence:
        # get the data of each class
        ds_class = ds.where(ds.PCM_LABELS == i, drop=True)
        # plot the mean profile on top
        plt.plot(ds_class['salinity'].mean(dim='nprof'),ds_class['temperature'].mean(dim='nprof'),c='k', marker=symbols[i],markersize=3,label=f'mean of class {i}',alpha=1)
        # plot the mean profile as scatter plot using the symbol of the class 
        # plt.scatter(ds_class['salinity'].mean(dim='nprof'),ds_class['temperature'].mean(dim='nprof'),c=cmap(i),marker=symbols[i],alpha=1)
        # plt.plot(ds_class['salinity'].mean(dim='nprof'),ds_class['temperature'].mean(dim='nprof'),c=cmap(i),marker=symbols[i],label=f'mean of class {i}',alpha=1)
        # plt.plot(ds_class['salinity'].mean(dim='nprof'),ds_class['temperature'].mean(dim='nprof'),c='k',alpha=1)

    plt.legend()
    plt.xlabel('Salinity')
    plt.ylabel('Temperature (°C)')


    plt.show()
