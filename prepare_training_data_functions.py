
# function that plot three histograms, the first is the month distribution, the second and third are the longtidue and latitude distribution
import matplotlib.pyplot as plt
import seaborn as sns

def plot_distribution(ds):
    months = ds.date.dt.month.values

    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
    fig, ax = plt.subplots(1,3,figsize=(20,5),dpi=300)
    sns.histplot(months,ax=ax[0])
    sns.histplot(ds.lon,ax=ax[1])
    sns.histplot(ds.lat,ax=ax[2])
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

#############################################################################################################
# function to plot month distribution
import matplotlib.pyplot as plt
import seaborn as sns

def plot_month_distribution(ds):
    months = ds.date.dt.month.values

    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
    fig, ax = plt.subplots(1,1,figsize=(5,5),dpi=300)
    sns.histplot(months,ax=ax)
    # set the x and y label
    ax.set_xlabel('Month')
    ax.set_ylabel('Count')
    # set the title
    # ax.set_title('Month distribution')
    plt.show()


#############################################################################################################
# function that plot the the number of profiles in each bin
import pandas as pd
import numpy as np
import plotly.graph_objs as go

def plot_latlon_bins(ds, lat_step, lon_step,show_threshold=False,threshold_quantile=0.95):
    # Convert the 'lat' and 'lon' variables into a pandas dataframe
    df_latlon = ds[['lat', 'lon']].to_dataframe()

    lon_bins = pd.cut(df_latlon['lon'], bins=np.arange(-180,185, lon_step))
    lat_bins = pd.cut(df_latlon['lat'], bins=np.arange(-90,95, lat_step))

    latlon_groups = df_latlon.groupby([lon_bins, lat_bins])

    g_len=[]
    for group_key in latlon_groups.groups:
        try:
            group = latlon_groups.get_group(group_key)
            #print(f"Group {group_key}: {len(group)} rows")
            g_len.append(len(group))
        except KeyError:
            # Handle the KeyError gracefully
            print(f"No data points for group {group_key}")


    g_len_sorted = sorted(g_len)
    transition_point = None
  
    g_len_diff=np.diff(g_len_sorted)

    threshold = np.quantile(g_len_diff,threshold_quantile)  # set a threshold for the difference between adjacent elements
    
    for i in range(1, len(g_len_sorted)):
        if g_len_sorted[i] - g_len_sorted[i-1] > threshold:
            transition_point = i-1
            break

    # Create a trace for the plot
    trace = go.Scatter(
        x = list(range(len(g_len_sorted))),
        y = g_len_sorted
    )

    # Add a red dot at the transition point
    if show_threshold:  
        if transition_point is not None:
            red_dot = go.Scatter(
                x=[transition_point],
                y=[g_len_sorted[transition_point]],
                mode='markers',
                marker=dict(color='red', size=10)
            )
            fig_data = [trace, red_dot]
        else:
            fig_data = [trace]
    else:
        fig_data = [trace]

    # Create the layout for the plot
    layout = go.Layout(
        xaxis=dict(title='Bin No.'),
        yaxis=dict(title='Number of profiles')
    )

    # Create the figure object and plot it
    fig = go.Figure(data=fig_data, layout=layout)
    fig.show()


#############################################################################################################
def cal_threshold(ds, lat_step, lon_step, quantile):
    # Convert the 'lat' and 'lon' variables into a pandas dataframe
    df_latlon = ds[['lat', 'lon']].to_dataframe()

    lon_bins = pd.cut(df_latlon['lon'], bins=np.arange(-180,185, lon_step))
    lat_bins = pd.cut(df_latlon['lat'], bins=np.arange(-90,95, lat_step))

    latlon_groups = df_latlon.groupby([lon_bins, lat_bins])

    g_len=[]
    for group_key in latlon_groups.groups:
        try:
            group = latlon_groups.get_group(group_key)
            #print(f"Group {group_key}: {len(group)} rows")
            g_len.append(len(group))
        except KeyError:
            # Handle the KeyError gracefully
            print(f"No data points for group {group_key}")


    g_len_sorted = sorted(g_len)
    transition_point = None
  
    g_len_diff=np.diff(g_len_sorted)

    threshold = np.quantile(g_len_diff,quantile)  # set a threshold for the difference between adjacent elements
    
    for i in range(1, len(g_len_sorted)):
        if g_len_sorted[i] - g_len_sorted[i-1] > threshold:
            transition_point = i-1
            break

    return g_len_sorted[transition_point]



#############################################################################################################
# function that calculate ds length
def cal_length(ds,lat_step,lon_step, threshold):
    # Convert the 'lat' and 'lon' variables into a pandas dataframe
    df_latlon = ds[['lat', 'lon']].to_dataframe()

    lon_bins = pd.cut(df_latlon['lon'], bins=np.arange(-180,185, lon_step))
    lat_bins = pd.cut(df_latlon['lat'], bins=np.arange(-90,95, lat_step))

    latlon_groups = df_latlon.groupby([lon_bins, lat_bins])

    # Create a list to store the groups with sample size larger than the threshold
    groups_drop_len_list = []

    # Loop through the groups
    for group_key in latlon_groups.groups:
        try:
            group = latlon_groups.get_group(group_key)
            if len(group) > threshold:
                groups_drop_len_list.append(len(group)-threshold)
                #print((len(group)-threshold))
        except KeyError:
            # Handle the KeyError gracefully
            # print(f"No data points for group {group_key}")
            continue

    ds_len=len(ds['nprof'])-sum(groups_drop_len_list)

    return ds_len

#############################################################################################################
# function that drop the profiles in the bins with sample size larger than the threshold
def drop_profiles(ds,lat_step,lon_step, threshold):
    # Convert the 'lat' and 'lon' variables into a pandas dataframe
    df_latlon = ds[['lat', 'lon']].to_dataframe()

    lon_bins = pd.cut(df_latlon['lon'], bins=np.arange(-180,185, lon_step))
    lat_bins = pd.cut(df_latlon['lat'], bins=np.arange(-90,95, lat_step))

    latlon_groups = df_latlon.groupby([lon_bins, lat_bins])

    # Create a list to store the groups with sample size larger than the threshold
    latlon_groups_list = []

    # Loop through the groups
    for group_key in latlon_groups.groups:
        try:
            group = latlon_groups.get_group(group_key)
            if len(group) > threshold:
                # randomly sample the group of the dropped data
                group_drop = group.sample(n=len(group)-threshold, random_state=0)
                #print(f"Dropped data: Group {group_key}: {len(group_drop)} rows")
                # replace the group with the group_sample
                latlon_groups_list.append(group_drop)
        except KeyError:
            # Handle the KeyError gracefully
            print(f"No data points for group {group_key}")

    # create a new xarray that contains the data with sample size smaller than the threshold
    ds_spatially_sampled = ds.copy()

    # loop through the latlon_groups_list and create a list to store the nprof of the dropped data
    nprof_dropped = []

    for group in latlon_groups_list:
        # get the nprof of the dropped data
        temp_index = group.index.values
        #print(temp_index)
        nprof_dropped.append(temp_index.tolist())
    
    # reshape the list to 1D
    nprof_dropped = [item for sublist in nprof_dropped for item in sublist]
    # print the nprof_dropped
    #print(nprof_dropped)
    
    # drop the data 
    ds_spatially_sampled = ds_spatially_sampled.drop_sel(nprof=nprof_dropped)
    
    return ds_spatially_sampled


########################################################################################################################



# sample profiles by month
import numpy as np
import xarray as xr

def ramdon_sample_by_month(ds, ratio, random_seed=0):
    if ratio < 0 or ratio > 1:
        raise ValueError("Ratio must be between 0 and 1.")
        
    if 'date' not in ds:
        raise ValueError("Input dataset must have a 'date' variable.")
        
    if 'nprof' not in ds.dims:
        raise ValueError("Input dataset must have a 'nprof' dimension.")
    
    # group the dataset by month
    ds_monthly = ds.groupby('date.month')
    months_with_data = [month for month in ds_monthly.groups.keys() if len(ds_monthly[month]) > 0]
    
    if not months_with_data:
        raise ValueError("Input dataset does not contain any data.")
    
    # find the month with the smallest amount of data
    smallest_month_size = np.min([ds_monthly[month].nprof.size for month in months_with_data])
    selected_size_monthly=ds.nprof.size*ratio/12
    
    if smallest_month_size < selected_size_monthly:
        selected_size_monthly=smallest_month_size
        # ratio=smallest_month_size/ds.nprof.size*12
        print(f"Ratio is too large, selected_size_monthly is reset to the smallest_month_size: {smallest_month_size}")
    

    # print(f"Ratio is {ratio}, {int(selected_size_monthly)} profiles are selected from each month")
    # sample a fixed amount of data from each month
    ds_selected = xr.concat([ds_monthly[month].isel(nprof=np.random.default_rng(random_seed).choice(ds_monthly[month].nprof.size, int(selected_size_monthly), replace=False)) for month in months_with_data], dim='nprof')
    
    ds_selected['nprof'] = np.arange(ds_selected.sizes['nprof'])
    
    return ds_selected


########################################################################################################################
# function to plot the heatmap of the profiles distribution
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

def plot_heatmap(ds_cleaned, bin_max=None):
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


########################################################################################################################

# find the bining scheme that maximizes the number of profiles

def latlon_bining_ana(ds):
    lat_bins=[2,5,10]
    lon_bins=[10,20,30,40,50,60,70,80]
    ds_len=[]
    threshold=[]
    for i in lat_bins:
        for j in lon_bins:
            # T=cal_threshold(ds, i, j, quantile=0.95)
            T=int(cal_threshold(ds, i, j, quantile=0.95))
            ds_spatially_sampled = drop_profiles(ds, i, j, T)
            ds_monthly_sampled = ramdon_sample_by_month(ds_spatially_sampled, 1, random_seed=0)
            threshold.append(T)
            ds_len.append(ds_monthly_sampled.nprof.size)

    threshold=np.array(threshold)
    threshold=threshold.reshape(len(lat_bins),len(lon_bins))
    threshold=threshold.T
    threshold=pd.DataFrame(threshold,index=lon_bins,columns=lat_bins)

    # plot the ds_len changes with i and j as line plot
    ds_len=np.array(ds_len)
    ds_len=ds_len.reshape(len(lat_bins),len(lon_bins))
    ds_len=ds_len.T
    ds_len=pd.DataFrame(ds_len,index=lon_bins,columns=lat_bins)

    # plot the heatmap of the threshold and number of profiles
    fig, axs = plt.subplots(1, 2, figsize=(14, 6),dpi=300)
    sns.set_theme(style="whitegrid")

    # plot the heatmap of the threshold
    sns.heatmap(threshold, annot=True, linewidths=.5, cmap='YlGnBu', ax=axs[0])
    axs[0].set_title('Threshold')
    axs[0].set_xlabel('Latitude bin')
    axs[0].set_ylabel('Longitude bin')

    # plot the heatmap of the number of profiles
    sns.heatmap(ds_len, annot=True, linewidths=.5, cmap='YlGnBu', ax=axs[1])
    axs[1].set_title('Number of profiles')
    axs[1].set_xlabel('Latitude bin')
    axs[1].set_ylabel('Longitude bin')
    
    plt.show()


# make a pipeline to prepare the training data
def prepare_training_data(lat_step,lon_step,quantile4T,ds_cleaned):
    threshold=int(cal_threshold(ds_cleaned, lat_step, lon_step, quantile=quantile4T))
    ds_cleaned_spatially_sampled = drop_profiles(ds_cleaned, lat_step, lon_step, threshold)
    ds_cleaned_monthly_sampled = ramdon_sample_by_month(ds_cleaned_spatially_sampled, 1, random_seed=0)
    ds_cleaned_monthly_sampled['lat_step']=lat_step
    ds_cleaned_monthly_sampled['lon_step']=lon_step
    ds_cleaned_monthly_sampled['quantile4T']=quantile4T
    print(f'threshold={threshold}, training dataset size={len(ds_cleaned_monthly_sampled.nprof)}')
    return ds_cleaned_monthly_sampled


def add_info(ds_train,ds_fit):
    ds_fit['lat_step']=ds_train['lat_step']
    ds_fit['lon_step']=ds_train['lon_step']
    ds_fit['quantile4T']=ds_train['quantile4T']
    ds_fit['nprof_of_train']=ds_train.nprof.size
    return ds_fit