import numpy as np
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import contextily as ctx
from spatial_tools import fof
import os
from stat_tools import estimations

list_locs = {
    'Manhica' : [32.80722050, -25.40221980], #From Pau Cisteró excel
    'Maputo' : [32.576388888888889, -25.915277666666],
    #'Montepuez' : [38.99972150, -13.12555980], #From Pau Cisteró excel
    'Chokwe' : [33.005166666666666667, -24.5252777777777777],
    #'Moatize' : [33.73333040, -16.11666620], #From Pau Cisteró excel
    #'Dondo' : [34.75, -19.6166666666666667],
    'Magude' : [32.64216410, -25.02049992], #From Pau Cisteró excel
    'Ilha Josina' : [32.92210000, -25.09330000], #From Pau Cisteró excel
    'Xinavane' : [32.791885, -25.048534],
    'Panjane' : [32.352430, -24.899469],
    'Motaze' : [32.860569, -24.810357],
    'Mapulanguene' : [32.081602, -24.491015],
    'Taninga' : [32.825796, -25.182094],
    'Palmeira' : [32.869766, -25.261457],
    #'Massinga' : [35.37405260,-23.32666250], #From Pau Cisteró excel
    #'Mopeia' : [35.71338490, -17.97391000], #From Pau Cisteró excel
    }

locations = pd.DataFrame({'location' : [i for i in list_locs], 'longitude': [list_locs[i][0] for i in list_locs], 'latitude': [list_locs[i][1] for i in list_locs]})
locations = geopandas.GeoDataFrame(locations, geometry = geopandas.points_from_xy(locations['longitude'], locations['latitude']))
locations = locations.set_crs(epsg=4326)
locations = locations.to_crs(epsg=3857)

def visualise_all_fofs(fof_catalogue_mip, fof_catalogue_c210, \
                       mipmon, cross210, mip_positive, c210_positive, \
                       mip_mask_year, c210_mask_year, \
                       mean_pr_fof_mip, mean_pr_fof_c210, \
                       fofid_mip, fofid_c210, \
                       min_size_fof = 2, min_pr_fof = 0., max_p_fof = 0.25, \
                      pop1name = 'MiPMon', pop2name = 'children 2-10', \
                      pos1name = r'$P.\ falciparum$ infection (detected by PCR)', \
                       pos2name = r'$P.\ falciparum$ infection (detected by PCR)', \
                      extra_title = "", xlims = None, ylims = None):

    print("Number of FOFs in " + pop1name+ ":", len(fof_catalogue_mip))
    mask_mip = (fof_catalogue_mip['positives']>=min_size_fof)&\
                (fof_catalogue_mip['mean_pr']>=min_pr_fof)&\
                (fof_catalogue_mip['p']<=max_p_fof)
    print("Number of FOFs in " + pop1name+ " with >=", str(min_size_fof), "positives and PR>=" + \
          str(min_pr_fof) + " and p-value <= " +str(max_p_fof) + " :", \
          np.sum(mask_mip))

    print("Number of FOFs in " + pop2name+ ":", len(fof_catalogue_c210))
    mask_c210 = (fof_catalogue_c210['positives']>=min_size_fof)&\
                (fof_catalogue_c210['mean_pr']>=min_pr_fof)&\
                (fof_catalogue_c210['p']<=max_p_fof)
    print("Number of FOFs in " + pop2name+ " with >=", str(min_size_fof), "positives and PR>=" +\
          str(min_pr_fof) + " and p-value <= " +str(max_p_fof) + " :", \
          np.sum(mask_c210))

    #Mapping FOFs
    ax = mipmon.plot(markersize = 0, figsize = [8,8], alpha = 0)
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    fof_catalogue_mip[mask_mip].plot(ax = ax, column = 'mean_pr', markersize = 30*fof_catalogue_mip['positives'][mask_mip], alpha = .75, \
                       label = pop1name, cmap = 'rainbow', vmin = 0, vmax = 1, legend = True)
    if np.sum(mask_mip) == 0:
        legend = True
    else:
        legend = False
    fof_catalogue_c210[mask_c210].plot(ax = ax, column = 'mean_pr', markersize = 30*fof_catalogue_c210['positives'][mask_c210], alpha = .75, \
                        label = pop2name, cmap = 'turbo', vmin = 0, vmax = 1, legend = legend, marker = 's')
    locations.plot(ax = ax, color = 'k', markersize = 20)
    locations.apply(lambda x: ax.annotate(s=x.location, xy=x.loc['geometry'].coords[0]), axis=1)
    plt.legend()
    plt.title("Spatial distribution of hotspots" + extra_title)
    if xlims is not None:
        plt.xlim(xlims[0], xlims[1])
    if ylims is not None:
        plt.ylim(ylims[0], ylims[1])
    plt.show()

    #Mapping pregnant women
    print("Number of FOFs:", len(fof_catalogue_mip[mask_mip]))
    unique_fofids = fof_catalogue_mip[mask_mip]['id'].unique()#TODO test
    ax = mipmon.plot(markersize = 0, figsize = [8,8], alpha = 0)
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    mipmon[mip_mask_year].plot(ax = ax, color = 'k', markersize = 5, alpha = 1, label = 'All study ' + pop1name)
    id_mask = np.array([f in unique_fofids for f in fofid_mip], dtype = bool)#change to kept ids

    mipmon[mip_positive&mip_mask_year][id_mask].plot(ax = ax, column = mean_pr_fof_mip[id_mask], \
                                                       markersize = 15, alpha = 1, \
                                                       label = pos1name, \
                                                       cmap = 'rainbow', legend = True)
    locations.plot(ax = ax, color = 'k', markersize = 20)
    locations.apply(lambda x: ax.annotate(s=x.location, xy=x.loc['geometry'].coords[0]), axis=1)
    plt.legend()
    plt.title("Spatial distribution of " + pop1name + " data" + extra_title)
    if xlims is not None:
        plt.xlim(xlims[0], xlims[1])
    if ylims is not None:
        plt.ylim(ylims[0], ylims[1])
    plt.show()


    #Mapping children
    print("Number of FOFs:", len(fof_catalogue_c210[mask_c210]))
    unique_fofids = fof_catalogue_c210[mask_c210]['id'].unique()#TODO test
    ax = mipmon.plot(markersize = 0, figsize = [8,8], alpha = 0)
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    cross210[c210_mask_year].plot(ax = ax, color = 'k', markersize = 5, alpha = 1, label = 'All study ' + pop2name)
    id_mask = np.array([f in unique_fofids for f in fofid_c210], dtype = bool)#change to kept ids
    cross210[c210_positive&c210_mask_year][id_mask].plot(ax = ax, column = mean_pr_fof_c210[id_mask], markersize = 15, alpha = 1, \
                                        label = pos2name, \
                                       cmap = 'rainbow', legend = True)
    locations.plot(ax = ax, color = 'k', markersize = 20)
    locations.apply(lambda x: ax.annotate(s=x.location, xy=x.loc['geometry'].coords[0]), axis=1)
    plt.legend()
    plt.title("Spatial distribution of "+pop2name + extra_title)
    if xlims is not None:
        plt.xlim(xlims[0], xlims[1])
    if ylims is not None:
        plt.ylim(ylims[0], ylims[1])
    plt.show()

    plot_fof_hists(fof_catalogue_mip, fof_catalogue_c210, pop1name = pop1name, \
                  pop2name = pop2name, fof_mip_mask = mask_mip, fof_c210_mask = mask_c210)

def plot_fof_hists(fof_catalogue_mip, fof_catalogue_c210, \
                   pop1name = 'MiPMon', pop2name = 'children', \
                   fof_mip_mask = None, fof_c210_mask = None):
    if fof_mip_mask is None:
        fof_cat_mip_plot = fof_catalogue_mip
    else:
        fof_cat_mip_plot = fof_catalogue_mip[fof_mip_mask]
    if fof_c210_mask is None:
        fof_cat_c210_plot = fof_catalogue_c210
    else:
        fof_cat_c210_plot = fof_catalogue_c210[fof_c210_mask]

    ranges = get_ranges(fof_cat_mip_plot['positives'], fof_cat_c210_plot['positives'])
    plt.hist(fof_cat_mip_plot['positives'], 20, range = ranges, \
             alpha = .5, label = pop1name)
    plt.hist(fof_cat_c210_plot['positives'], 20, range = ranges, \
             alpha = .5, label = pop2name)
    plt.title("Number of positives per FOF")
    plt.xlabel("Number of positives in FOF")
    plt.ylabel("Number of FOF found")
    plt.legend()
    plt.show()

    ranges = get_ranges(fof_cat_mip_plot['total'], fof_cat_c210_plot['total'])
    plt.hist(fof_cat_mip_plot['total'], 20, range = ranges, \
             alpha = .5, label = pop1name)
    plt.hist(fof_cat_c210_plot['total'], 20, range = ranges, \
             alpha = .5, label = pop2name)
    plt.title("Number of samples per FOF")
    plt.xlabel("Number of samples in FOF")
    plt.ylabel("Number of FOF found")
    plt.legend()
    plt.show()

    ranges = get_ranges(fof_cat_mip_plot['p'], fof_cat_c210_plot['p'])
    plt.hist(fof_cat_mip_plot['p'], 40, range = ranges, \
             alpha = .5, label = pop1name)
    plt.hist(fof_cat_c210_plot['p'], 40, range = ranges, \
             alpha = .5, label = pop2name)
    plt.title("P-value distribution of FOF")
    plt.xlabel("P-value")
    plt.ylabel("Number of FOF found")
    plt.legend()
    plt.show()

def get_ranges(var1, var2):
    mins = min([var1.min(), \
                var2.min()])
    maxs = max([var1.max(), \
                var2.max()])
    ranges = [mins, maxs]
    return ranges

def get_fof_data(scale, mipmon_positions, mipmon_test, cross210_positions, cross210_test):#TODO add get_rands, nrands
    #Running FOF for MiPMon and building its database
    fofid_mip, mean_pr_fof_mip, pval_fof_mip, fof_catalogue_mip = fof.get_fof_PR(mipmon_positions, mipmon_test, scale, fofid = None)
    #TODO get N (nrands) random fof catalogues from shuffled mipmon_test

    #Running FOF for cross-sectionals and building its database
    fofid_c210, mean_pr_fof_c210, pval_fof_c210, fof_catalogue_c210 = fof.get_fof_PR(cross210_positions, cross210_test, scale, fofid = None)
    #TODO get N (nrands) random fof catalogues from shuffled cross210_test
    return fofid_mip, mean_pr_fof_mip, pval_fof_mip, fof_catalogue_mip, fofid_c210, mean_pr_fof_c210, pval_fof_c210, fof_catalogue_c210 #TODO return random_fofs is computed

def get_time_masks(cross_df, mipmon_df, year, time_width = None, verbose = False):
    #Mask to select the year of the data
    cross_mask = cross_df['year'] == year
    if verbose:
        print('Sample size cross-sectional:', np.sum(cross_mask))
    if time_width is None:
        mip_mask = mipmon_df['year'] == year
    else:
        mean_cross_data = cross_df['visdate'][cross_mask].mean()
        mip_mask = get_time_mask_mipmon(mipmon_df, mean_cross_data, time_width, verbose = verbose)
    return cross_mask, mip_mask

def get_time_mask_mipmon(mipmon_df, date, time_width, verbose = True):
    #Selection a window aroung the cross-sectional data for ANC
    time_0 = date - pd.to_timedelta(time_width/2, unit = 'D')
    time_1 = date + pd.to_timedelta(time_width/2, unit = 'D')
    mip_mask = (mipmon_df['visdate'] >= time_0)&(mipmon_df['visdate'] <= time_1)
    if verbose:
        print('Sample size all 1st ANC:', np.sum(mip_mask))
    return mip_mask

def get_pos_test(mipmon, mip_mask_year, cross210, c210_mask_year, pos1_name = 'pcrpos', pos2_name = 'pospcr'):
    mipmon_positions = np.array((mipmon['x'][mip_mask_year], mipmon['y'][mip_mask_year])).T
    cross210_positions = np.array((cross210['x'][c210_mask_year], cross210['y'][c210_mask_year])).T
    mipmon_test = np.array(mipmon[pos1_name][mip_mask_year])
    cross210_test = np.array(cross210[pos2_name][c210_mask_year])
    return mipmon_positions, cross210_positions, mipmon_test, cross210_test

general_sero = ['MSP1','HSP40', 'Etramp', 'ACS5','EBA175', \
 'PfTramp','GEXP18','PfRH2','PfRH5', 'pAMA1', 'PvLDH', \
'PfHRP2', 'PfLDH']
pregnancy_sero = ['P1', 'P39','P5', 'P8','PD', 'DBL6e','DBL34']
peptides = ['P1', 'P39','P5', 'P8','PD']

def import_spatial_data(data_path = '~/isglobal/projects/pregmal/data/', \
                        cross_name = 'cross_merged.csv', \
                        mipmon_name = 'mipmon_merged.csv', \
                       serology_name = 'MiPMON_serostatus_wide.csv', \
                       intermediate_val = 1, \
                       excluded_antigens = ['pAMA1', 'PvLDH', 'P5', 'DBL6e', 'PfHRP2', \
                                             'PfLDH', 'general_pos', 'pregnancy_pos', \
                                           'breadth_general', 'breadth_pregnancy', \
                                            'breadth_peptides'],
                       study_year = True):
    cross_filename = data_path + cross_name
    mipmon_filename = data_path + mipmon_name
    if serology_name is None:
        serology_filename = None
    else:
        serology_filename = data_path + serology_name
    #Load cross
    cross = pd.read_csv(cross_filename)
    #Geo locate cross
    cross = geopandas.GeoDataFrame(cross, geometry = geopandas.points_from_xy(cross['lng'], cross['lat']))
    cross = cross.set_crs(epsg=4326)
    cross = cross.to_crs(epsg=3857)
    #Filter Cross to only 2-10
    cross210 = cross[(cross['age']>=2)&(cross['age']<10)]
    #Load mipmon
    mipmon = pd.read_csv(mipmon_filename)
    #Geo locate mipmon
    mipmon = geopandas.GeoDataFrame(mipmon, geometry = geopandas.points_from_xy(mipmon['longitude'], mipmon['latitude']))
    mipmon = mipmon.set_crs(epsg=4326)
    mipmon = mipmon.to_crs(epsg=3857)
    cross['visdate'] = pd.to_datetime(cross['visdate'])
    cross210['visdate'] = pd.to_datetime(cross210['visdate'])
    mipmon['visdate'] = pd.to_datetime(mipmon['visdate'])
    if study_year:
        mipmon['year'] = get_study_year(mipmon)
        cross['year'] = get_study_year(cross)
        cross210['year'] = get_study_year(cross210)
    else:
        mipmon['year'] = mipmon['visdate'].dt.year
        cross['year'] = cross['visdate'].dt.year
        cross210['year'] = cross210['visdate'].dt.year
    #Quantify tests
    cross['pospcr'][cross['pospcr'] == 'Negative'] = 0.
    cross['pospcr'][cross['pospcr'] == 'Positive'] = 1.
    cross['rdt'][cross['rdt'] == 'Negative'] = 0.
    cross['rdt'][cross['rdt'] == 'Positive'] = 1.
    cross210['pospcr'][cross210['pospcr'] == 'Negative'] = 0.
    cross210['pospcr'][cross210['pospcr'] == 'Positive'] = 1.
    cross210['rdt'][cross210['rdt'] == 'Negative'] = 0.
    cross210['rdt'][cross210['rdt'] == 'Positive'] = 1.

    mipmon['pcrpos'][mipmon['pcrpos'] == 'PCR-'] = 0
    mipmon['pcrpos'][mipmon['pcrpos'] == 'PCR+'] = 1
    #Fixing one crazy value
    cross['lat'][cross['lat']>0] = -cross['lat'][cross['lat']>0]

    if serology_filename is not None:
        serology = pd.read_csv(serology_filename)
        #Merging serological data with MiPMon data
        mipmon = pd.merge(mipmon, serology, on = 'nida', how = 'inner')

        #Defining antigens and their values
        antigens = []
        for a in general_sero:
            if 'FMM_' + a in mipmon.columns and a not in excluded_antigens:
                antigens.append(a)
        for a in pregnancy_sero:
            if 'FMM_' + a in mipmon.columns and a not in excluded_antigens:
                antigens.append(a)

        #Defining binary quantities of cutoff. Intermediates are considered negative for the moment
        for cutoff in ['FMM_', 'NC_']:
            for a in antigens:
                mipmon[cutoff+a][mipmon[cutoff+a] == 'negative'] = 0
                mipmon[cutoff+a][mipmon[cutoff+a] == 'positive'] = 1
                mipmon[cutoff+a][mipmon[cutoff+a] == 'intermediate'] = intermediate_val
            #Defining breadth and overall positivity
            mipmon[cutoff+'breadth_general'] = pd.Series(np.zeros_like(mipmon['pcrpos']))
            mipmon[cutoff+'breadth_pregnancy'] = pd.Series(np.zeros_like(mipmon['pcrpos']))
            mipmon[cutoff+'breadth_peptides'] = pd.Series(np.zeros_like(mipmon['pcrpos']))
            for a in antigens:
                if a in peptides:
                    mipmon[cutoff+'breadth_peptides'] += mipmon[cutoff + a]
                if a in general_sero:
                    mipmon[cutoff+'breadth_general'] += mipmon[cutoff + a]
                elif a in pregnancy_sero:
                    mipmon[cutoff+'breadth_pregnancy'] += mipmon[cutoff + a]
                else:
                    print("Warning: invalid serological option:", a)
            mipmon[cutoff+'general_pos'] = np.array(mipmon[cutoff+'breadth_general'] >= 1, dtype = int)
            mipmon[cutoff+'pregnancy_pos'] = np.array(mipmon[cutoff+'breadth_pregnancy'] >= 1, dtype = int)
            mipmon[cutoff+'All peptides'] = np.array(mipmon[cutoff+'breadth_peptides'] >= 1, dtype = int)
        if 'breadth_general' not in excluded_antigens:
            antigens.append('breadth_general')
        if 'breadth_pregnancy' not in excluded_antigens:
            antigens.append('breadth_pregnancy')
        if 'general_pos' not in excluded_antigens:
            antigens.append('general_pos')
        if 'pregnancy_pos' not in excluded_antigens:
            antigens.append('pregnancy_pos')
        if 'All peptides' not in excluded_antigens:
            antigens.append('All peptides')

        return mipmon, cross, cross210, antigens
    else:
        return mipmon, cross, cross210

def get_label_list(df_list, label = 'tempID'):
    """
    This method gives the unique values of a column in a list
    of data frames.

    Parameters:
    -----------
    df_list: list of pandas.DataFrames
        List of dataframes
    label: str
        Name of column to select

    Returns:
    --------
    label_list: list
        List of unique values of the column over all dataframes
    """
    for i in range(len(df_list)):
        mask = df_list[i][label].notnull()
        if i == 0:
            label_list = df_list[i][label].loc[mask].unique()
        else:
            label_list = np.unique(np.concatenate((label_list, df_list[i][label].loc[mask].unique())))
    return label_list

def get_temporal_hotspots(index, time_width, time_steps, scale, min_num, linking_time, \
                          linking_dist, test_result = None, show_maps = True, gif_delay = 30, method = 'fof', \
                         output_path = '/tmp/', save = True, bins2d = 50, label2plot = 'lifetime', \
                         kernel_size = 1, max_p_lifetimes = 1, name_end = '', plot_date = 'month', \
                         xlims = None, ylims = None):
    #temporal loop
    min_date = index['date'].min()
    max_date = index['date'].max()
    #Reseting loop variables
    mean_date = []
    hot_num = []
    num_cases_in_hot = []
    mean_hot_size = []
    fraction_cases_in_hot = []
    fof_catalogues = []
    step_num = 0
    if type(bins2d) is int:
        hist2d_hotspots = np.zeros((bins2d, bins2d))
    elif len(bins2d) == 2:
        hist2d_hotspots = np.zeros((bins2d[0], bins2d[1]))
    else:
        print("Warning: wrong assignment of bins2d variable. Taking only first value")
        hist2d_hotspots = np.zeros((bins2d[0], bins2d[0]))
    while min_date + pd.to_timedelta(time_steps*step_num + time_width, unit = 'D') < max_date: #the last time frame is the last fully overlapping the data

        #select data
        selected_data = (index['date'] >= min_date + pd.to_timedelta(time_steps*step_num, unit = 'D'))& \
                        (index['date'] <= min_date + pd.to_timedelta(time_steps*step_num + time_width, unit = 'D'))

        #get hotspots
        positions = np.array((index[selected_data]['x'], index[selected_data]['y'])).T
        if test_result is None:
            test_result_selected = np.ones(index[selected_data]['x'].shape[0])
        else:
            test_result_selected = np.array(test_result[selected_data])
        if method == 'fof':
            #fofid = fof.get_fofid(positions, scale)
            fofid, mean_pr_fof, pval_fof, fof_catalogue = fof.get_fof_PR(positions, test_result_selected, scale)
            #Removing FOFs with less than min_num cases
            fofid = fof_cut(fofid, min_num)
            fof_catalogue = fof_catalogue[fof_catalogue['positives'] >= min_num]
        elif method == 'radial':
            #fofid = fof.get_fofid(positions, scale, min_num)
            fofid, mean_pr_fof, pval_fof, fof_catalogue = fof.get_fof_PR(positions, test_result_selected, scale, \
                                                               min_neighbours = min_num)

        fof_catalogue = fof_catalogue[fof_catalogue['p'] <= max_p_lifetimes]#Removing non-significant hotspots TODO test
        hotspot = fofid
        for i, h in enumerate(hotspot):#Removing non-significant hotspots from fofid of positives TODO test
            if h not in fof_catalogue['id'].unique():
                hotspot[i] = 0

        #Plotting maps
        xrange = [index['x'].min()-1000, index['x'].max()+1000]
        yrange = [index['y'].min()-1000, index['y'].max()+1000]
        positives = test_result_selected == 1
        negatives = test_result_selected == 0
        df_to_plot = index[selected_data]
        if save or show_maps:
            ax = df_to_plot.plot(markersize = 0, figsize = [8,8], alpha = 0)
            if np.sum(negatives) > 0:
                df_to_plot[negatives].plot(ax = ax, color = 'tab:green', markersize = 15, figsize = [8,8], alpha = .5, label = 'Negative case')
            df_to_plot[positives].plot(ax = ax, color = 'tab:red', markersize = 15, figsize = [8,8], alpha = 1, label = 'Positive case outside hotspot')
            if np.sum(hotspot > 0) == 0:#To keep the legend of cases in hotspots even when there are no cases
                mockdf = pd.DataFrame({'lng' : [.0], 'lat' : [.0]})
                mockdf = geopandas.GeoDataFrame(mockdf, geometry = geopandas.points_from_xy(mockdf['lng'], mockdf['lat']))
                mockdf.plot(ax = ax, color = 'tab:orange', markersize = 15, figsize = [8,8], alpha = 1, label = 'Positive case in hotspot')
            else:
                df_to_plot[positives][hotspot > 0].plot(ax = ax, color = 'tab:orange', markersize = 15, figsize = [8,8], alpha = 1, label = 'Positive case in hotspot')
            ax.set_xlim(xrange[0], xrange[1])
            ax.set_ylim(yrange[0], yrange[1])
            ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
            plt.legend(loc = 'upper left')
            date1 = pd.to_datetime(pd.Series(min_date + pd.to_timedelta(time_steps*step_num, unit = 'D')))
            date1 = date1.dt.strftime('%d/%m/%Y')[0]
            date2 = pd.to_datetime(pd.Series(min_date + pd.to_timedelta(time_steps*(step_num + 1), unit = 'D')))
            date2 = date2.dt.strftime('%d/%m/%Y')[0]
            mean_month = pd.to_datetime(pd.Series(min_date + pd.to_timedelta(time_steps*(step_num + .5), unit = 'D')))
            mean_month = mean_month.dt.strftime('%m/%Y')[0]
            if plot_date == 'range':
                plt.annotate(str(date1) + ' - ' + str(date2), xy = [.6,.8], xycoords = 'figure fraction')
            elif plot_date == 'month':
                plt.annotate(str(mean_month), xy = [.65,.8], xycoords = 'figure fraction')
            if xlims is not None:
                plt.xlim(xlims[0], xlims[1])
            if ylims is not None:
                plt.ylim(ylims[0], ylims[1])
            if save:
                save_file_names = method + '_map_tw' + str(time_width) + '_ts' + str(time_steps) + '_min' + str(min_num) + '_scale' + str(scale) + name_end
                plt.savefig(os.path.join(output_path,save_file_names + '_' + str(1000+step_num) + '.png'))
            if show_maps:
                plt.show()
            else:
                plt.close()

        #Save quantities
        hist2d_hotspots += np.histogram2d(positions[:,0][positives][hotspot > 0], \
                                          positions[:,1][positives][hotspot > 0], \
                                          bins = bins2d, range = (xrange, yrange))[0] > 0
        mean_date.append(min_date + pd.to_timedelta(time_steps*(step_num + .5), unit = 'D'))
        fof_catalogue['Date'] = mean_date[-1]
        fof_catalogues.append(fof_catalogue)

        num_cases_in_hot.append(np.sum(hotspot>0))
        fraction_cases_in_hot.append(num_cases_in_hot[-1]/len(hotspot))
        hot_num.append(len(np.unique(fofid[fofid>0])))
        if hot_num[-1] == 0:
            mean_hot_size.append(np.nan)
        else:
            mean_hot_size.append(num_cases_in_hot[-1]/hot_num[-1])

        step_num +=1

    fof_catalogues_lowp = [fcat[fcat['p'] <= max_p_lifetimes] for fcat in fof_catalogues]
    fof_catalogues_lowp = fof.get_temp_id(fof_catalogues_lowp, linking_time, linking_dist)
    #translating time_steps to days in lifetime.
    #TODO: get the dates in the fof_catalogues and use them for linking hotspots
    for t in range(len(fof_catalogues)):
        fof_catalogues_lowp[t]['lifetime'] *= time_steps

    mean_date = pd.to_datetime(mean_date)

    #TODO make plots below for fof_catalogues_lowp
    #Plotting hotspot statistics over time
    plt.figure(figsize = [8,8])
    hist2d_hotspots[hist2d_hotspots == 0] = np.nan
    plt.imshow(hist2d_hotspots.T[::-1], cmap = 'turbo')
    plt.title('Number of hotspot appearences per pixel')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.show()

    plt.plot(mean_date, hot_num)
    plt.title('Number of hotspots')
    plt.ylabel('Number of hotspots')
    plt.xlabel('Date')
    plt.xticks(index['date'].sort_values()[::int(len(index['date'])/3)-1])
    plt.show()

    plt.plot(mean_date, mean_hot_size, marker = 'o')
    plt.title('Mean hotspot size')
    plt.ylabel('Mean hotspot size')
    plt.xlabel('Date')
    plt.xticks(index['date'].sort_values()[::int(len(index['date'])/3)-1])
    plt.show()

    plt.plot(mean_date, num_cases_in_hot)
    plt.title('Number of cases in hotspots')
    plt.ylabel('Number of cases in hotspots')
    plt.xlabel('Date')
    plt.xticks(index['date'].sort_values()[::int(len(index['date'])/3)-1])
    plt.show()

    plt.plot(mean_date, fraction_cases_in_hot)
    plt.title('Fraction of cases in hotspots')
    plt.ylabel('Fraction of cases in hotspots')
    plt.xlabel('Date')
    plt.xticks(index['date'].sort_values()[::int(len(index['date'])/3)-1])
    plt.show()

    plot_label(fof_catalogues_lowp, xrange, yrange, label2plot, xlims = xlims, ylims = ylims)

    hist_timelifes(fof_catalogues_lowp)

    lifetime_timeline(fof_catalogues_lowp, mean_date, time_steps, kernel_size = kernel_size)

    if save:
        #Generating GIF of maps
        gif_output_name = output_path + method + '_tw' + str(time_width) + '_ts' + str(time_steps) + '_min' + str(min_num) + '_scale' + str(scale)  + name_end + '.gif'
        os.system('convert -delay ' + str(gif_delay) + ' -loop 0 ' + output_path + save_file_names + '*png ' +gif_output_name)
        print("GIF saved in", gif_output_name)

    return mean_date, hot_num, num_cases_in_hot, mean_hot_size, fraction_cases_in_hot, \
            fof_catalogues_lowp

def plot_label(fof_cat_list, xrange, yrange, label = 'lifetime', vmin = None, vmax = None, \
              xlims = None, ylims = None):
    timelifes = fof.get_label_list(fof_cat_list, label = label)
    plot_started = False
    for t in range(len(fof_cat_list)):
        if  len(fof_cat_list[t]) > 0:
            if plot_started is False:
                ax = fof_cat_list[t].plot(column = label, markersize = 15, \
                                             figsize = [9,9], vmin = vmin, vmax = vmax, \
                                             cmap = 'turbo', legend = True)
                plot_started = True
            else:
                 fof_cat_list[t].plot(ax = ax, column = label, markersize = 15, \
                                         vmin = 0, vmax = vmax, cmap = 'turbo')
    ax.set_xlim(xrange)
    ax.set_ylim(yrange)
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    if xlims is not None:
        plt.xlim(xlims[0], xlims[1])
    if ylims is not None:
        plt.ylim(ylims[0], ylims[1])
    plt.show()

def lifetime_timeline(fof_cat, mean_date_test, time_steps, kernel_size = 1):
    tempids = fof.get_label_list(fof_cat, label = 'tempID')
    lifetimes = fof.get_label_list(fof_cat, label = 'lifetime')
    print(lifetimes)
    print("Maximum limelife:", max(lifetimes))

    for i, tempid in enumerate(tempids):
        num_cases = []
        dates = []
        min_date = fof_cat[0]['Date'].min()
        for f in range(len(fof_cat)):
            dates.append(min_date + pd.to_timedelta(f*time_steps, unit = 'D'))
            if tempid in fof_cat[f]['tempID'].unique():
                mask = fof_cat[f]['tempID'] == tempid
                num_cases.append(np.sum(fof_cat[f][mask]['total']))
                lifetime = fof_cat[f][mask]['lifetime'].iloc[0]
            else:
                num_cases.append(0)
        dates = pd.to_datetime(dates)
        num_cases = np.array(num_cases)

        num_cases_convolved = estimations.convolve_ones(num_cases, kernel_size)
        plt.plot(mean_date_test, num_cases_convolved, color = cm.turbo(lifetime/max(lifetimes)), \
                lw = 2)
    plt.xlabel("Date")
    plt.xticks(mean_date_test[::int(len(mean_date_test)/4)])
    plt.ylabel("Number of cases in hotspot")
    plt.ylim(ymin = 0.1)
    cNorm = colors.Normalize(vmin=0, vmax=max(lifetimes))
    plt.colorbar(cm.ScalarMappable(norm = cNorm, cmap='turbo'), label = 'Lifetime')
    plt.show()


def fof_cut(fofid, min_num):
    """
    This method remove the FOF ids with less than a given
    number of cases.

    Parameters:
    fofid: np.array
        List of IDs of the FOFs, with 0 meaning no FOF
    min_num: int
        Minimum number of cases in FOFs to be kept

    Returns:
    fofid: np.array
        Final list of FOF IDs
    """
    for i, f in enumerate(np.unique(fofid[fofid>0])):
        if np.sum(fofid == f) < min_num:
            fofid[fofid == f] = 0
    return fofid

def hist_timelifes(fof_catalogues, show = True, label = '', alpha = 1, \
                   c = None, range = None):
    all_tempids = fof.get_label_list(fof_catalogues, label = 'tempID')
    if len(all_tempids) > 0:
        lifetimes = []
        for tempid in all_tempids:
            for t in fof_catalogues:
                if tempid in t['tempID'].unique():
                    mask = t['tempID'] == tempid
                    lifetimes.append(np.mean(t['lifetime'].loc[mask]))
                    break
        plt.hist(lifetimes, 50, label = label, alpha = alpha, color = c, \
                range = range)
        plt.xlabel("Time duration of hotspots (days)")
        plt.ylabel("Number of hotspots")
        if show:
            plt.show()

def generate_mock_data(population_size, positivity_rate, positive_distribution, \
                       seed = None, save = True, output_file = 'mock_dataframe.csv'):
    """
    This method generates mock data of a spatial distribution
    of negative and positive cases.

    Parameters:
    -----------
    population_size: int
        Sample size
    positivity_rate: float
        Fraction of positive cases
    positive_distribution: str {'rand', 'clustered', 'sinusoidal'}
        Option for the distribution of positive cases
    seed: int
        Random seed for the random number generation
    save: bool
        If true, the dataframe is saved as a csv
    output_file: str
        Name of output file

    Returns:
    --------
    x_rand: np.array
        Positions in x-axis
    y_rand: np.array
        Positions in y-axis
    position_rand: np.ndarray
        2D-vector with x and y positions
    positive: np.array
        Boolean mask for positive cases
    test_rand: np.array
        Test results, 1 for positive, 0 for negative
    mock_df: pd.DataFrame
        Dataframe of the mock data
    """
    if seed is not None:
        np.random.seed(seed)
    #Negative cases
    x_rand = np.random.rand(int(population_size*(1 - positivity_rate)))
    y_rand = np.random.rand(int(population_size*(1 - positivity_rate)))
    #Positive cases
    if positive_distribution == 'rand':
        x_rand_p = np.random.rand(int(population_size*positivity_rate))
        y_rand_p = np.random.rand(int(population_size*positivity_rate))
        #Concatenating all cases
        x_rand = np.concatenate((x_rand, x_rand_p))
        y_rand = np.concatenate((y_rand, y_rand_p))
    elif positive_distribution == 'sinusoidal':
        x_rand_p = np.random.rand(int(population_size*positivity_rate))
        y_rand_p = .5 + .5*np.sin(10*x_rand_p)
        #Concatenating all cases
        x_rand = np.concatenate((x_rand, x_rand_p))
        y_rand = np.concatenate((y_rand, y_rand_p))
    elif positive_distribution == 'clustered':
        for i in range(4):
            x_rand_p = np.random.normal(.1 + .8*np.random.rand(), .05, int(population_size*positivity_rate/4))
            y_rand_p = np.random.normal(.1 + .8*np.random.rand(), .05, int(population_size*positivity_rate/4))
            #Concatenating all cases
            x_rand = np.concatenate((x_rand, x_rand_p))
            y_rand = np.concatenate((y_rand, y_rand_p))
    #2D vector for positions
    position_rand = np.array([x_rand,y_rand]).T
    #Defining positivity and test result
    positive = np.arange(population_size) > population_size*(1 - positivity_rate)
    test_rand = np.array(positive, dtype = float)
    mock_df = get_mock_dataframe(x_rand, y_rand, test_rand, save, output_file)
    return x_rand, y_rand, position_rand, positive, test_rand, mock_df

def get_mock_dataframe(x_rand, y_rand, test_rand, save = True, output_file = 'mock_dataframe.csv'):
    """
    This method creates a dataframe of the mock data.

    Parameters:
    -----------
    x_rand: np.array
        Positions in x-axis
    y_rand: np.array
        Positions in y-axis
    test_rand: np.array
        Test results, 1 for positive, 0 for negative
    save: bool
        If true, the dataframe is saved as a csv
    output_file: str
        Name of output file

    Returns:
    --------
    mock_df: pd.DataFrame
        Dataframe of the mock data
    """
    mock_df = pd.DataFrame(
    {
        'id' : np.arange(len(x_rand), dtype = int),
        'x' : x_rand,
        'y' : y_rand,
        'test' : test_rand,
        'case_count' : np.ones_like(x_rand),
        'dates' : pd.to_datetime(['2017-02-02' for i in range(len(x_rand))])
    })
    mock_df = geopandas.GeoDataFrame(mock_df, geometry = geopandas.points_from_xy(mock_df['x'], mock_df['y']))
    if save:
        mock_df.to_csv(output_file)
    return mock_df

def visualise_fof_results(x_data, y_data, positive, pr_data, id_data, fof_catalogue, \
                         show_plots = ['id', 'pr', 'p-value', 'p-hist', 'ncases-hist', 'falseposfrac'], \
                         excluded_plots = []):
    if 'id' in show_plots and 'id' not in excluded_plots:
        plt.scatter(x_data, y_data, c = 'tab:grey')
        plt.scatter(x_data[positive][id_data > 0], y_data[positive][id_data > 0], \
                    c = id_data[id_data > 0], cmap = 'turbo')
        plt.colorbar()
        plt.title("ID of hotspots")
        plt.show()

    if 'pr' in show_plots and 'pr' not in excluded_plots:
        plt.scatter(x_data, y_data, c = 'tab:grey')
        plt.scatter(x_data[positive][id_data > 0], y_data[positive][id_data > 0], \
                    c = pr_data[id_data > 0], cmap = 'turbo')
        plt.colorbar()
        plt.title("PR of hotspots")
        plt.show()

    if 'p-value' in show_plots and 'p-value' not in excluded_plots:
        plt.scatter(x_data, y_data, c = 'tab:grey')
        p_vals = np.array([fof_catalogue[fof_catalogue['id'] == i]['p'] for i in id_data[id_data > 0]], \
                          dtype = float)
        plt.scatter(x_data[positive][id_data > 0], y_data[positive][id_data > 0], \
                    c = p_vals, cmap = 'turbo')
        plt.colorbar()
        plt.title("P-value of hotspots")
        plt.show()

    if 'p-hist' in show_plots and 'p-hist' not in excluded_plots:
        size_thresholds = [3,4,5]
        colors = [cm.inferno((i+1)/(len(size_thresholds)+1)) for i in range(len(size_thresholds))]
        hist, edges, patches = plt.hist(fof_catalogue['p'], 40, range = [0,1], \
                                        color = 'k', label = 'All sizes')
        for i, s in enumerate(size_thresholds):
            plt.hist(fof_catalogue['p'][fof_catalogue['total'] > s], 40, \
                     range = [0,1], label = '>'+str(s)+' cases', color = colors[i], alpha = .75)
        plt.xlim(0,1)
        plt.vlines(.05, 0, np.max(hist)*1.1, color = 'tab:orange')
        plt.ylabel("Number of hotspots")
        plt.xlabel("P-value")
        plt.legend()
        plt.show()

    if 'ncases-hist' in show_plots and 'ncases-hist' not in excluded_plots:
        hist_all, edges, patches = plt.hist(fof_catalogue['total'], 20, \
                                            range = [np.min(fof_catalogue['total']), \
                                                     np.max(fof_catalogue['total'])], \
                                            color = 'tab:red', label = 'p < 0.05')
        hist_highp, edges, patches = plt.hist(fof_catalogue['total'][fof_catalogue['p']>.05], 20, \
                                              range = [np.min(fof_catalogue['total']), \
                                                       np.max(fof_catalogue['total'])], \
                                              color = 'tab:blue', label = 'p > 0.05')
        plt.ylabel("Number of hotspots")
        plt.xlabel("Number of cases in hotspots")
        plt.legend()
        plt.show()

    if 'falseposfrac' in show_plots and 'falseposfrac' not in excluded_plots:
        mask = np.isfinite(hist_highp/hist_all)
        plt.plot(edges[:-1][mask], 1 - hist_highp[mask]/hist_all[mask], label = 'False positive fraction')
        plt.plot(edges[:-1][mask], 0*edges[:-1][mask] + .05, c = 'tab:grey', label = '10%')
        plt.xlabel("Number of cases in hotspots")
        plt.ylabel("False positive fraction")
        plt.legend()
        plt.show()


def visualise_satscan_results(x_data, y_data, positive, points_data, clusters_data, \
                         show_plots = ['id', 'pr', 'p-value', 'p-hist', 'ncases-hist', 'falseposfrac'], \
                         excluded_plots = []):
    if 'id' in show_plots and 'id' not in excluded_plots:
        plt.scatter(x_data, y_data, c = 'tab:grey')
        mask = points_data['gis.LOC_OBS'] == 1
        plt.scatter(points_data['gis.LOC_LONG'][mask], points_data['gis.LOC_LAT'][mask], \
                    c = points_data['gis.CLUSTER'][mask], cmap = 'turbo')
        plt.colorbar()
        plt.title("ID of hotspots")
        plt.show()

    if 'pr' in show_plots and 'pr' not in excluded_plots:
        plt.scatter(x_data, y_data, c = 'tab:grey')
        mask = points_data['gis.LOC_OBS'] == 1
        plt.scatter(points_data['gis.LOC_LONG'][mask], points_data['gis.LOC_LAT'][mask], \
                    c = points_data['gis.CLU_OBS'][mask]/points_data['gis.CLU_POP'][mask], cmap = 'turbo')
        plt.colorbar()
        plt.title("PR of hotspots")
        plt.show()

    if 'p-value' in show_plots and 'p-value' not in excluded_plots:
        plt.scatter(x_data, y_data, c = 'tab:grey')
        mask = points_data['gis.LOC_OBS'] == 1
        plt.scatter(points_data['gis.LOC_LONG'][mask], points_data['gis.LOC_LAT'][mask], \
                    c = points_data['gis.P_VALUE'][mask], cmap = 'turbo')
        plt.colorbar()
        plt.title("P-value of hotspots")
        plt.show()

    if 'p-hist' in show_plots and 'p-hist' not in excluded_plots:
        size_thresholds = [3,4,5]
        colors = [cm.inferno((i+1)/(len(size_thresholds)+1)) for i in range(len(size_thresholds))]
        hist, edges, patches = plt.hist(clusters_data['col.P_VALUE'], 40, range = [0,1], \
                                        color = 'k', label = 'All sizes')
        for i, s in enumerate(size_thresholds):
            plt.hist(clusters_data['col.P_VALUE'][clusters_data['col.POPULATION'] > s], 40, \
                     range = [0,1], label = '>'+str(s)+' cases', color = colors[i], alpha = .75)
        plt.xlim(0,1)
        plt.vlines(.05, 0, np.max(hist)*1.1, color = 'tab:orange')
        plt.ylabel("Number of hotspots")
        plt.xlabel("P-value")
        plt.legend()
        plt.show()

    if 'ncases-hist' in show_plots and 'ncases-hist' not in excluded_plots:
        hist_all, edges, patches = plt.hist(clusters_data['col.POPULATION'], 20, \
                                            range = [np.min(clusters_data['col.POPULATION']), \
                                                     np.max(clusters_data['col.POPULATION'])], \
                                            color = 'tab:red', label = 'p < 0.05')
        hist_highp, edges, patches = plt.hist(clusters_data['col.POPULATION'][clusters_data['col.P_VALUE']>.05], 20, \
                                              range = [np.min(clusters_data['col.POPULATION']), \
                                                       np.max(clusters_data['col.POPULATION'])], \
                                              color = 'tab:blue', label = 'p > 0.05')
        plt.ylabel("Number of hotspots")
        plt.xlabel("Number of cases in hotspots")
        plt.legend()
        plt.show()

    if 'falseposfrac' in show_plots and 'falseposfrac' not in excluded_plots:
        mask = np.isfinite(hist_highp/hist_all)
        plt.plot(edges[:-1][mask], 1 - hist_highp[mask]/hist_all[mask], label = 'False positive fraction')
        plt.plot(edges[:-1][mask], 0*edges[:-1][mask] + .05, c = 'tab:grey', label = '10%')
        plt.xlabel("Number of cases in hotspots")
        plt.ylabel("False positive fraction")
        plt.legend()
        plt.show()

def get_study_year(mipmon):
    year_1 = ['2016-11-01', '2017-10-31']
    year_2 = ['2017-11-01', '2018-10-31']
    year_3 = ['2018-11-01', '2019-10-31']
    study_year = np.zeros_like(mipmon['visdate'], dtype = int)
    for i in range(len(mipmon['visdate'])):
        if mipmon['visdate'].iloc[i] >= pd.to_datetime(year_1[0]) and mipmon['visdate'].iloc[i] <= pd.to_datetime(year_1[1]):
            study_year[i] = 1
        elif mipmon['visdate'].iloc[i] >= pd.to_datetime(year_2[0]) and mipmon['visdate'].iloc[i] <= pd.to_datetime(year_2[1]):
            study_year[i] = 2
        elif mipmon['visdate'].iloc[i] >= pd.to_datetime(year_3[0]) and mipmon['visdate'].iloc[i] <= pd.to_datetime(year_3[1]):
            study_year[i] = 3
    return study_year
