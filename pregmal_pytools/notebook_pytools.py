import numpy as np
import pandas as pd
import geopandas
from genomic_tools import utils
from genomic_tools import stats
from matplotlib import cm
import matplotlib.pyplot as plt
import scipy.stats as sci_stats
import scipy.optimize as optimization
from stat_tools.estimations import smooth
from stat_tools.errors import chi_square
import pdb

def get_cross_prev_bins(cross210, cross_areas, test_type, cross_bins, \
print_sizes = True, cross_mask = None, verbose = True, ret_resamples = False):
    """
    This method calculates the mean prevalence in some time bins for
    cross-sectional surveys.

    Parameters:
    -----------
    cross210: pd.DataFrame
        Data frame of cross-sectional
    cross_areas: list
        List of areas to include from the cross-sectional data
    test_type: str {'pcr', 'rdt'}
        Type of test information used
    cross_bins: list
        Edges of time bins
    print_sizes: bool
        It specifies whether the sample sizes per bin are shown (default True)
    cross_mask: pd.DataFrame
        Boolean mask defining a selection of cross-sectional samples
    verbose: bool
        It specifies the verbose mode
    ret_resamples: bool
        If True, the measurements of all the resamples are return

    Returns:
    --------
    cross_dates: list
        Mean dates of data per time bin
    cross_mean: np.array
        Mean prevalence per time bin
    cross_err: np.array
        Error of mean prevalence per time bin
    """
    #Define mean Dates per cross-sectional
    if cross_mask is None:
        cross_areas_mask = utils.get_cross_area_mask(cross210, cross_areas)
    else:
        cross_areas_mask = utils.get_cross_area_mask(cross210, cross_areas)&cross_mask
    if test_type == 'rdt':
        mask = cross210['rdt'].notnull()&cross_areas_mask
        test_cross = cross210['rdt']
    elif test_type == 'pcr':
        mask = cross210['pospcr'].notnull()&cross_areas_mask
        test_cross = cross210['pospcr']
    #Prevalence of Cross
    cross_dates, cross_mean, cross_err, cross_means = \
    stats.mean_prev_time_bins(cross210['visdate'], test_cross, \
                                data_mask = mask, nbins = cross_bins, \
                                nrands = 1000, weights = cross210['weight'], \
                                verbose = verbose, ret_resamples = True)

    if print_sizes:
        for i in range(len(cross_bins)-1):
            time_mask = (cross210['visdate'] >= cross_bins[i])&(cross210['visdate'] < cross_bins[i+1])
            print("Number of Cross samples in bin " + str(i) + ": " + str(np.sum(mask&time_mask)))
    if ret_resamples:
        return cross_dates, cross_mean, cross_err, cross_means
    else:
        return cross_dates, cross_mean, cross_err

def get_mipmon_prev_bins(mipmon, mipmon_areas, test_type, cross_dates, time_width, \
                        time_shift, mask = None, print_sizes = True, \
                        verbose = True, ret_resamples = False):
    """
    This method calculates the mean prevalence of MiPMon data in some cross-sectional dates.

    Parameters:
    -----------
    mipmon: pd.DataFrame
        Data frame for MiPMon samples
    mipmon_areas: list
        List of areas to incluse from MiPMon samples
    test_type: str {'pcr', 'rdt'}
        Type of test information used
    cross_dates: list
        List of dates of cross-sectionals
    time_width: int
        Time width of MiPMon time bins to use (in days)
    time_shift: int
        Number of days to shift MiPMon cases
    mask: np.array
        Boolean mask to select a MiPMon subsample
    print_sizes: bool
        It specifies whether the sample sizes per bin are shown (default True)
    verbose: bool
        It specifies the verbose mode
    ret_resamples: bool
        If True, the measurements of all the resamples are return

    Returns:
    --------
    mipmon_dates: list
        Mean dates of data per time bin
    mipmon_mean: np.array
        Mean prevalence per time bin
    mipmon_err: np.array
        Error of mean prevalence per time bin
    """
    #Define area and test masks
    mipmon_areas_mask = utils.get_mipmon_area_mask(mipmon, mipmon_areas)
    if mask is None:
        mask = mipmon['visdate'].notnull()
    if test_type == 'rdt':
        mask = mask&mipmon_areas_mask
        test_mipmon = mipmon['density'] >= 100
    elif test_type == 'pcr':
        mask = mask&mipmon['pcrpos'].notnull()&mipmon_areas_mask
        test_mipmon = mipmon['pcrpos']
    else:
        mask = mask&mipmon[test_type].notnull()&mipmon_areas_mask
        test_mipmon = mipmon[test_type]
    #Define MiPMon time bins
    mipmon_bins = []
    for i in cross_dates:
        time_0 = pd.to_datetime(i) + pd.to_timedelta(time_shift - time_width/2, unit = 'D')
        time_1 = pd.to_datetime(i) + pd.to_timedelta(time_shift + time_width/2, unit = 'D')
        mipmon_bins.append(time_0)
        mipmon_bins.append(time_1)
        if print_sizes:
            time_mask = (mipmon['visdate'] >= time_0)&(mipmon['visdate'] < time_1)
            print("Number of MiPMon samples in bin around " + str(i) + ": " + \
            str(np.sum(mask&time_mask)))

    #Prevalence of MiPMon
    mipmon_dates, mipmon_mean, mipmon_err, mipmon_means = \
    stats.mean_prev_time_bins(mipmon['visdate'], test_mipmon, \
                                data_mask = mask, nbins = mipmon_bins, \
                                nrands = 1000, verbose = verbose, \
                                ret_resamples = True)
    mipmon_dates, mipmon_mean, mipmon_err, mipmon_means = mipmon_dates[::2], \
                            mipmon_mean[::2], mipmon_err[::2], mipmon_means[::2]
    if ret_resamples:
        return mipmon_dates, mipmon_mean, mipmon_err, mipmon_means
    else:
        return mipmon_dates, mipmon_mean, mipmon_err

def scatter_linfit_pcc(cross210, mipmon, mipmon_areas, cross_areas, cross_test_type, mip_test_type, cross_bins, \
                           time_width, time_shift, print_sizes = True, \
                           cross_mask = None, mip_mask = None, title = '', \
                           verbose = True, show_fit = True, show_identity = True, show = True, \
                      xmin = None, xmax = None, colors = None):
    """
    This method shows the scatter comparison between MiPMon and Cross-sectional
    data and outputs the linear fit parameters and the Pearson CC.
    """
    if colors is None:
        colors = [cm.turbo((i+1)/float(len(mipmon_areas) + 1)) for i in range(len(mipmon_areas) + 1)]
    #plt.figure(figsize = [9,6])
    all_cross = np.array([])
    all_mip = np.array([])
    all_cross_err = np.array([])
    all_mip_err = np.array([])
    plt.figure(figsize = [.9*5,.9*3.5])
    for i in range(len(mipmon_areas)):
        cross_dates, cross_mean, cross_err = get_cross_prev_bins(cross210, cross_areas[i], cross_test_type, cross_bins, \
                                                                 print_sizes = print_sizes, cross_mask = cross_mask, \
                                                                 verbose = verbose)
        #MiPMon
        if mip_mask is None:
            mipmon_mask = mipmon['visit'].notnull()
        else:
            mipmon_mask = mipmon['visit'].notnull()&mip_mask
        mipmon_dates, mipmon_mean, mipmon_err = get_mipmon_prev_bins(mipmon, mipmon_areas[i], mip_test_type, cross_dates, \
                                                                     time_width, time_shift, mask = mipmon_mask, \
                                                                     print_sizes = print_sizes, \
                                                                     verbose = verbose)
        plt.errorbar(cross_mean, mipmon_mean, mipmon_err, xerr = cross_err, label = mipmon_areas[i][0], c = colors[i], \
                     linestyle = '', lw = 1, marker = 's')
        all_cross = np.concatenate((all_cross, cross_mean))
        all_cross_err = np.concatenate((all_cross_err, cross_err))
        all_mip = np.concatenate((all_mip, mipmon_mean))
        all_mip_err = np.concatenate((all_mip_err, mipmon_err))

    #Linear fits
    mask = all_cross**2. >= 0
    p_all = np.polyfit(all_cross[mask], all_mip[mask], 1)#, w = 1/(all_cross_err**2 + all_mip_err**2))
    #Printing results
    print("Cross vs MiPMon " + title + ':')
    print("Linear parameter fitting: origin = " + str(p_all[1]) + ", slope = " + str(p_all[0]))
    pcorr_all = sci_stats.pearsonr(all_cross[mask], all_mip[mask])[0]
    print("Pearson CC: ", pcorr_all)
    print()
    #Plotting results
    all_vals = np.concatenate((all_cross[mask] + all_cross_err[mask], \
                                all_mip[mask] + all_mip_err[mask]))
    all_vals_min = np.concatenate((all_cross[mask] - all_cross_err[mask],
                                    all_mip[mask] - all_mip_err[mask]))
    if xmin is None:
        xmin = np.min(all_vals)*.8 - .02
    if xmax is None:
        xmax = np.max(all_vals)*1.05
    xplot = np.array([xmin, xmax])
    if show_identity:
        plt.plot(xplot, xplot, color = 'tab:grey', label = 'Identity')
    if show_fit:
        plt.plot(xplot, p_all[1] + p_all[0]*xplot, lw = 2, color = 'k', linestyle = '--', label = 'Linear regression')
    if cross_test_type == 'rdt':
        testlabel = 'RDT'
    elif cross_test_type == 'pcr':
        testlabel = 'qPCR'
    plt.xlabel(r'$Pf\rm{PR}_{\rm{'+testlabel+'}}$ cross-sectional (2-10 years)')
    if mip_test_type == 'rdt':
        testlabel = 'RDT'
    elif mip_test_type == 'pcr':
        testlabel = 'qPCR'
    plt.ylabel(r'$Pf\rm{PR}_{\rm{'+testlabel+'}}$ pregnant women')
    plt.title(title)
    plt.xlim(xmin, xmax)
    plt.legend()
    if show:
        plt.show()
    return p_all, pcorr_all

def show_scatter_cross_mip(cross210, mipmon, mipmon_areas, cross_areas, cross_test_type, mip_test_type, cross_bins, \
                           time_width, time_shift, print_sizes = True, lin_fit = True, \
                           show = ['pn', 'all', 'pg'], cross_mask = None, mip_mask = None):
    """
    This method shows the scatter comparison between MiPMon and cross-sectional data
    """
    colors = [cm.turbo((i+1)/float(len(mipmon_areas) + 1)) for i in range(len(mipmon_areas) + 1)]
    plt.figure(figsize = [9,6])
    if lin_fit:
        all_cross = np.array([])
        all_mip_pn = np.array([])
        all_mip = np.array([])
        all_mip_pn_pg = np.array([])
        all_cross_err = np.array([])
        all_mip_pn_err = np.array([])
        all_mip_err = np.array([])
        all_mip_pn_pg_err = np.array([])
    for i in range(len(mipmon_areas)):
        cross_dates, cross_mean, cross_err = get_cross_prev_bins(cross210, cross_areas[i], cross_test_type, cross_bins, \
                                                                 print_sizes = print_sizes, cross_mask = cross_mask)
        #MiPMon at PN
        if mip_mask is None:
            mipmon_mask = mipmon['visit'] == 'PN'
        else:
            mipmon_mask = (mipmon['visit'] == 'PN')&mip_mask
        mipmon_dates_pn, mipmon_mean_pn, mipmon_err_pn = get_mipmon_prev_bins(mipmon, mipmon_areas[i], mip_test_type, cross_dates, \
                                                                     time_width, time_shift, mask = mipmon_mask, \
                                                                     print_sizes = print_sizes)
        #All MiPMon
        if mip_mask is None:
            mipmon_mask = mipmon['visit'].notnull()
        else:
            mipmon_mask = mipmon['visit'].notnull()&mip_mask
        mipmon_dates, mipmon_mean, mipmon_err = get_mipmon_prev_bins(mipmon, mipmon_areas[i], mip_test_type, cross_dates, \
                                                                     time_width, time_shift, mask = mipmon_mask, \
                                                                     print_sizes = print_sizes)
        #MiPMon PG at PN
        if mip_mask is None:
            mipmon_mask = (mipmon['visit'] == 'PN')&(mipmon['gestnum'] == 1)
        else:
            mipmon_mask = (mipmon['visit'] == 'PN')&(mipmon['gestnum'] == 1)&mip_mask
        mipmon_dates_pn_pg, mipmon_mean_pn_pg, mipmon_err_pn_pg = get_mipmon_prev_bins(mipmon, mipmon_areas[i], mip_test_type, cross_dates, \
                                                                     time_width, time_shift, mask = mipmon_mask, \
                                                                     print_sizes = print_sizes)

        if 'pn' in show:
            plt.errorbar(cross_mean, mipmon_mean_pn, mipmon_err_pn, xerr = cross_err, label = 'PN ' + mipmon_areas[i][0], c = colors[i], \
                     linestyle = '', lw = 1, marker = 'o')
        if 'all' in show:
            plt.errorbar(cross_mean, mipmon_mean, mipmon_err, xerr = cross_err, label = 'All ' + mipmon_areas[i][0], c = colors[i], \
                     linestyle = '', lw = 1, marker = 's')
        if 'pg' in show:
            plt.errorbar(cross_mean, mipmon_mean_pn_pg, mipmon_err_pn_pg, xerr = cross_err, label = 'PG at PN ' + mipmon_areas[i][0], c = colors[i], \
                     linestyle = '', lw = 1, marker = '^')

        if lin_fit:
            all_cross = np.concatenate((all_cross, cross_mean))
            all_cross_err = np.concatenate((all_cross_err, cross_err))
            all_mip_pn = np.concatenate((all_mip_pn, mipmon_mean_pn))
            all_mip_pn_err = np.concatenate((all_mip_pn_err, mipmon_err_pn))
            all_mip = np.concatenate((all_mip, mipmon_mean))
            all_mip_err = np.concatenate((all_mip_err, mipmon_err))
            all_mip_pn_pg = np.concatenate((all_mip_pn_pg, mipmon_mean_pn_pg))
            all_mip_pn_pg_err = np.concatenate((all_mip_pn_pg_err, mipmon_err_pn_pg))
    if lin_fit:
        #Linear fits
        mask = all_cross**2. >= 0
        p_pn = np.polyfit(all_cross[mask], all_mip_pn[mask], 1)#, w = 1/(all_cross_err**2 + all_mip_pn_err**2))
        p_all = np.polyfit(all_cross[mask], all_mip[mask], 1)#, w = 1/(all_cross_err**2 + all_mip_err**2))
        p_pn_pg = np.polyfit(all_cross[mask], all_mip_pn_pg[mask], 1)#, w = 1/(all_cross_err**2 + all_mip_pn_pg_err**2))
        #Printing results
        if 'pn' in show:
            print("Cross vs Prenatal visits:")
            print("Linear parameter fitting: origin = " + str(p_pn[1]) + ", slope = " + str(p_pn[0]))
            pcorr_pn = sci_stats.pearsonr(all_cross[mask], all_mip_pn[mask])[0]
            print("Pearson CC: ", pcorr_pn)
            print()
        if 'all' in show:
            print("Cross vs all MiPMon:")
            print("Linear parameter fitting: origin = " + str(p_all[1]) + ", slope = " + str(p_all[0]))
            pcorr_all = sci_stats.pearsonr(all_cross[mask], all_mip[mask])[0]
            print("Pearson CC: ", pcorr_all)
            print()
        if 'pg' in show:
            print("Cross vs Primigravid at prenatal:")
            print("Linear parameter fitting: origin = " + str(p_pn_pg[1]) + ", slope = " + str(p_pn_pg[0]))
            pcorr_pn_pg = sci_stats.pearsonr(all_cross[mask], all_mip_pn_pg[mask])[0]
            print("Pearson CC: ", pcorr_pn_pg)
        #Plotting results
        all_vals = np.concatenate((all_cross[mask] + all_cross_err[mask], all_mip_pn[mask] + \
                                   all_mip_pn_err[mask], all_mip[mask] + all_mip_err[mask], all_mip_pn_pg[mask] + \
                                   all_mip_pn_pg_err[mask]))
        all_vals_min = np.concatenate((all_cross[mask] - all_cross_err[mask], all_mip_pn[mask] - \
                                       all_mip_pn_err[mask], all_mip[mask] - all_mip_err[mask], \
                                       all_mip_pn_pg[mask] - all_mip_pn_pg_err[mask]))
        xmin, xmax = np.min(all_vals)*.9, np.max(all_vals)*1.02
        xplot = np.array([xmin, xmax])
        #import pdb; pdb.set_trace()
        if 'pn' in show:
            plt.plot(xplot, p_pn[1] + p_pn[0]*xplot, lw = 2, color = 'k', linestyle = '-', label = 'PN')
        if 'all' in show:
            plt.plot(xplot, p_all[1] + p_all[0]*xplot, lw = 2, color = 'k', linestyle = '--', label = 'All')
        if 'pg' in show:
            plt.plot(xplot, p_pn_pg[1] + p_pn_pg[0]*xplot, lw = 2, color = 'k', linestyle = ':', label = 'PG at PN')
        plt.plot(xplot, xplot, color = 'tab:grey')

    plt.xlabel('Prevalence cross-sectional (2-10 years)')
    plt.ylabel('Prevalence pregnant women')
    plt.legend()
    plt.show()

def clinic_in_time_bins(clinic_data, time_bins, time_shift, clinic_val = 'cases'):
    """
    This method calculated the mean number of weekly clinical cases in time bins.

    Parameters:
    -----------
    clinic_data: pd.DataFrame
        Dataframe of clinical data
    time_bins: list of DateTime data
        Edges of time bins
    time_shift: int
        Time shift to apply backwards for clinical cases in number of days
    clinic_val: str {'cases', 'positivity'}
        Value to use for the statistics of clinical data

    Returns:
    --------
    clinic_bins: np.array
        Mean number of weekly cases per time bin
    clinic_err: np.array
        Error on the mean number of clinical cases
    """
    clinic_bins = []
    clinic_err = []
    for t in range(len(time_bins) - 1):
        mask = (clinic_data['date'] >= time_bins[t] - pd.to_timedelta(time_shift, unit = 'D'))&\
                                (clinic_data['date'] < time_bins[t + 1] - pd.to_timedelta(time_shift, unit = 'D'))
        #mean, err, mean_b = stats.bootstrap_mean_err(clinic_data['malaria'][mask])
        if clinic_val == 'cases':
            clinic_bins.append(np.mean(clinic_data['malaria'][mask]))
            clinic_err.append(np.std(clinic_data['malaria'][mask])/np.sqrt(np.sum(mask)))
        elif clinic_val == 'positivity':#Taking the mean of each weekly positivity
            clinic_bins.append(np.mean((clinic_data['malaria']/clinic_data['teastdone'])[mask]))
            clinic_err.append(np.std((clinic_data['malaria']/clinic_data['testdone'])[mask])/np.sqrt(np.sum(mask)))
        elif clinic_val == 'incidence':
            clinic_bins.append(np.mean((clinic_data['malaria']/clinic_data['incidence'])[mask]))
            clinic_err.append(np.std((clinic_data['malaria']/clinic_data['incidence'])[mask])/np.sqrt(np.sum(mask)))
    clinic_bins = np.array(clinic_bins)
    clinic_err = np.array(clinic_err)
    return clinic_bins, clinic_err

def clinic_mipmon_bins(mipmon, rrs, opd_2to9, mipmon_areas, clinic_areas, \
                        time_bins, norm = True, time_shift = 0, \
                        mipmon_mask = None, clinic_val = 'cases', mip_test_type = 'pcr', \
                      ret_resamples = False):
    """
    This methods calculates the mean prevalence of MiPMon data and the mean number
    of weekly clinical cases in different time bins.

    Parameters:
    -----------
    mipmon: pd.DataFrame
        DataFrame of MiPMon samples
    rrs: pd.DataFrame
        DataFrame of rapid reporting system
    opd_2to9: pd.DataFrame
        DataFrame of Manhica clinical cases
    mipmon_areas: list
        List of area names for MiPMon samples
    clinic_areas: list
        List of area names for clinical cases
    time_bins: int
        Number of bins for MiPMon samples
    norm: bool
        If True, the clinical cases are renormalised to that they have the same
        mean than the MiPMon prevalences
    time_shift: int
        Time shift to apply backwards for clinical cases in number of days
    mipmon_mask: np.array
        Mask to apply to the MiPMon samples
    clinic_val: str {'cases', 'positivity'}
        Value to use for the statistics of clinical data
    mip_test_type: str {'pcr', 'rdt'}
        It specifies the type of test for MiPMon
    ret_resamples: bool
        If True, the measurements of all the MiPMon resamples are returned

    Returns:
    --------
    dates: list
        Mean MiPMon date times per time bin
    mipmon_prev: np.array
        MiPMon prevalence per time bin
    mipmon_err: np.array
        Error on MiPMon prevalence per time bin
    clinic_bins: np.array
        Mean weekly clinical cases per time bin
    clinic_err: np.array
        Error on the mean clinical cases
    """
    #Define area masks
    mipmon_areas_mask = utils.get_mipmon_area_mask(mipmon, mipmon_areas)
    clinic_areas_mask = utils.get_clinic_area_mask(rrs, opd_2to9, clinic_areas)
    #Defines clinic data
    clinic_data = utils.clinic_df(clinic_areas, rrs, opd_2to9)
    if len(clinic_areas) > 1:
        clinic_data = clinic_data[clinic_areas_mask].groupby(by = 'date')[['malaria', 'testdone', 'incidence']].sum()
        clinic_data['date'] = clinic_data.index
    #Define clinic and mipmon mask
    if mipmon_mask is None:
        mipmon_mask = mipmon['pcrpos'].notnull()&mipmon_areas_mask
    else:
        mipmon_mask = mipmon['pcrpos'].notnull()&mipmon_areas_mask&mipmon_mask
    clinic_mask = clinic_data['date'].notnull()
    if len(clinic_areas) == 1:
        clinic_mask = clinic_mask&clinic_areas_mask
    #Define MiPMon time bin edges
    out, time_bins = pd.cut(mipmon['visdate'][mipmon_mask], time_bins, retbins = True)
    #MiPMon prevalence
    if mip_test_type == 'pcr':
        mip_pos = mipmon['pcrpos']
    elif mip_test_type == 'rdt':
        mip_pos = mipmon['density'] >= 100
    else:
        if mip_test_type in mipmon.columns:
            mip_pos = mipmon[mip_test_type]
        else:
            print("Wrong mipmon test type assignment: ", mip_test_type)
            print("Using pcrpos test instead")
            mip_pos = mipmon['pcrpos']
    out = stats.mean_prev_time_bins(mipmon['visdate'], \
                                    mip_pos, \
                                    data_mask = mipmon_mask&mip_pos.notnull(), \
                                    nbins = time_bins, \
                                    nrands = 1000, \
                                    ret_resamples = ret_resamples)
    if ret_resamples:
        dates, mipmon_prev, mipmon_err, mipmon_prev_r = out
    else:
        dates, mipmon_prev, mipmon_err = out
    #Clinic cases in time bins
    clinic_bins, clinic_err = clinic_in_time_bins(clinic_data[clinic_mask], \
                                                  time_bins, time_shift, \
                                                  clinic_val = clinic_val)
    dates = pd.to_datetime(dates) - pd.to_timedelta(time_shift, unit = 'D')
    if norm:
        mask = np.isfinite(mipmon_prev)&np.isfinite(clinic_bins)
        k = np.mean(mipmon_prev[mask])/np.mean(clinic_bins[mask])
        clinic_bins *= k
        clinic_err *= k
    if ret_resamples:
        return dates, mipmon_prev, mipmon_err, clinic_bins, clinic_err, mipmon_prev_r
    else:
        return dates, mipmon_prev, mipmon_err, clinic_bins, clinic_err

def show_scatter_clinic_mip(mipmon_areas, clinic_areas, mip_test_type, mipmon_bins, \
                            time_shift, lin_fit = True, show = ['pn', 'all', 'pg'], clinic_val = 'cases', colors = None):
    """
    This method shows the scatter comparison between MiPMon and data from clinics
    """
    if colors is None:
        colors = [cm.turbo((i+1)/float(len(mipmon_areas) + 1)) for i in range(len(mipmon_areas) + 1)]
    plt.figure(figsize = [9,6])
    if lin_fit:
        all_clinic = np.array([])
        all_mip_pn = np.array([])
        all_mip = np.array([])
        all_mip_pn_pg = np.array([])
        all_clinic_err = np.array([])
        all_mip_pn_err = np.array([])
        all_mip_err = np.array([])
        all_mip_pn_pg_err = np.array([])
    for i in range(len(mipmon_areas)):
        #MiPMon area mask
        mipmon_areas_mask = utils.get_mipmon_area_mask(mipmon, mipmon_areas[i])&mipmon['pcrpos'].notnull()
        #MiPMon at PN
        mipmon_mask = mipmon['visit'] == 'PN'
        mipmon_dates_pn, mipmon_mean_pn, mipmon_err_pn, clinic_bins, clinic_err = clinic_mipmon_bins(mipmon, rrs, \
                                                                                opd_2to9, mipmon_areas[i], \
                                                                             clinic_areas[i], mipmon_bins, \
                                                                             time_shift = time_shift, norm = False, \
                                                                             mipmon_mask = mipmon_mask, \
                                                                            clinic_val = clinic_val, \
                                                                                mip_test_type = mip_test_type)
        if 'pn' in show:
            plt.errorbar(clinic_bins, mipmon_mean_pn, mipmon_err_pn, xerr = clinic_err, label = 'PN ' + mipmon_areas[i][0], \
                     c = colors[i], linestyle = '', lw = 1, marker = 'o')
        #All MiPMon
        mipmon_mask = mipmon['visit'].notnull()
        mipmon_dates, mipmon_mean, mipmon_err, clinic_bins, clinic_err = clinic_mipmon_bins(mipmon, rrs, opd_2to9, mipmon_areas[i], \
                                                                             clinic_areas[i], mipmon_bins, \
                                                                             time_shift = time_shift, norm = False, \
                                                                             mipmon_mask = mipmon_mask, \
                                                                            clinic_val = clinic_val, \
                                                                            mip_test_type = mip_test_type)
        if 'all' in show:
            plt.errorbar(clinic_bins, mipmon_mean, mipmon_err, xerr = clinic_err, label = 'All ' + mipmon_areas[i][0], \
                         c = colors[i], linestyle = '', lw = 1, marker = 's')
        #MiPMon PG at PN
        mipmon_mask = (mipmon['visit'] == 'PN')&(mipmon['gestnum'] == 1)
        mipmon_dates, mipmon_mean_pn_pg, mipmon_err_pn_pg, clinic_bins, clinic_err = clinic_mipmon_bins(mipmon, rrs, opd_2to9, mipmon_areas[i], \
                                                                             clinic_areas[i], mipmon_bins, \
                                                                             time_shift = time_shift, norm = False, \
                                                                             mipmon_mask = mipmon_mask, \
                                                                            clinic_val = clinic_val, \
                                                                            mip_test_type = mip_test_type)
        if 'pg' in show:
            plt.errorbar(clinic_bins, mipmon_mean_pn_pg, mipmon_err_pn_pg, xerr = clinic_err, label = 'PG at PN ' + mipmon_areas[i][0], \
                         c = colors[i], linestyle = '', lw = 1, marker = '^')
        if lin_fit:
            all_clinic = np.concatenate((all_clinic, clinic_bins))
            all_clinic_err = np.concatenate((all_clinic_err, clinic_err))
            all_mip_pn = np.concatenate((all_mip_pn, mipmon_mean_pn))
            all_mip_pn_err = np.concatenate((all_mip_pn_err, mipmon_err_pn))
            all_mip = np.concatenate((all_mip, mipmon_mean))
            all_mip_err = np.concatenate((all_mip_err, mipmon_err))
            all_mip_pn_pg = np.concatenate((all_mip_pn_pg, mipmon_mean_pn_pg))
            all_mip_pn_pg_err = np.concatenate((all_mip_pn_pg_err, mipmon_err_pn_pg))
    if lin_fit:
        #Linear fits
        mask = all_clinic**2. >= 0
        p_pn = np.polyfit(all_clinic[mask], all_mip_pn[mask], 1)#, w = 1/(all_cross_err**2 + all_mip_pn_err**2))
        p_all = np.polyfit(all_clinic[mask], all_mip[mask], 1)#, w = 1/(all_cross_err**2 + all_mip_err**2))
        p_pn_pg = np.polyfit(all_clinic[mask], all_mip_pn_pg[mask], 1)#, w = 1/(all_cross_err**2 + all_mip_pn_pg_err**2))
        #Printing results
        if 'pn' in show:
            print("Cross vs Prenatal visits:")
            print("Linear parameter fitting: origin = " + str(p_pn[1]) + ", slope = " + str(p_pn[0]))
            pcorr_pn = sci_stats.pearsonr(all_clinic[mask], all_mip_pn[mask])[0]
            print("Pearson CC: ", pcorr_pn)
            print()
        if 'all' in show:
            print("Cross vs all MiPMon:")
            print("Linear parameter fitting: origin = " + str(p_all[1]) + ", slope = " + str(p_all[0]))
            pcorr_all = sci_stats.pearsonr(all_clinic[mask], all_mip[mask])[0]
            print("Pearson CC: ", pcorr_all)
            print()
        if 'pg' in show:
            print("Cross vs Primigravid at prenatal:")
            print("Linear parameter fitting: origin = " + str(p_pn_pg[1]) + ", slope = " + str(p_pn_pg[0]))
            pcorr_pn_pg = sci_stats.pearsonr(all_clinic[mask], all_mip_pn_pg[mask])[0]
            print("Pearson CC: ", pcorr_pn_pg)
        #Plotting results
        all_vals = np.concatenate((all_clinic[mask] + all_clinic_err[mask], all_mip_pn[mask] + \
                                   all_mip_pn_err[mask], all_mip[mask] + all_mip_err[mask], all_mip_pn_pg[mask] + \
                                   all_mip_pn_pg_err[mask]))
        all_vals_min = np.concatenate((all_clinic[mask] - all_clinic_err[mask], all_mip_pn[mask] - \
                                       all_mip_pn_err[mask], all_mip[mask] - all_mip_err[mask], \
                                       all_mip_pn_pg[mask] - all_mip_pn_pg_err[mask]))
        xmin, xmax = np.min(all_vals)*.9, np.max(all_vals)*1.02
        xplot = np.array([xmin, xmax])
        #import pdb; pdb.set_trace()
        if 'pn' in show:
            plt.plot(xplot, p_pn[1] + p_pn[0]*xplot, lw = 2, color = 'k', linestyle = '-', label = 'PN')
        if 'all' in show:
            plt.plot(xplot, p_all[1] + p_all[0]*xplot, lw = 2, color = 'k', linestyle = '--', label = 'All')
        if 'pg' in show:
            plt.plot(xplot, p_pn_pg[1] + p_pn_pg[0]*xplot, lw = 2, color = 'k', linestyle = ':', label = 'PG at PN')
    if clinic_val == 'cases':
        plt.xlabel('Mean number of weekly clinical cases')
    elif clinic_val == 'positivity':
        plt.xlabel('Positive rate of clinical cases')
    plt.ylabel('Prevalence pregnant women')
    plt.legend()
    plt.show()

def show_scatter_clinic_cross(cross210, rrs, opd_2to9, cross_areas, clinic_areas, cross_test_type, \
                              cross_bins, clinic_bin_edges = None, time_width = 30, lin_fit = True, clinic_val = 'cases', \
                              colors = None, cross_mask = None, show = True):
    """
    This method shows the scatter comparison between Cross-sectionals and data from clinics
    """
    if colors is None:
        colors = [cm.turbo((i+1)/float(len(cross_areas) + 1)) for i in range(len(cross_areas) + 1)]
    #plt.figure(figsize = [9,6])
    if lin_fit:
        all_clinic = np.array([])
        all_cross = np.array([])
        all_clinic_err = np.array([])
        all_cross_err = np.array([])
    for i in range(len(cross_areas)):
        #Cross area mask
        #cross_areas_mask = utils.get_cross_area_mask(cross210, cross_areas[i])
        #Cross
        cross_dates, cross_prev, cross_err, clinic_bins, clinic_err = clinic_cross_bins(cross210, rrs, \
                                        opd_2to9, cross_areas[i], clinic_areas[i], \
                                       cross_bins, clinic_bin_edges = clinic_bin_edges, norm = False, \
                                       time_width = time_width, cross_mask = cross_mask, \
                                       test_name = cross_test_type, clinic_val = clinic_val)

        plt.errorbar(clinic_bins, cross_prev, cross_err, xerr = clinic_err, label = cross_areas[i][0], \
                     c = colors[i], linestyle = '', lw = 1, marker = 'o')
        if lin_fit:
            all_clinic = np.concatenate((all_clinic, clinic_bins))
            all_clinic_err = np.concatenate((all_clinic_err, clinic_err))
            all_cross = np.concatenate((all_cross, cross_prev))
            all_cross_err = np.concatenate((all_cross_err, cross_err))
    if lin_fit:
        #Linear fits
        mask = all_clinic**2. >= 0
        p = np.polyfit(all_clinic[mask], all_cross[mask], 1)
        #Printing results
        print("Cross vs clinical cases:")
        print("Linear parameter fitting: origin = " + str(p[1]) + ", slope = " + str(p[0]))
        pcorr_pn = sci_stats.pearsonr(all_clinic[mask], all_cross[mask])[0]
        print("Pearson CC: ", pcorr_pn)
        print()
        #Plotting results
        all_vals = np.concatenate((all_clinic[mask] + all_clinic_err[mask], all_cross[mask] + \
                                   all_cross_err[mask]))
        all_vals_min = np.concatenate((all_clinic[mask] - all_clinic_err[mask], all_cross[mask] - \
                                       all_cross_err[mask]))
        xmin, xmax = np.min(all_vals)*.9, np.max(all_vals)*1.02
        xplot = np.array([xmin, xmax])
        plt.plot(xplot, p[1] + p[0]*xplot, lw = 2, color = 'k', linestyle = '-', label = 'Cross vs clinic')
    if clinic_val == 'cases':
        plt.xlabel('Mean number of weekly clinical cases')
    elif clinic_val == 'positivity':
        plt.xlabel('Positive rate of clinical cases')
    elif clinic_val == 'incidence':
        plt.xlabel('incidence of clinical cases')
        plt.ylim((0, np.max(all_cross[mask] + all_cross_err[mask])))
        plt.xlim((0, np.max(all_clinic[mask] + all_clinic_err[mask])))
    if cross_test_type == 'pospcr':
            cross_test_type = 'pcr'
    plt.ylabel(r'$\rm{PR}_{\rm{'+cross_test_type.upper()+'}}$ cross-sectional (2-10 years)')
    plt.legend()
    if show:
        plt.show()

def clinic_cross_bins(cross, rrs, opd_2to9, cross_areas, clinic_areas, time_bins, clinic_bin_edges = None, norm = True, \
                       time_width = 30, cross_mask = None, test_name = 'pospcr', clinic_val = 'cases'):
    """
    This methods calculates the mean prevalence of Cross-sectional surveys and the mean number
    of weekly clinical cases in different time bins.

    Parameters:
    -----------
    cross: pd.DataFrame
        DataFrame of Cross-sectional samples
    rrs: pd.DataFrame
        DataFrame of rapid reporting system
    opd_2to9: pd.DataFrame
        DataFrame of Manhica clinical cases
    cross_areas: list
        List of area names for Cross-sectional samples
    clinic_areas: list
        List of area names for clinical cases
    time_bins: int
        Number of bins for Cross-sectional samples
    clinic_bin_edges: list
        List of time bin edges for clinical data
    norm: bool
        If True, the clinical cases are renormalised to that they have the same
        mean than the MiPMon prevalences
    time_shift: int
        Time shift to apply backwards for clinical cases in number of days
    cross_mask: np.array
        Mask to apply to the Cross-sectional samples
    test_name: str
        Name of malaria test to use
    clinic_val: str {'cases', 'positivity'}
        Value to use for the statistics of clinical data

    Returns:
    --------
    dates: list
        Mean Cross-sectional date times per time bin
    cross_prev: np.array
        Cross-sectional prevalence per time bin
    cross_err: np.array
        Error on MiPMon prevalence per time bin
    clinic_bins: np.array
        Mean weekly clinical cases per time bin
    clinic_err: np.array
        Error on the mean clinical cases
    """
    #Define area masks
    cross_areas_mask = utils.get_cross_area_mask(cross, cross_areas)
    clinic_areas_mask = utils.get_clinic_area_mask(rrs, opd_2to9, clinic_areas)
    #Defines clinic data
    clinic_data = utils.clinic_df(clinic_areas, rrs, opd_2to9)
    if len(clinic_areas) > 1:
        clinic_data = clinic_data[clinic_areas_mask].groupby(by = 'date')[['malaria', 'testdone', 'incidence']].sum()
        clinic_data['date'] = clinic_data.index
    #Define clinic and Cross mask
    if cross_mask is None:
        cross_mask = cross[test_name].notnull()&cross_areas_mask
    else:
        cross_mask = cross[test_name].notnull()&cross_areas_mask&cross_mask
    clinic_mask = clinic_data['date'].notnull()
    if len(clinic_areas) == 1:
        clinic_mask = clinic_mask&clinic_areas_mask
    #Cross prevalence
    dates, cross_prev, cross_err = stats.mean_prev_time_bins(cross['visdate'], cross[test_name], data_mask = cross_mask, nbins = time_bins, nrands = 1000)
    #Clinic cases in time bins
    if clinic_bin_edges is None:
        clinic_bins, clinic_err = clinic_in_times(clinic_data[clinic_mask], dates, time_width, \
                                                  clinic_val = clinic_val)
    else:
        if clinic_val == 'cases':
            clinic_vals = clinic_data['malaria']
        elif clinic_val == 'positivity':#Taking the mean of each weekly positivity
            clinic_vals = (clinic_data['malaria']/clinic_data['testdone'])
        elif clinic_val == 'incidence':
            clinic_vals = clinic_data['malaria']/clinic_data['incidence']
        dates_clinic, clinic_bins, clinic_err = stats.mean_prev_time_bins(clinic_data['date'], clinic_vals, data_mask = clinic_mask, nbins = clinic_bin_edges, nrands = 1000)
    if norm:
        mask = np.isfinite(clinic_bins)
        k = np.mean(cross_prev[mask])/np.mean(clinic_bins[mask])
        clinic_bins *= k
        clinic_err *= k
    return dates, cross_prev, cross_err, clinic_bins, clinic_err

def clinic_in_times(clinic_data, times, time_width, clinic_val = 'cases'):
    """
    This method calculated the mean number of weekly clinical cases at some times.

    Parameters:
    -----------
    clinic_data: pd.DataFrame
        Dataframe of clinical data
    times: list of DateTime data
        Dates around which clinical cases are averaged
    time_width: int
        Time width to apply backwards and forward for clinical cases in number of days
    clinic_val: str {'cases', 'positivity'}
        Value to use for the statistics of clinical data

    Returns:
    --------
    clinic_bins: np.array
        Mean number of weekly cases per time bin
    clinic_err: np.array
        Error on the mean number of clinical cases
    """
    clinic_bins = []
    clinic_err = []
    for t in times:
        mask = (clinic_data['date'] >= t - pd.to_timedelta(time_width, unit = 'D'))&\
                                (clinic_data['date'] < t + pd.to_timedelta(time_width, unit = 'D'))
        #mean, err, mean_b = stats.bootstrap_mean_err(clinic_data['malaria'][mask])
        if clinic_val == 'cases':
            clinic_bins.append(np.mean(clinic_data['malaria'][mask]))
            clinic_err.append(np.std(clinic_data['malaria'][mask])/np.sqrt(np.sum(mask)))
        elif clinic_val == 'positivity':#Taking the mean of each weekly positivity
            mask = mask&(clinic_data['testdone'] > 0)
            clinic_bins.append(np.mean((clinic_data['malaria']/clinic_data['testdone'])[mask]))
            clinic_err.append(np.std((clinic_data['malaria']/clinic_data['testdone'])[mask])/np.sqrt(np.sum(mask)))
        elif clinic_val == 'incidence':
            mask = mask&(clinic_data['incidence'] > 0)
            clinic_bins.append(np.mean((clinic_data['malaria']/clinic_data['incidence'])[mask]))
            clinic_err.append(np.std((clinic_data['malaria']/clinic_data['incidence'])[mask])/np.sqrt(np.sum(mask)))
    clinic_bins = np.array(clinic_bins)
    clinic_err = np.array(clinic_err)
    return clinic_bins, clinic_err

def time_comparison_plot(cross210, mipmon, rrs, opd_2to9, cross_areas, mipmon_areas, clinic_areas, \
                         cross_bins, mipmon_bins, ylim = None, n_factor = [1,1], area_name = None, \
                         time_shift = 0, show = ['all'], smooth_size = 15, min_date = '2017-03', \
                         ylabel = 'Prevalence/Positivity/cases', figsize = [10,6], \
                         show_plot = True, title = 'Cross vs MiPMon prevalence in '):
    """This method show a plot comparing the time evolution of the burden  between
    cross-sectionals, pregnant women and clinical cases for a given area.

    Parameters:
    -----------
    cross210: pd.DataFrame
        Dataframe of cross-sectionals
    mipmon: pd.DataFrame
        Dataframe of MiPMon samples
    rrs: pd.DataFrame
        Dataframe of RRS samples
    opd_2to9: pd.DataFrame
        Dataframe of OPD samples
    cross_areas: list
        List of area names for cross-sectionals
    mipmon_areas: list
        List of area names for MiPMon samples
    clinic_areas: list
        List of area names for clinical cases
    cross_bins: list
        Date bins for cross-sectional survey data
    mipmon_bins: int
        Number of bins for MiPMon samples
    ylim: list
        Limits of y-axis shown in the plot
    n_factor: [float, float]
        Factor dividing the number of clinical [cases, positivity] for comparison
    area_name: str
        Name of area to show in the title
    time_shift: int
        Number of days to shift clinical cases
    show: list
        A list specifying the what data to show
    smooth_size: int
        Size of smoothing window function for clinical data
    min_date: str
        Earliest date to be shown in the plot
    ylabel: str
        Y label in plot
    figsize: list of length 2 or None
        It defines the size of the plot
    show_plot: bool
        It specifies if the plot is closed and shown
    title: str
        Title to show in the plot. The area name will be added after

    Returns:
    --------
    Plot comparing the different samples
        """

    #Define area masks
    cross_areas_mask, mipmon_areas_mask, clinic_areas_mask = utils.get_area_masks(cross210, mipmon, rrs, opd_2to9, \
                                                                        cross_areas, mipmon_areas, clinic_areas)
    clinic_data = utils.clinic_df(clinic_areas, rrs, opd_2to9)
    if len(clinic_areas) > 1:
        clinic_data = clinic_data[clinic_areas_mask].groupby(by = 'date')[['malaria', 'testdone', 'incidence']].sum()
        clinic_data['date'] = clinic_data.index

    if figsize is not None:
        plt.figure(figsize=figsize)
    min_date = pd.to_datetime(min_date)
    #MiPMon all PCR
    if any([i in ['all', 'mipmon', 'mipmon_pcr'] for i in show]):
        mask = mipmon['pcrpos'].notnull()&mipmon_areas_mask
        dates, mean, err = stats.mean_prev_time_bins(mipmon['visdate'], mipmon['pcrpos'], data_mask = mask, nbins = mipmon_bins, nrands = 1000)
        plt.errorbar(pd.to_datetime(dates) - pd.to_timedelta(time_shift, unit='D'), mean, err, label = 'MiPMon', c = 'tab:green', lw = 3)
        min_date = min(min_date, dates[0])

    #MiPMon all RDT-like
    if any([i in ['all', 'mipmon', 'mipmon_rdt'] for i in show]):
        mask = mipmon['pcrpos'].notnull()&mipmon_areas_mask&(mipmon['visit'] == 'PN')
        dates, mean, err = stats.mean_prev_time_bins(mipmon['visdate'], mipmon['density']>=100, data_mask = mask, nbins = mipmon_bins, nrands = 1000)
        plt.errorbar(pd.to_datetime(dates) - pd.to_timedelta(time_shift, unit='D'), mean, err, label = 'MiPMon d > 100', c = 'tab:purple', lw = 3, linestyle = ':')
        min_date = min(min_date, dates[0])

    #MiPMon Primigravid RDT-like
    if any([i in ['all', 'mipmon', 'mipmon_rdt_pg'] for i in show]):
        mask = mipmon['pcrpos'].notnull()&mipmon_areas_mask&(mipmon['visit'] == 'PN')&(mipmon['gestnum'] == 1)
        dates, mean, err = stats.mean_prev_time_bins(mipmon['visdate'], mipmon['density']>=100, data_mask = mask, nbins = mipmon_bins, nrands = 1000)
        plt.errorbar(pd.to_datetime(dates) - pd.to_timedelta(time_shift, unit='D'), mean, err, label = 'PG d > 100', c = 'm', lw = 3, linestyle = ':')
        min_date = min(min_date, dates[0])

    #MiPMon at prenatal
    if any([i in ['all', 'mipmon', 'PN'] for i in show]):
        mask = mipmon['pcrpos'].notnull()&mipmon_areas_mask&(mipmon['visit'] == 'PN')
        dates, mean, err = stats.mean_prev_time_bins(mipmon['visdate'], mipmon['pcrpos'], data_mask = mask, nbins = mipmon_bins, nrands = 1000)
        plt.errorbar(pd.to_datetime(dates) - pd.to_timedelta(time_shift, unit='D'), mean, err, label = 'MiPMon prenatal', c = 'tab:blue', lw = 3, linestyle = '--')
        min_date = min(min_date, dates[0])

    #MiPMon Primigravid at prenatal
    if any([i in ['all', 'mipmon', 'mipmon_pg'] for i in show]):
        mask = mipmon['pcrpos'].notnull()&mipmon_areas_mask&(mipmon['visit'] == 'PN')&(mipmon['gestnum'] == 1)
        dates, mean, err = stats.mean_prev_time_bins(mipmon['visdate'], mipmon['pcrpos'], data_mask = mask, nbins = mipmon_bins, nrands = 1000)
        plt.errorbar(pd.to_datetime(dates) - pd.to_timedelta(time_shift, unit='D'), mean, err, label = 'MiPMon primigravid', c = 'tab:cyan', lw = 3, linestyle = '-.')
        min_date = min(min_date, dates[0])

    #MiPMon at delivery
    if any([i in ['all', 'mipmon', 'MA'] for i in show]):
        mask = mipmon['pcrpos'].notnull()&mipmon_areas_mask&(mipmon['visit'] == 'MA')
        dates, mean, err = stats.mean_prev_time_bins(mipmon['visdate'], mipmon['pcrpos'], data_mask = mask, nbins = mipmon_bins, nrands = 1000)
        plt.errorbar(pd.to_datetime(dates) - pd.to_timedelta(time_shift, unit='D'), mean, err, label = 'MiPMon delivery', c = 'tab:pink', lw = 3, linestyle = '--')
        min_date = min(min_date, dates[0])

    #Cross PCR
    if any([i in ['all', 'cross', 'cross_pcr'] for i in show]):
        mask = cross210['pospcr'].notnull()&cross_areas_mask
        dates, mean, err = stats.mean_prev_time_bins(cross210['visdate'], cross210['pospcr'], data_mask = mask, nbins = cross_bins, nrands = 1000, weights = cross210['weight'])
        plt.errorbar(dates, mean, err, label = 'Cross PCR', lw = 3, c = 'tab:orange', linestyle = '', marker = 'o')
        min_date = min(min_date, dates[0])

    #Cross RDT
    if any([i in ['all', 'cross', 'cross_rdt'] for i in show]):
        mask = cross210['rdt'].notnull()&cross_areas_mask
        dates, mean, err = stats.mean_prev_time_bins(cross210['visdate'], cross210['rdt'], data_mask = mask, nbins = cross_bins, nrands = 1000, weights = cross210['weight'])
        plt.errorbar(dates, mean, err, label = 'Cross RDT', lw = 3, c = 'tab:red', linestyle = '', marker = 'o')
        min_date = min(min_date, dates[0])

    #Clinical cases
    if any([i in ['all', 'clinic', 'clinic_cases', 'clinic_pos'] for i in show]):
        mask = (clinic_data['date']>min_date - pd.to_timedelta(60, unit = 'D'))
        if len(clinic_areas) == 1:
            mask = mask&clinic_areas_mask
        if any([i in ['all', 'clinic', 'clinic_cases'] for i in show]):
            plt.plot(clinic_data['date'][mask], \
                     smooth(clinic_data['malaria'][mask], size = smooth_size)/n_factor[0], \
                     label = 'Clinic cases /' + str(n_factor[0]), c = 'k', lw = 3)
        if any([i in ['all', 'clinic', 'clinic_pos'] for i in show]):
            plt.plot(clinic_data['date'][mask], \
                     (smooth(clinic_data['malaria'], size = smooth_size)/smooth(clinic_data['testdone'], \
                            size = smooth_size))[mask]/n_factor[1], label = 'Clinic positivity /' + \
                     str(n_factor[1]), c = [.5,.5,.5], lw = 3)
        if any([i in ['all', 'clinic', 'clinic_inc'] for i in show]):
            plt.plot(clinic_data['date'][mask], \
                     (smooth(clinic_data['malaria'], size = smooth_size)/smooth(clinic_data['incidence'], \
                            size = smooth_size))[mask]/n_factor[1], label = 'Clinic incidence /' + \
                     str(n_factor[1]), c = [.5,.5,.5], lw = 3)
    if area_name is None:
        area_name = cross_areas[0]
    plt.title(title + area_name)
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel(ylabel)
    plt.ylim(ylim)
    if show_plot:
        plt.show()


def show_chi2_k_vs_dtime(mipmon, rrs, opd_2to9, mipmon_areas, clinic_areas, mipmon_bins, dtimes, \
                         show_k = True, show_mean = True, show = ['pn', 'pg'], clinic_val = 'cases'):
    """
    This method shows the Chi2 between MiPMon and clinical data
    as a function of their time shift.
    """
    colors = [cm.turbo((i + 1)/float(len(mipmon_areas) + 1)) for i in range(len(mipmon_areas) + 1)]
    plt.figure(0, figsize = [9,6])
    plt.grid()
    if show_k:
        plt.figure(1, figsize = [9,6])
    if show_mean:
        mean_chi_pn = np.zeros(len(dtimes))
        mean_chi_all = np.zeros(len(dtimes))
        mean_chi_pn_pg = np.zeros(len(dtimes))
    for i in range(len(mipmon_areas)):
        chi_pn = []
        chi_all = []
        chi_pn_pg = []
        k_pn = []
        k_all = []
        k_pn_pg = []
        for dt in dtimes:
            #MiPMon at PN
            if 'pn' in show:
                mipmon_mask = mipmon['visit'] == 'PN'
                dates, mipmon_prev, mipmon_err, clinic_bins, clinic_err = clinic_mipmon_bins(mipmon, rrs, opd_2to9, mipmon_areas[i], \
                                                                                 clinic_areas[i], mipmon_bins[i], \
                                                                                 time_shift = dt, norm = True, \
                                                                                 mipmon_mask = mipmon_mask, clinic_val = clinic_val)
                chi2, k = get_clinic_amplitude_chi2(mipmon_prev, mipmon_err, clinic_bins, clinic_err)
                chi_pn.append(chi2)
                k_pn.append(k)
            #All MiPMon
            if 'all' in show:
                mipmon_mask = mipmon['visit'].notnull()
                dates, mipmon_prev, mipmon_err, clinic_bins, clinic_err = clinic_mipmon_bins(mipmon, rrs, opd_2to9, mipmon_areas[i], \
                                                                                 clinic_areas[i], mipmon_bins[i], \
                                                                                 time_shift = dt, norm = True, \
                                                                                 mipmon_mask = mipmon_mask, clinic_val = clinic_val)
                chi2, k = get_clinic_amplitude_chi2(mipmon_prev, mipmon_err, clinic_bins, clinic_err)
                chi_all.append(chi2)
                k_all.append(k)
            #MiPMon PG at PN
            if 'pg' in show:
                mipmon_mask = (mipmon['visit'] == 'PN')&(mipmon['gestnum'] == 1)
                dates, mipmon_prev, mipmon_err, clinic_bins, clinic_err = clinic_mipmon_bins(mipmon, rrs, opd_2to9, mipmon_areas[i], \
                                                                                 clinic_areas[i], mipmon_bins[i], \
                                                                                 time_shift = dt, norm = True, \
                                                                                 mipmon_mask = mipmon_mask, clinic_val = clinic_val)
                chi2, k = get_clinic_amplitude_chi2(mipmon_prev, mipmon_err, clinic_bins, clinic_err)
                chi_pn_pg.append(chi2)
                k_pn_pg.append(k)

        if show_mean:
            if 'pn' in show:
                mean_chi_pn = mean_chi_pn + np.array(chi_pn)/len(mipmon_areas)
            if 'all' in show:
                mean_chi_all = mean_chi_all + np.array(chi_all)/len(mipmon_areas)
            if 'pg' in show:
                mean_chi_pn_pg = mean_chi_pn_pg + np.array(chi_pn_pg)/len(mipmon_areas)

        plt.figure(0)
        if 'pn' in show:
            plt.plot(dtimes, chi_pn, label = 'PN ' + mipmon_areas[i][0], c = colors[i], linestyle = '-', lw = 3)
        if 'all' in show:
            plt.plot(dtimes, chi_all, label = 'All ' + mipmon_areas[i][0], c = colors[i], linestyle = '--', lw = 3)
        if 'pg' in show:
            plt.plot(dtimes, chi_pn_pg, label = 'PG at PN ' + mipmon_areas[i][0], c = colors[i], linestyle = ':', lw = 3)
        if show_k:
            plt.figure(1)
            if 'pn' in show:
                plt.plot(dtimes, k_pn, label = 'PN ' + mipmon_areas[i][0], c = colors[i], linestyle = '-', lw = 3)
            if 'all' in show:
                plt.plot(dtimes, k_all, label = 'All ' + mipmon_areas[i][0], c = colors[i], linestyle = '--', lw = 3)
            if 'pg' in show:
                plt.plot(dtimes, k_pn_pg, label = 'PG at PN ' + mipmon_areas[i][0], c = colors[i], linestyle = ':', lw = 3)

    if show_mean:
        if 'pn' in show:
            plt.plot(dtimes, mean_chi_pn, label = 'PN in all areas', c = 'k', linestyle = '-', lw = 3)
        if 'all' in show:
            plt.plot(dtimes, mean_chi_all, label = 'All in all areas', c = 'k', linestyle = '--', lw = 3)
        if 'pg' in show:
            plt.plot(dtimes, mean_chi_pn_pg, label = 'PG at PN in all areas', c = 'k', linestyle = ':', lw = 3)
    plt.figure(0)
    plt.xlabel('Time shift (days)')
    plt.ylabel(r'$\chi^2$')
    plt.legend()
    if show_k:
        plt.figure(1)
        plt.xlabel('Time shift (days)')
        plt.ylabel(r'$K$')
        plt.legend()

def clinic_in_time_bins(clinic_data, time_bins, time_shift, clinic_val = 'cases'):
    """
    This method calculated the mean number of weekly clinical cases in time bins.

    Parameters:
    -----------
    clinic_data: pd.DataFrame
        Dataframe of clinical data
    time_bins: list of DateTime data
        Edges of time bins
    time_shift: int
        Time shift to apply backwards for clinical cases in number of days
    clinic_val: str {'cases', 'positivity'}
        Value to use for the statistics of clinical data

    Returns:
    --------
    clinic_bins: np.array
        Mean number of weekly cases per time bin
    clinic_err: np.array
        Error on the mean number of clinical cases
    """
    clinic_bins = []
    clinic_err = []
    for t in range(len(time_bins) - 1):
        mask = (clinic_data['date'] >= time_bins[t] - pd.to_timedelta(time_shift, unit = 'D'))&\
                                (clinic_data['date'] < time_bins[t + 1] - pd.to_timedelta(time_shift, unit = 'D'))
        #mean, err, mean_b = stats.bootstrap_mean_err(clinic_data['malaria'][mask])
        if clinic_val == 'cases':
            clinic_bins.append(np.mean(clinic_data['malaria'][mask]))
            clinic_err.append(np.std(clinic_data['malaria'][mask])/np.sqrt(np.sum(mask)))
        elif clinic_val == 'positivity':#Taking the mean of each weekly positivity
            clinic_bins.append(np.mean((clinic_data['malaria']/clinic_data['testdone'])[mask]))
            clinic_err.append(np.std((clinic_data['malaria']/clinic_data['testdone'])[mask])/np.sqrt(np.sum(mask)))
        elif clinic_val == 'incidence':
            clinic_bins.append(np.mean((clinic_data['malaria']/clinic_data['incidence'])[mask]))
            clinic_err.append(np.std((clinic_data['malaria']/clinic_data['incidence'])[mask])/np.sqrt(np.sum(mask)))
    clinic_bins = np.array(clinic_bins)
    clinic_err = np.array(clinic_err)
    return clinic_bins, clinic_err


def get_clinic_amplitude_chi2(mipmon_prev, mipmon_err, clinic_bins, clinic_err, get_pcc = False, \
                             mipmon_prev_r = None):
    """
    This method estimates the amplitude normalisation factor of the clinical
    cases that minimises the chi2 with MiPMon data.

    Parameters:
    -----------
    mipmon_prev: np.array
        MiPMon prevalence per time bin
    mipmon_err: np.array
        Error on MiPMon prevalence per time bin
    clinic_bins: np.array
        Mean weekly clinical cases per time bin
    clinic_err: np.array
        Error on the mean clinical cases
    get_pcc: bool
        If True it returns the Pearson correlation coefficient of the
        renormalised data
    mipmon_prev_r: np.ndarray
        MiPMon prevalence per time bin for each bootstrap resample

    Returns:
    --------
    chi2: float
        Chi2 deviation between MiPMon and clinic with k factor
    k: float
        The optimal amplitude factor for clinical cases
    pcorr: float
        The Pearson correlation coefficient between the renormalised
        estimates
    """
    k, out = optimization.leastsq(residual, 1., args=(mipmon_prev, mipmon_err, clinic_bins, clinic_err))
    k = np.abs(k)
    chi2 = chi_square(mipmon_prev, mipmon_err, k*clinic_bins, k*clinic_err)
    if get_pcc:
        if len(mipmon_prev) < 2:
            print("Pearson CC cannot be computed with data of only "+\
                  str(len(mipmon_prev)) + " elements")
            pcorr = np.nan
        else:
            pcorr = sci_stats.pearsonr(mipmon_prev, k*clinic_bins)[0]
        if mipmon_prev_r is not None:
            pcorr_r = []
            nrands = mipmon_prev_r.shape[1]
            for r in range(nrands):
                k_r, out = optimization.leastsq(residual, 1., args=(mipmon_prev_r[:,r], mipmon_err, clinic_bins, clinic_err))
                pcorr_r.append(sci_stats.pearsonr(mipmon_prev_r[:,r], k_r*clinic_bins)[0])
            pcorr_r = np.array(pcorr_r)
            sorted_pcorrs = np.sort(pcorr_r)
            conf_95 = [sorted_pcorrs[int(nrands*.025)], sorted_pcorrs[int(nrands*.975)]]
            conf_68 = [sorted_pcorrs[int(nrands*.16)], sorted_pcorrs[int(nrands*.84)]]
            return chi2, k, pcorr, conf_68, conf_95
        else:
            return chi2, k, pcorr
    else:
        return chi2, k


def str_replace(text, ch1, ch2):
    new_text = ''
    for i in text:
        if i == ch1:
            new_text += ch2
        else:
            new_text += i
    return new_text

def residual(k, var_1, err_1, var_2, err_2):
    """
    This method returns the residual of the chi2 between
    two variables and their errors, where the second variable
    have been multiplied by a non-negative constant k.

    Parameters:
    -----------
    k: float
        Amplitude factor of second variable
    var_1: np.array
        First variable
    err_1: np.array
        Error of first variable
    var_2: np.array
        Second variable
    err_2: np.array
        Error of second variable

    Returns:
    --------
    chi2: float
        The residual chi2
    """
    mask = np.isfinite(var_1)&np.isfinite(var_2)
    mask = mask&np.isfinite(err_1)&np.isfinite(err_2)
    chi2 = np.mean((var_1[mask] - np.abs(k)*var_2[mask])**2. /(err_1[mask]**2. + np.abs(k)*err_2[mask]**2))
    return chi2

def get_mipmon_cross_chi2(cross210, mipmon, cross_areas, mipmon_areas, test_type, cross_bins, time_width, time_shift, mipmon_mask = None, print_sizes = True):
    """
    This method calculates the chi2 between Cross-sectional and MiPMon prevalences overall
    the overlapping crossectionals.

    Parameters:
    -----------
    cross210: pd.DataFrame
        Data frame of cross-sectional
    mipmon: pd.DataFrame
        Data frame for MiPMon samples
    cross_areas: list
        List of areas to include from the cross-sectional data
    mipmon_areas: list
        List of areas to incluse from MiPMon samples
    test_type: str {'pcr', 'rdt'}
        Type of test information used
    cross_bins: list
        Edges of time bins
    time_width: int
        Time width of MiPMon time bins to use (in days)
    time_shift: int
        Number of days to shift MiPMon cases
    mipmon_mask: np.array
        Boolean mask to select a MiPMon subsample
    print_sizes: bool
        It specifies whether the sample sizes per bin are shown (default True)

    Returns:
    --------
    cross_dates: list
        Mean dates of data per time bin
    cross_mean: np.array
        Mean prevalence per time bin
    cross_err: np.array
        Error of mean prevalence per time bin
    mipmon_dates: list
        Mean dates of data per time bin
    mipmon_mean: np.array
        Mean prevalence per time bin
    mipmon_err: np.array
        Error of mean prevalence per time bin
    chi2: float
        Chi2 between MiPMon and cross-sectionals
    """
    cross_dates, cross_mean, cross_err = get_cross_prev_bins(cross210, cross_areas, test_type, cross_bins, print_sizes = print_sizes)
    mipmon_dates, mipmon_mean, mipmon_err = get_mipmon_prev_bins(mipmon, mipmon_areas, test_type, cross_dates, time_width, time_shift, mask = mipmon_mask, print_sizes = print_sizes)
    chi2 = chi_square(cross_mean, cross_err, mipmon_mean, mipmon_err, use_err = True)
    return cross_dates, cross_mean, cross_err, mipmon_dates, mipmon_mean, mipmon_err, chi2

def show_chi2_cross_mip_vs_dtime(mipmon_areas, cross_areas, test_type, cross_bins, \
                                 time_width, dtimes, print_sizes = True, show_mean = True, \
                                show = ['pn', 'pg']):
    """
    This method shows the Chi2 between MiPMon and clinical data
    as a function of their time shift.
    """
    plt.figure(0, figsize = [9,6])
    plt.grid()
    colors = [cm.turbo((i+1)/float(len(mipmon_areas) + 1)) for i in range(len(mipmon_areas) + 1)]
    if show_mean:
        mean_chi_pn = np.zeros(len(dtimes))
        mean_chi_all = np.zeros(len(dtimes))
        mean_chi_pn_pg = np.zeros(len(dtimes))
    for i in range(len(mipmon_areas)):
        chi_pn = []
        chi_all = []
        chi_pn_pg = []
        for dt in dtimes:
            #MiPMon at PN
            if 'pn' in show:
                mipmon_mask = mipmon['visit'] == 'PN'
                cross_dates, cross_mean, cross_err, mipmon_dates, mipmon_mean, mipmon_err, chi2 = get_mipmon_cross_chi2(cross210, \
                                                                                mipmon, cross_areas[i], mipmon_areas[i], \
                                                                                test_type, cross_bins, time_width, \
                                                                                dt, mipmon_mask = mipmon_mask, \
                                                                                print_sizes = print_sizes)

                chi_pn.append(chi2)
            #All MiPMon
            if 'all' in show:
                mipmon_mask = mipmon['visit'].notnull()
                cross_dates, cross_mean, cross_err, mipmon_dates, mipmon_mean, mipmon_err, chi2 = get_mipmon_cross_chi2(cross210, \
                                                                                mipmon, cross_areas[i], mipmon_areas[i], \
                                                                                test_type, cross_bins, time_width, \
                                                                                dt, mipmon_mask = mipmon_mask, \
                                                                                print_sizes = print_sizes)
                chi_all.append(chi2)
            #MiPMon PG at PN
            if 'pg' in show:
                mipmon_mask = (mipmon['visit'] == 'PN')&(mipmon['gestnum'] == 1)
                cross_dates, cross_mean, cross_err, mipmon_dates, mipmon_mean, mipmon_err, chi2 = get_mipmon_cross_chi2(cross210, \
                                                                                mipmon, cross_areas[i], mipmon_areas[i], \
                                                                                test_type, cross_bins, time_width, \
                                                                                dt, mipmon_mask = mipmon_mask, \
                                                                                print_sizes = print_sizes)
                chi_pn_pg.append(chi2)

        if show_mean:
            if 'pn' in show:
                mean_chi_pn = mean_chi_pn + np.array(chi_pn)/len(mipmon_areas)
            if 'all' in show:
                mean_chi_all = mean_chi_all + np.array(chi_all)/len(mipmon_areas)
            if 'pg' in show:
                mean_chi_pn_pg = mean_chi_pn_pg + np.array(chi_pn_pg)/len(mipmon_areas)

        plt.figure(0)
        if 'pn' in show:
            plt.plot(dtimes, chi_pn, label = 'PN ' + mipmon_areas[i][0], c = colors[i], linestyle = '-', lw = 3)
        if 'all' in show:
            plt.plot(dtimes, chi_all, label = 'All ' + mipmon_areas[i][0], c = colors[i], linestyle = '--', lw = 3)
        if 'pg' in show:
            plt.plot(dtimes, chi_pn_pg, label = 'PG at PN ' + mipmon_areas[i][0], c = colors[i], linestyle = ':', lw = 3)

    if show_mean:
        if 'pn' in show:
            plt.plot(dtimes, mean_chi_pn, label = 'PN in all areas', c = 'k', linestyle = '-', lw = 3)
        if 'all' in show:
            plt.plot(dtimes, mean_chi_all, label = 'All in all areas', c = 'k', linestyle = '--', lw = 3)
        if 'pg' in show:
            plt.plot(dtimes, mean_chi_pn_pg, label = 'PG at PN in all areas', c = 'k', linestyle = ':', lw = 3)

    plt.figure(0)
    plt.xlabel('Time shift (days)')
    plt.ylabel(r'$\chi^2$')
    plt.legend()

def show_2dstats_parity(dataframe, variable1, variable2, threshold1, threshold2, \
                        parity_list = ['All prenatal', 'Primigravid', 'Multigravid'], \
                        figsize = None, show = True, xlims = None, ylims = None):
    colours = [cm.turbo(i/len(parity_list)) for i in range(len(parity_list))]
    plt.figure(figsize = figsize)
    for i in dataframe.index:
        for c, p in enumerate(parity_list):
            plt.annotate(str(i), xy = [dataframe[p + ' ' + variable1].loc[i], dataframe[p + ' ' + variable2].loc[i]], c = colours[c])

    xmin = min([dataframe[i + ' ' + variable1].min() for i in parity_list])
    xmax = max([dataframe[i + ' ' + variable1].max() for i in parity_list])
    ymin = min([dataframe[i + ' ' + variable2].min() for i in parity_list])
    ymax = max([dataframe[i + ' ' + variable2].max() for i in parity_list])
    if xlims is None:
        xlims = [xmin, xmax*1.3]
    if ylims is None:
        ylims = [ymin*.95, ymax*1.05]

    if threshold1 is not None:
        plt.vlines(threshold1, ylims[0], ylims[1], color = 'tab:grey')
    if threshold2 is not None:
        plt.hlines(threshold2, xlims[0], xlims[1], color = 'tab:grey')

    yshift = 1
    for c, p in enumerate(parity_list):
            plt.annotate(p, xy = [xmax*1.3, ymax*yshift], c = colours[c])
            yshift*=.95
    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.xlabel(variable1)
    plt.ylabel(variable2)
    if show:
        plt.show()

def show_stats_parity(dataframe, variable, threshold = None):
    plt.scatter(dataframe['All prenatal ' + variable], dataframe.index, color = 'tab:blue', marker = 's', label = 'All prenatal')
    plt.scatter(dataframe['Primigravid ' + variable], dataframe.index, color = 'tab:orange', marker = '^', label = 'Primigravid')
    plt.scatter(dataframe['Multigravid ' + variable], dataframe.index, color = 'tab:red', label = 'Multigravid')
    if threshold is not None:
        plt.vlines(threshold, dataframe.index.min(), dataframe.index.max(), color = 'tab:grey')
    if variable == 'chi2':
        plt.xlabel(r'$\chi^2$')
    else:
        plt.xlabel(variable)
    plt.legend()
    plt.show()

general_sero = ['MSP1','HSP40', 'Etramp', 'ACS5','EBA175', \
 'PfTramp','GEXP18','PfRH2','PfRH5', 'pAMA1', 'PvLDH', \
'PfHRP2', 'PfLDH']
pregnancy_sero = ['P1', 'P39','P5', 'P8','PD', 'DBL6e','DBL34']
peptides = ['P1', 'P39','P5', 'P8','PD']

def import_data(data_path = '~/isglobal/projects/pregmal/data/', \
                        mipmon_name = 'mipmon_merged.csv', \
                        serology_name = 'MiPMON_serostatus_wide.csv', \
                        rrs_name = 'RRS_data_age.csv', \
                        opd_name = 'weekly_OPD_cases_2014_2019_6_posts.csv', \
                        opd_5_name = 'weekly_OPD_cases_2014_2019_6_posts_age_5.csv', \
                        opd_2to9_name = 'weekly_OPD_cases_2014_2019_6_posts_age2to9.csv', \
                        excluded_antigens = ['pAMA1', 'PvLDH', 'P5', 'DBL6e', 'PfHRP2', \
                                             'PfLDH', 'general_pos', 'pregnancy_pos', \
                                             'breadth_general', 'breadth_pregnancy', \
                                            'breadth_peptides']):
    #Defining data paths
    mipmon_filename = data_path +  mipmon_name
    serology_filename = data_path + serology_name
    rrs_filename = data_path + rrs_name
    opd_filename = data_path + opd_name
    opd_5_filename = data_path + opd_5_name
    opd_2to9_filename = data_path + opd_2to9_name

    #Loading and preprocessing data
    mipmon = pd.read_csv(mipmon_filename)
    serology = pd.read_csv(serology_filename)
    rrs = pd.read_csv(rrs_filename)
    #rrs = rrs.rename(columns = {'yr' : 'year', 'mon' : 'month'})
    rrs = rrs.sort_values(['yr', 'week'])
    rrs['tot'] = rrs['tot_a'] #+ rrs['tot_b']
    rrs['tot_test'] = rrs['tot_test_a'] #+ rrs['tot_test_b']
    rrs['malaria'] = rrs['mal_a'] #+ rrs['mal_b']
    opd = pd.read_csv(opd_filename)
    opd_5 = pd.read_csv(opd_5_filename)
    opd_2to9 = pd.read_csv(opd_2to9_filename)
    #Reformating dates
    mipmon['visdate'] = pd.to_datetime(mipmon['visdate'])
    rrs['date'] = pd.to_datetime(rrs['yr'], format = "%Y") + pd.to_timedelta(7*rrs['week'], unit='D')
    opd['date'] = pd.to_datetime(opd['year'], format = "%Y") + pd.to_timedelta(7*opd['week'], unit='D')
    opd_5['date'] = pd.to_datetime(opd_5['year'], format = "%Y") + pd.to_timedelta(7*opd_5['week'], unit='D')
    opd_2to9['date'] = pd.to_datetime(opd_2to9['year'], format = "%Y") + pd.to_timedelta(7*opd_2to9['week'], unit='D')
    #Total aggregated data
    total_rrs = rrs.groupby(by = 'date')[['tot', 'tot_test', 'malaria']].sum()#['tot_a', 'tot_b', 'tot', 'tot_test_a', 'tot_test_b', 'tot_test', 'mal_a', 'mal_b', 'malaria']].sum()
    total_opd = opd.groupby(by = 'date')[['date', 'visits', 'malaria']].sum()
    total_opd_5 = opd_5.groupby(by = 'date')[['date', 'visits', 'malaria']].sum()
    total_opd_2to9 = opd_2to9.groupby(by = 'date')[['date', 'visits', 'malaria']].sum()
    #Quantify tests
    mipmon['pcrpos'][mipmon['pcrpos'] == 'PCR-'] = 0
    mipmon['pcrpos'][mipmon['pcrpos'] == 'PCR+'] = 1

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
            mipmon[cutoff+a][mipmon[cutoff+a] == 'intermediate'] = 1
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

    #Building incidence
    opd_2to9['incidence'] = 0
    rrs['incidence'] = 52802*.7 #Magude Sede is approximately 70% of all population in Magude
    place = 'Ilha Josina'
    mask = (opd_2to9['place'] == place)&(opd_2to9['date'] < '2016-07-01')
    opd_2to9['incidence'].loc[mask] = 2389*np.ones(np.sum(mask))
    mask = (opd_2to9['place'] == place)&(opd_2to9['date'] >= '2016-07-01')&(opd_2to9['date'] < '2017-07-01')
    opd_2to9['incidence'].loc[mask] = 2353*np.ones(np.sum(mask))
    mask = (opd_2to9['place'] == place)&(opd_2to9['date'] >= '2017-07-01')&(opd_2to9['date'] < '2018-07-01')
    opd_2to9['incidence'].loc[mask] = 2346*np.ones(np.sum(mask))
    mask = (opd_2to9['place'] == place)&(opd_2to9['date'] >= '2018-07-01')&(opd_2to9['date'] < '2019-07-01')
    opd_2to9['incidence'].loc[mask] = 2262*np.ones(np.sum(mask))
    mask = (opd_2to9['place'] == place)&(opd_2to9['date'] >= '2019-07-01')&(opd_2to9['date'] < '2020-07-01')
    opd_2to9['incidence'].loc[mask] = 2304*np.ones(np.sum(mask))

    place = 'Manhiça'
    mask = (opd_2to9['place'] == place)&(opd_2to9['date'] < '2016-07-01')
    opd_2to9['incidence'].loc[mask] = 9417*np.ones(np.sum(mask))
    mask = (opd_2to9['place'] == place)&(opd_2to9['date'] >= '2016-07-01')&(opd_2to9['date'] < '2017-07-01')
    opd_2to9['incidence'].loc[mask] = 9470*np.ones(np.sum(mask))
    mask = (opd_2to9['place'] == place)&(opd_2to9['date'] >= '2017-07-01')&(opd_2to9['date'] < '2018-07-01')
    opd_2to9['incidence'].loc[mask] = 9712*np.ones(np.sum(mask))
    mask = (opd_2to9['place'] == place)&(opd_2to9['date'] >= '2018-07-01')&(opd_2to9['date'] < '2019-07-01')
    opd_2to9['incidence'].loc[mask] = 9857*np.ones(np.sum(mask))
    mask = (opd_2to9['place'] == place)&(opd_2to9['date'] >= '2019-07-01')&(opd_2to9['date'] < '2020-07-01')
    opd_2to9['incidence'].loc[mask] = 9760*np.ones(np.sum(mask))

    return mipmon, rrs, opd, opd_5, opd_2to9, total_rrs, total_opd, total_opd_5, total_opd_2to9, antigens
