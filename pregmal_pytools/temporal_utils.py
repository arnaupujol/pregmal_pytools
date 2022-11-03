import pandas as pd
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import scipy.stats as sci_stats
from genomic_tools import utils, stats
import scipy.optimize as optimization
from stat_tools.errors import chi_square
from stat_tools import estimations
import pdb

general_sero = ['MSP1','HSP40', 'Etramp', 'ACS5','EBA175', \
 'PfTramp','GEXP18','PfRH2','PfRH5', 'pAMA1', 'PvLDH', \
'PfHRP2', 'PfLDH']
pregnancy_sero = ['P1', 'P39','P5', 'P8','PD', 'DBL6e','DBL34']
peptides = ['P1', 'P39','P5', 'P8','PD']

def import_temporal_data(data_path = '~/isglobal/projects/pregmal/data/', \
                        mipmon_name = 'mipmon_merged.csv', \
                        serology_name = 'MiPMON_serostatus_wide.csv', \
                        rrs_name = 'RRS_data_age.csv', \
                        opd_name = 'weekly_OPD_cases_2014_2019_6_posts.csv', \
                        opd_5_name = 'weekly_OPD_cases_2016_2019_age_5.csv', \
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
    antigens = []#TODO make for a in general_sero, pregnancy_sero: if 'FMM_' + a in mipmon.columns: antigens.append(a)
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

def clinic_pw_comp(mipmon, rrs, opd_5, mipmon_selection, all_mipmon_areas, all_clinic_areas, mipmon_bins, \
                  time_shift = 0, norm = True, clinic_val = 'cases', mip_test_type = 'pcr', rounding = 3, \
                   get_pcc = False, mipmon_prev_r = None):
    colors = [cm.turbo((i+1)/float(len(mipmon_selection) + 1)) for i in range(len(mipmon_selection) + 1)]

    chi2_results = {}
    mean_errs = {}
    pcorr_results = {}
    pcorr_95CI = {}
    pcorr_68CI = {}
    if mipmon_bins == 3:
        changes = {}
    for i, mipmon_areas in enumerate(all_mipmon_areas):
        chi2_results[mipmon_areas[0]] = {}
        mean_errs[mipmon_areas[0]] = {}
        pcorr_results[mipmon_areas[0]] = {}
        pcorr_95CI[mipmon_areas[0]] = {}
        pcorr_68CI[mipmon_areas[0]] = {}
        clinic_areas = all_clinic_areas[i]
        if mipmon_bins == 3:
            changes['1st yr in ' + mipmon_areas[0]] = {}
            changes['2nd yr in ' + mipmon_areas[0]] = {}
        print(mipmon_areas[0])
        for j,s in enumerate(mipmon_selection):
            mipmon_mask = mipmon_selection[s]
            if get_pcc and mipmon_prev_r is None:
                    dates, mipmon_prev, mipmon_err, clinic_bins, clinic_err, mipmon_prev_resample = \
                                clinic_mipmon_bins(mipmon, rrs, opd_5, mipmon_areas, \
                                                                    clinic_areas, mipmon_bins, \
                                                                    time_shift = time_shift, norm = norm, \
                                                                    mipmon_mask = mipmon_mask, \
                                                                    clinic_val = clinic_val, \
                                                                    mip_test_type = mip_test_type, \
                                                                   ret_resamples = True)
            else:
                dates, mipmon_prev, mipmon_err, clinic_bins, clinic_err = \
                                clinic_mipmon_bins(mipmon, rrs, opd_5, mipmon_areas, \
                                                                    clinic_areas, mipmon_bins, \
                                                                    time_shift = time_shift, norm = norm, \
                                                                    mipmon_mask = mipmon_mask, \
                                                                    clinic_val = clinic_val, \
                                                                    mip_test_type = mip_test_type)

            mask = (mipmon_err>0)&(clinic_err>0)&(mipmon_prev>0)&(clinic_bins>0)
            if get_pcc:
                if mipmon_prev_r is not None:
                    mipmon_prev_resample = mipmon_prev_r
                chi2, k, pcorr, conf_68, conf_95, pcorr_r = \
                get_clinic_amplitude_chi2(mipmon_prev[mask], mipmon_err[mask], \
                                                                 clinic_bins[mask], clinic_err[mask], \
                                                          get_pcc = True, mipmon_prev_r = mipmon_prev_resample[mask])
            else:
                chi2, k = get_clinic_amplitude_chi2(mipmon_prev[mask], mipmon_err[mask], \
                                                                 clinic_bins[mask], clinic_err[mask])
            chi2_results[mipmon_areas[0]][s] = round(chi2, rounding)
            mean_errs[mipmon_areas[0]][s] = round(np.mean(mipmon_err[mask]), rounding)
            pcorr_results[mipmon_areas[0]][s] = round(pcorr, rounding)
            pcorr_68CI[mipmon_areas[0]][s] = [round(conf_68[0], rounding),round(conf_68[1], rounding)]
            pcorr_95CI[mipmon_areas[0]][s] = [round(conf_95[0], rounding),round(conf_95[1], rounding)]
            if mipmon_bins == 3:
                if j == 0:
                    changes['1st yr in ' + mipmon_areas[0]]['clinics'] = round(clinic_bins[1]/clinic_bins[0], rounding)
                    changes['2nd yr in ' + mipmon_areas[0]]['clinics'] = round(clinic_bins[2]/clinic_bins[0], rounding)
                changes['1st yr in ' + mipmon_areas[0]][s] = round(mipmon_prev[1]/mipmon_prev[0], rounding)
                changes['2nd yr in ' + mipmon_areas[0]][s] = round(mipmon_prev[2]/mipmon_prev[0], rounding)
            print(s, chi2, k)
            if j == 0:
                plt.errorbar(dates, k*clinic_bins, k*clinic_err, lw = 3, label = 'Clinical cases', c = 'k')
            plt.errorbar(dates, mipmon_prev, mipmon_err, label = s, c = colors[j], lw = 3)
        plt.legend()
        plt.xlabel('Date')
        plt.ylabel(r'$\rm{PR}_{\rm{'+mip_test_type.upper()+'}}$/' + clinic_val)
        plt.xticks(dates[::int(len(dates)/4)])
        plt.title(str_replace(mipmon_areas[0], '-', ' '))
        plt.show()
    if mipmon_bins == 3:
        return chi2_results, changes
    else:
        if get_pcc:
            return chi2_results, mean_errs, pcorr_results, pcorr_95CI, pcorr_68CI
        else:
            return chi2_results, mean_errs

def str_replace(text, ch1, ch2):
    new_text = ''
    for i in text:
        if i == ch1:
            new_text += ch2
        else:
            new_text += i
    return new_text

def chi2_vs_timeshift(mipmon, rrs, opd_5, mipmon_selection, all_ts, all_mipmon_areas, all_clinic_areas, \
                     mipmon_bins, clinic_val, mip_test_type, show_all = False, show_all_means = False, \
                     show = True, convolve_size = None, get_CI = False):#TODO get PCC CI here
    lstyles = ['-.', '--', ':']
    colors = [cm.turbo((i+1)/float(len(mipmon_selection) + 1)) for i in range(len(mipmon_selection) + 1)]
    all_chi2_mean = np.zeros(len(all_ts))
    chi2_mean = {}
    pcorr_mean = {}
    min_chi2 = {}
    max_pcorr = {}
    time_lag_chi2 = {}
    time_lag_pcorr = {}
    #Defining dictionary
    chi2_vs_t = {}
    pcorr_vs_t = {}
    pcorr025_vs_t = {}
    pcorr975_vs_t = {}
    for j,s in enumerate(mipmon_selection):
        chi2_vs_t[s] = {}
        pcorr_vs_t[s] = {}
        pcorr025_vs_t[s] = {}
        pcorr975_vs_t[s] = {}
        for i, mipmon_areas in enumerate(all_mipmon_areas):
            chi2_vs_t[s][mipmon_areas[0]] = []
            pcorr_vs_t[s][mipmon_areas[0]] = []
            pcorr025_vs_t[s][mipmon_areas[0]] = []
            pcorr975_vs_t[s][mipmon_areas[0]] = []
        chi2_vs_t[s]['All'] = []
        pcorr_vs_t[s]['All'] = []
        pcorr025_vs_t[s]['All'] = []
        pcorr975_vs_t[s]['All'] = []
        chi2_vs_t[s]['Mean'] = []
        pcorr_vs_t[s]['Mean'] = []
        pcorr025_vs_t[s]['Mean'] = []
        pcorr975_vs_t[s]['Mean'] = []
    #Running analysis
    for j,s in enumerate(mipmon_selection):
        mipmon_mask = mipmon_selection[s]
        for ts in all_ts:
            all_mip_prev = np.array([])
            all_mip_err = np.array([])
            all_clinic = np.array([])
            all_clinic_err = np.array([])
            all_mip_prev_r = []
            all_pcorr_r = []
            for i, mipmon_areas in enumerate(all_mipmon_areas):
                clinic_areas = all_clinic_areas[i]
                dates, mipmon_prev, mipmon_err, clinic_bins, clinic_err, mipmon_prev_r = \
                                clinic_mipmon_bins(mipmon, rrs, opd_5, mipmon_areas, \
                                                    clinic_areas, mipmon_bins, \
                                                    time_shift = ts, norm = True, \
                                                    mipmon_mask = mipmon_mask, \
                                                    clinic_val = clinic_val, \
                                                    mip_test_type = mip_test_type, \
                                                  ret_resamples = True)

                mask = (mipmon_err>0)&(clinic_err>0)&(mipmon_prev>0)&(clinic_bins>0)
                chi2, k, pcorr, conf_68, conf_95, pcorr_r = get_clinic_amplitude_chi2(mipmon_prev[mask], \
                                                                             mipmon_err[mask], \
                                                                 clinic_bins[mask], clinic_err[mask], \
                                                                           get_pcc = True, \
                                                                            mipmon_prev_r = mipmon_prev_r[mask])
                chi2_vs_t[s][mipmon_areas[0]].append(chi2)
                pcorr_vs_t[s][mipmon_areas[0]].append(pcorr)
                #CI of PCC for each area and time
                pcorr025_vs_t[s][mipmon_areas[0]].append(conf_95[0])#TODO test
                pcorr975_vs_t[s][mipmon_areas[0]].append(conf_95[1])#TODO test
                #Concatenating all areas
                all_mip_prev = np.concatenate((all_mip_prev, mipmon_prev)).flatten()
                all_mip_err = np.concatenate((all_mip_err, mipmon_err)).flatten()
                all_clinic = np.concatenate((all_clinic, k*clinic_bins)).flatten()
                all_clinic_err = np.concatenate((all_clinic_err, k*clinic_err)).flatten()
                all_mip_prev_r.append(mipmon_prev_r)
                all_pcorr_r.append(pcorr_r)
            #TODO get pcc and CI for eachtime for All
            all_mip_prev_r = np.array(all_mip_prev_r)
            all_mip_prev_r = np.reshape(all_mip_prev_r, \
                                        (all_mip_prev_r.shape[0]*all_mip_prev_r.shape[1], \
                                         all_mip_prev_r.shape[2]))
            mask = (all_mip_prev>0)&(all_mip_err>0)&(all_clinic>0)&(all_clinic_err>0)
            chi2, k, pcorr, conf_68, conf_95, pcorr_r = get_clinic_amplitude_chi2(all_mip_prev[mask], \
                                                                             all_mip_err[mask], \
                                                                 all_clinic[mask], all_clinic_err[mask], \
                                                                           get_pcc = True, \
                                                                            mipmon_prev_r = all_mip_prev_r[mask])
            pcorr025_vs_t[s]['All'].append(conf_95[0])#TODO test
            pcorr975_vs_t[s]['All'].append(conf_95[1])#TODO test
            pcorr_vs_t[s]['All'].append(pcorr)#TODO test
            pcorr_vs_t[s]['Mean'].append((pcorr_vs_t[s]['Magude-sede'][-1] + pcorr_vs_t[s]['Manhica-Sede'][-1] + \
                                          pcorr_vs_t[s]['Ilha-Josina'][-1])/3.)
            #get CI of pcorr Mean
            mean_pcorr_r = np.sum(np.array(all_pcorr_r), axis = 0)/len(all_pcorr_r)#TODO test
            mean_pcorr_r_sorted = np.sort(np.array(mean_pcorr_r))
            pcorr025_vs_t[s]['Mean'].append(mean_pcorr_r_sorted[int(.025*len(mean_pcorr_r))])#TODO test
            pcorr975_vs_t[s]['Mean'].append(mean_pcorr_r_sorted[int(.975*len(mean_pcorr_r))])#TODO test

        for i, mipmon_areas in enumerate(all_mipmon_areas):
            chi2_vs_t[s][mipmon_areas[0]] = np.array(chi2_vs_t[s][mipmon_areas[0]])
            pcorr_vs_t[s][mipmon_areas[0]] = np.array(pcorr_vs_t[s][mipmon_areas[0]])
            pcorr025_vs_t[s][mipmon_areas[0]] = np.array(pcorr025_vs_t[s][mipmon_areas[0]])
            pcorr975_vs_t[s][mipmon_areas[0]] = np.array(pcorr975_vs_t[s][mipmon_areas[0]])
            if show_all:
                plt.figure(0)
                plt.plot(all_ts, chi2_vs_t[s][mipmon_areas[0]], label = mipmon_areas[0] + ' ' + s, \
                         c = colors[j], lw = 3, linestyle = lstyles[i])
                plt.figure(1)
                plt.plot(all_ts, pcorr_vs_t[s][mipmon_areas[0]], label = mipmon_areas[0] + ' ' + s, \
                         c = colors[j], lw = 3, linestyle = lstyles[i])

        pcorr_vs_t[s]['All'] = np.array(pcorr_vs_t[s]['All'])
        pcorr025_vs_t[s]['All'] = np.array(pcorr025_vs_t[s]['All'])
        pcorr975_vs_t[s]['All'] = np.array(pcorr975_vs_t[s]['All'])
        pcorr_vs_t[s]['Mean'] = np.array(pcorr_vs_t[s]['Mean'])
        pcorr025_vs_t[s]['Mean'] = np.array(pcorr025_vs_t[s]['Mean'])
        pcorr975_vs_t[s]['Mean'] = np.array(pcorr975_vs_t[s]['Mean'])

        chi2_vs_t[s]['Mean'] = (chi2_vs_t[s]['Magude-sede'] + chi2_vs_t[s]['Manhica-Sede'] + chi2_vs_t[s]['Ilha-Josina'])/3.
        chi2_vs_t[s]['Mean'] = estimations.convolve_ones(chi2_vs_t[s]['Mean'], convolve_size)
        pcorr_vs_t[s]['Mean'] = estimations.convolve_ones(pcorr_vs_t[s]['Mean'], convolve_size)

        chi2_mean[s] = chi2_vs_t[s]['Mean']
        pcorr_mean[s] = pcorr_vs_t[s]['Mean']
        if convolve_size is None:
            min_chi2[s] = min(chi2_vs_t[s]['Mean'])
            max_pcorr[s] = max(pcorr_vs_t[s]['Mean'])
        else:
            min_chi2[s] = min(chi2_vs_t[s]['Mean'][convolve_size:-convolve_size])
            max_pcorr[s] = max(pcorr_vs_t[s]['Mean'][convolve_size:-convolve_size])
        w = np.where(chi2_vs_t[s]['Mean'] == min_chi2[s])[0][0]
        time_lag_chi2[s] = all_ts[w]
        w = np.where(pcorr_vs_t[s]['Mean'] == max_pcorr[s])[0][0]
        time_lag_pcorr[s] = all_ts[w]
        all_chi2_mean += chi2_vs_t[s]['Mean']
        plt.figure(0)
        if convolve_size is None:
            plt.plot(all_ts, chi2_vs_t[s]['Mean'], label = 'Mean ' + s, c = colors[j], lw = 3, linestyle = '-')
        else:
            plt.plot(all_ts[convolve_size:-convolve_size], chi2_vs_t[s]['Mean'][convolve_size:-convolve_size], label = 'Mean ' + s, c = colors[j], lw = 3, linestyle = '-')
        plt.figure(1)
        if convolve_size is None:
            plt.plot(all_ts, pcorr_vs_t[s]['Mean'], label = 'Mean ' + s, c = colors[j], lw = 3, linestyle = '-')
        else:
            plt.plot(all_ts[convolve_size:-convolve_size], \
                     pcorr_vs_t[s]['Mean'][convolve_size:-convolve_size], \
                     label = 'Mean ' + s, c = colors[j], lw = 3, linestyle = '-')
    if show_all_means:
        all_chi2_mean /= (j+1)
        plt.figure(0)
        if convolve_size is None:
            plt.plot(all_ts, all_chi2_mean, label = 'Total mean', c = 'k', lw = 3, linestyle = '-')
        else:
            plt.plot(all_ts[convolve_size:-convolve_size], all_chi2_mean[convolve_size:-convolve_size], label = 'Total mean', c = 'k', lw = 3, linestyle = '-')

    plt.figure(0)
    plt.legend()
    plt.xlabel('Time shift (days)')
    plt.ylabel(r'$\chi^2$')
    plt.figure(1)
    plt.legend()
    plt.xlabel('Time shift (days)')
    plt.ylabel(r'Pearson Correlation Coefficient')
    if show:
        plt.show()
    if get_CI:
        return min_chi2, max_pcorr, time_lag_chi2, time_lag_pcorr, chi2_mean, pcorr_mean, all_ts, pcorr_vs_t, pcorr025_vs_t, pcorr975_vs_t
    else:
        return min_chi2, max_pcorr, time_lag_chi2, time_lag_pcorr, chi2_mean, pcorr_mean, all_ts


def chi2_vs_timeshift_old(mipmon, rrs, opd_5, mipmon_selection, all_ts, all_mipmon_areas, all_clinic_areas, \
                     mipmon_bins, clinic_val, mip_test_type, show_all = False, show_all_means = False, \
                     show = True, convolve_size = None):#TODO get PCC CI here
    lstyles = ['-.', '--', ':']
    colors = [cm.turbo((i+1)/float(len(mipmon_selection) + 1)) for i in range(len(mipmon_selection) + 1)]
    all_chi2_mean = np.zeros(len(all_ts))
    all_pcorr_mean = np.zeros(len(all_ts))
    chi2_mean = {}
    pcorr_mean = {}
    min_chi2 = {}
    max_pcorr = {}
    time_lag_chi2 = {}
    time_lag_pcorr = {}
    for j,s in enumerate(mipmon_selection):
        mipmon_mask = mipmon_selection[s]
        chi2_vs_t = {}
        pcorr_vs_t = {}
        for i, mipmon_areas in enumerate(all_mipmon_areas):
            clinic_areas = all_clinic_areas[i]
            chi2_vs_t[mipmon_areas[0]] = []
            pcorr_vs_t[mipmon_areas[0]] = []
            for ts in all_ts:
                dates, mipmon_prev, mipmon_err, clinic_bins, clinic_err, mipmon_prev_r = \
                                clinic_mipmon_bins(mipmon, rrs, opd_5, mipmon_areas, \
                                                                    clinic_areas, mipmon_bins, \
                                                                    time_shift = ts, norm = True, \
                                                                    mipmon_mask = mipmon_mask, \
                                                                    clinic_val = clinic_val, \
                                                                   mip_test_type = mip_test_type, \
                                                  ret_resamples = True)

                mask = (mipmon_err>0)&(clinic_err>0)&(mipmon_prev>0)&(clinic_bins>0)
                chi2, k, pcorr = get_clinic_amplitude_chi2(mipmon_prev[mask], mipmon_err[mask], \
                                                                 clinic_bins[mask], clinic_err[mask], \
                                                                           get_pcc = True)
                chi2_vs_t[mipmon_areas[0]].append(chi2)
                pcorr_vs_t[mipmon_areas[0]].append(pcorr)
            chi2_vs_t[mipmon_areas[0]] = np.array(chi2_vs_t[mipmon_areas[0]])
            pcorr_vs_t[mipmon_areas[0]] = np.array(pcorr_vs_t[mipmon_areas[0]])
            if show_all:
                plt.figure(0)
                plt.plot(all_ts, chi2_vs_t[mipmon_areas[0]], label = mipmon_areas[0] + ' ' + s, c = colors[j], lw = 3, linestyle = lstyles[i])
                plt.figure(1)
                plt.plot(all_ts, pcorr_vs_t[mipmon_areas[0]], label = mipmon_areas[0] + ' ' + s, c = colors[j], lw = 3, linestyle = lstyles[i])
        chi2_vs_t['Mean'] = (chi2_vs_t['Magude-sede'] + chi2_vs_t['Manhica-Sede'] + chi2_vs_t['Ilha-Josina'])/3.
        pcorr_vs_t['Mean'] = (pcorr_vs_t['Magude-sede'] + pcorr_vs_t['Manhica-Sede'] + pcorr_vs_t['Ilha-Josina'])/3.
        chi2_vs_t['Mean'] = estimations.convolve_ones(chi2_vs_t['Mean'], convolve_size)
        pcorr_vs_t['Mean'] = estimations.convolve_ones(pcorr_vs_t['Mean'], convolve_size)
        chi2_mean[s] = chi2_vs_t['Mean']
        pcorr_mean[s] = pcorr_vs_t['Mean']
        if convolve_size is None:
            min_chi2[s] = min(chi2_vs_t['Mean'])
        else:
            min_chi2[s] = min(chi2_vs_t['Mean'][convolve_size:-convolve_size])
        max_pcorr[s] = max(pcorr_vs_t['Mean'])
        w = np.where(chi2_vs_t['Mean'] == min_chi2[s])[0][0]
        time_lag_chi2[s] = all_ts[w]
        w = np.where(pcorr_vs_t['Mean'] == max_pcorr[s])[0][0]
        time_lag_pcorr[s] = all_ts[w]
        all_chi2_mean += chi2_vs_t['Mean']
        all_pcorr_mean += pcorr_vs_t['Mean']
        plt.figure(0)
        if convolve_size is None:
            plt.plot(all_ts, chi2_vs_t['Mean'], label = 'Mean ' + s, c = colors[j], lw = 3, linestyle = '-')
        else:
            plt.plot(all_ts[convolve_size:-convolve_size], chi2_vs_t['Mean'][convolve_size:-convolve_size], label = 'Mean ' + s, c = colors[j], lw = 3, linestyle = '-')
        plt.figure(1)
        if convolve_size is None:
            plt.plot(all_ts, pcorr_vs_t['Mean'], label = 'Mean ' + s, c = colors[j], lw = 3, linestyle = '-')
        else:
            plt.plot(all_ts[convolve_size:-convolve_size], \
                     pcorr_vs_t['Mean'][convolve_size:-convolve_size], \
                     label = 'Mean ' + s, c = colors[j], lw = 3, linestyle = '-')
    if show_all_means:
        all_chi2_mean /= (j+1)
        all_pcorr_mean /= (j+1)
        plt.figure(0)
        if convolve_size is None:
            plt.plot(all_ts, all_chi2_mean, label = 'Total mean', c = 'k', lw = 3, linestyle = '-')
        else:
            plt.plot(all_ts[convolve_size:-convolve_size], all_chi2_mean[convolve_size:-convolve_size], label = 'Total mean', c = 'k', lw = 3, linestyle = '-')
        plt.figure(1)
        if convolve_size is None:
            plt.plot(all_ts, all_pcorr_mean, label = 'Total mean', c = 'k', lw = 3, linestyle = '-')
        else:
            plt.plot(all_ts[convolve_size:-convolve_size], all_pcorr_mean[convolve_size:-convolve_size], label = 'Total mean', c = 'k', lw = 3, linestyle = '-')
    plt.figure(0)
    plt.legend()
    plt.xlabel('Time shift (days)')
    plt.ylabel(r'$\chi^2$')
    plt.figure(1)
    plt.legend()
    plt.xlabel('Time shift (days)')
    plt.ylabel(r'Pearson Correlation Coefficient')
    if show:
        plt.show()
    return min_chi2, max_pcorr, time_lag_chi2, time_lag_pcorr, chi2_mean, pcorr_mean, all_ts

def chi2_vs_timeshift_vs_ags(mipmon, rrs, opd_5, mipmon_selection, all_ts, all_mipmon_areas, all_clinic_areas, \
                     mipmon_bins, clinic_val, antigens, show_all = False, show_all_means = False, cutoff = 'FMM', \
                             figsize = [12,10], ymax_chi2 = None, convolve_size = None, show = True, get_CI = False):
    lstyles = ['-.', '--', ':']
    colors = [cm.turbo((i+1)/float(len(antigens) + 1)) for i in range(len(antigens) + 1)]
    all_chi2_mean = np.zeros(len(all_ts))
    all_pcorr_mean = np.zeros(len(all_ts))
    chi2_mean = {}
    pcorr_mean = {}
    min_chi2 = {}
    max_pcorr = {}
    time_lag_chi2 = {}
    time_lag_pcorr = {}
    #Defining dictionary
    chi2_vs_t = {}
    pcorr_vs_t = {}
    pcorr025_vs_t = {}
    pcorr975_vs_t = {}
    plt.figure(0, figsize = figsize)
    plt.figure(1, figsize = figsize)
    for j,antigen in enumerate(antigens):
        chi2_vs_t[antigen] = {}
        pcorr_vs_t[antigen] = {}
        pcorr025_vs_t[antigen] = {}
        pcorr975_vs_t[antigen] = {}
        for i, mipmon_areas in enumerate(all_mipmon_areas):
            chi2_vs_t[antigen][mipmon_areas[0]] = []
            pcorr_vs_t[antigen][mipmon_areas[0]] = []
            pcorr025_vs_t[antigen][mipmon_areas[0]] = []
            pcorr975_vs_t[antigen][mipmon_areas[0]] = []
        chi2_vs_t[antigen]['Mean'] = []
        pcorr_vs_t[antigen]['Mean'] = []
        pcorr025_vs_t[antigen]['Mean'] = []
        pcorr975_vs_t[antigen]['Mean'] = []
    #Running analysis
    for j,antigen in enumerate(antigens):
        for ts in all_ts:
            all_mip_prev = np.array([])
            all_mip_err = np.array([])
            all_clinic = np.array([])
            all_clinic_err = np.array([])
            all_mip_prev_r = []
            all_pcorr_r = []
            for i, mipmon_areas in enumerate(all_mipmon_areas):
                clinic_areas = all_clinic_areas[i]
                dates, mipmon_prev, mipmon_err, clinic_bins, clinic_err, mipmon_prev_r = \
                                clinic_mipmon_bins(mipmon, rrs, opd_5, mipmon_areas, \
                                                    clinic_areas, mipmon_bins, \
                                                    time_shift = ts, norm = True, \
                                                    mipmon_mask = mipmon_selection, \
                                                    clinic_val = clinic_val, \
                                                    mip_test_type = cutoff + '_' + antigen, \
                                                  ret_resamples = True)

                mask = (mipmon_err>0)&(clinic_err>0)&(mipmon_prev>0)&(clinic_bins>0)
                chi2, k, pcorr, conf_68, conf_95, pcorr_r = get_clinic_amplitude_chi2(mipmon_prev[mask], \
                                                                                      mipmon_err[mask], \
                                                                 clinic_bins[mask], clinic_err[mask], \
                                                                           get_pcc = True, \
                                                                            mipmon_prev_r = mipmon_prev_r[mask])
                chi2_vs_t[antigen][mipmon_areas[0]].append(chi2)
                pcorr_vs_t[antigen][mipmon_areas[0]].append(pcorr)
                all_pcorr_r.append(pcorr_r)
                #CI of PCC for each area and time
                pcorr025_vs_t[antigen][mipmon_areas[0]].append(conf_95[0])#TODO test
                pcorr975_vs_t[antigen][mipmon_areas[0]].append(conf_95[1])#TODO test
            pcorr_vs_t[antigen]['Mean'].append((pcorr_vs_t[antigen]['Magude-sede'][-1] + pcorr_vs_t[antigen]['Manhica-Sede'][-1] + \
                                          pcorr_vs_t[antigen]['Ilha-Josina'][-1])/3.)
            #get CI of pcorr Mean
            mean_pcorr_r = np.sum(np.array(all_pcorr_r), axis = 0)/len(all_pcorr_r)#TODO test
            mean_pcorr_r_sorted = np.sort(np.array(mean_pcorr_r))
            pcorr025_vs_t[antigen]['Mean'].append(mean_pcorr_r_sorted[int(.025*len(mean_pcorr_r))])#TODO test
            pcorr975_vs_t[antigen]['Mean'].append(mean_pcorr_r_sorted[int(.975*len(mean_pcorr_r))])#TODO test
        for i, mipmon_areas in enumerate(all_mipmon_areas):
            chi2_vs_t[antigen][mipmon_areas[0]] = np.array(chi2_vs_t[antigen][mipmon_areas[0]])
            pcorr_vs_t[antigen][mipmon_areas[0]] = np.array(pcorr_vs_t[antigen][mipmon_areas[0]])
            pcorr025_vs_t[antigen][mipmon_areas[0]] = np.array(pcorr025_vs_t[antigen][mipmon_areas[0]])
            pcorr975_vs_t[antigen][mipmon_areas[0]] = np.array(pcorr975_vs_t[antigen][mipmon_areas[0]])
            if show_all:
                plt.figure(0)
                plt.plot(all_ts, chi2_vs_t[antigen][mipmon_areas[0]], label = mipmon_areas[0] + ' ' + antigen, c = colors[j], lw = 3, linestyle = lstyles[i])
                plt.figure(1)
                plt.plot(all_ts, pcorr_vs_t[antigen][mipmon_areas[0]], label = mipmon_areas[0] + ' ' + antigen, c = colors[j], lw = 3, linestyle = lstyles[i])
        pcorr_vs_t[antigen]['Mean'] = np.array(pcorr_vs_t[antigen]['Mean'])
        pcorr025_vs_t[antigen]['Mean'] = np.array(pcorr025_vs_t[antigen]['Mean'])
        pcorr975_vs_t[antigen]['Mean'] = np.array(pcorr975_vs_t[antigen]['Mean'])

        chi2_vs_t[antigen]['Mean'] = (chi2_vs_t[antigen]['Magude-sede'] + chi2_vs_t[antigen]['Manhica-Sede'] + chi2_vs_t[antigen]['Ilha-Josina'])/3.
        chi2_vs_t[antigen]['Mean'] = estimations.convolve_ones(chi2_vs_t[antigen]['Mean'], convolve_size)
        pcorr_vs_t[antigen]['Mean'] = estimations.convolve_ones(pcorr_vs_t[antigen]['Mean'], convolve_size)

        chi2_mean[antigen] = chi2_vs_t[antigen]['Mean']
        pcorr_mean[antigen] = pcorr_vs_t[antigen]['Mean']
        if convolve_size is None:
            min_chi2[antigen] = min(chi2_vs_t[antigen]['Mean'])
            max_pcorr[antigen] = max(pcorr_vs_t[antigen]['Mean'])
        else:
            min_chi2[antigen] = min(chi2_vs_t[antigen]['Mean'][convolve_size:-convolve_size])
            max_pcorr[antigen] = max(pcorr_vs_t[antigen]['Mean'][convolve_size:-convolve_size])

        w = np.where(chi2_vs_t[antigen]['Mean'] == min_chi2[antigen])[0][0]
        time_lag_chi2[antigen] = all_ts[w]
        w = np.where(pcorr_vs_t[antigen]['Mean'] == max_pcorr[antigen])[0][0]
        time_lag_pcorr[antigen] = all_ts[w]
        all_chi2_mean += chi2_vs_t[antigen]['Mean']

        plt.figure(0)
        if convolve_size is None:
            plt.plot(all_ts, chi2_vs_t[antigen]['Mean'], label = 'Mean ' + antigen, c = colors[j], lw = 3, linestyle = '-')
        else:
            plt.plot(all_ts[convolve_size:-convolve_size], \
                     chi2_vs_t[antigen]['Mean'][convolve_size:-convolve_size], \
                     label = 'Mean ' + antigen, c = colors[j], lw = 3, linestyle = '-')
        plt.figure(1)
        if convolve_size is None:
            plt.plot(all_ts, pcorr_vs_t[antigen]['Mean'], label = 'Mean ' + antigen, c = colors[j], lw = 3, linestyle = '-')
        else:
            plt.plot(all_ts[convolve_size:-convolve_size], \
                     pcorr_vs_t[antigen]['Mean'][convolve_size:-convolve_size], \
                     label = 'Mean ' + antigen, c = colors[j], lw = 3, linestyle = '-')



    if show_all_means:
        all_chi2_mean /= (j+1)
        all_pcorr_mean /= (j+1)
        plt.figure(0)
        if convolve_size is None:
            plt.plot(all_ts, all_chi2_mean, label = 'Total mean', c = 'k', lw = 3, linestyle = '-')
        else:
            plt.plot(all_ts[convolve_size:-convolve_size], all_chi2_mean[convolve_size:-convolve_size], label = 'Total mean', c = 'k', lw = 3, linestyle = '-')
        plt.figure(1)
        if convolve_size is None:
            plt.plot(all_ts, all_pcorr_mean, label = 'Total mean', c = 'k', lw = 3, linestyle = '-')
        else:
            plt.plot(all_ts[convolve_size:-convolve_size], all_pcorr_mean[convolve_size:-convolve_size], label = 'Total mean', c = 'k', lw = 3, linestyle = '-')
    plt.figure(0)
    plt.ylim(ymax = ymax_chi2)
    plt.legend()
    plt.xlabel('Time shift (days)')
    plt.ylabel(r'$\chi^2$')
    plt.figure(1)
    plt.legend()
    plt.xlabel('Time shift (days)')
    plt.ylabel(r'Pearson Correlation Coefficient')
    plt.show()
    if show:
        plt.show()
    if get_CI:
        return min_chi2, max_pcorr, time_lag_chi2, time_lag_pcorr, chi2_mean, pcorr_mean, all_ts, pcorr_vs_t, pcorr025_vs_t, pcorr975_vs_t
    else:
        return min_chi2, max_pcorr, time_lag_chi2, time_lag_pcorr, chi2_mean, pcorr_mean, all_ts


def chi2_vs_timeshift_vs_ags_old(mipmon, rrs, opd_5, mipmon_selection, all_ts, all_mipmon_areas, all_clinic_areas, \
                     mipmon_bins, clinic_val, antigens, show_all = False, show_all_means = False, cutoff = 'FMM', \
                             figsize = [12,10], ymax_chi2 = None, convolve_size = None):
    lstyles = ['-.', '--', ':']
    colors = [cm.turbo((i+1)/float(len(antigens) + 1)) for i in range(len(antigens) + 1)]
    all_chi2_mean = np.zeros(len(all_ts))
    all_pcorr_mean = np.zeros(len(all_ts))
    chi2_mean = {}
    pcorr_mean = {}
    min_chi2 = {}
    max_pcorr = {}
    time_lag_chi2 = {}
    time_lag_pcorr = {}
    plt.figure(0, figsize = figsize)
    plt.figure(1, figsize = figsize)
    for j,antigen in enumerate(antigens):
        chi2_vs_t = {}
        pcorr_vs_t = {}
        for i, mipmon_areas in enumerate(all_mipmon_areas):
            clinic_areas = all_clinic_areas[i]
            chi2_vs_t[mipmon_areas[0]] = []
            pcorr_vs_t[mipmon_areas[0]] = []
            for ts in all_ts:
                dates, mipmon_prev, mipmon_err, clinic_bins, clinic_err = \
                                clinic_mipmon_bins(mipmon, rrs, opd_5, mipmon_areas, \
                                                                    clinic_areas, mipmon_bins, \
                                                                    time_shift = ts, norm = True, \
                                                                    mipmon_mask = mipmon_selection, \
                                                                    clinic_val = clinic_val, \
                                                                   mip_test_type = cutoff + '_' + antigen)

                mask = (mipmon_err>0)&(clinic_err>0)&(mipmon_prev>0)&(clinic_bins>0)
                chi2, k, pcorr = get_clinic_amplitude_chi2(mipmon_prev[mask], mipmon_err[mask], \
                                                                 clinic_bins[mask], clinic_err[mask], \
                                                                           get_pcc = True)
                chi2_vs_t[mipmon_areas[0]].append(chi2)
                pcorr_vs_t[mipmon_areas[0]].append(pcorr)
            chi2_vs_t[mipmon_areas[0]] = np.array(chi2_vs_t[mipmon_areas[0]])
            pcorr_vs_t[mipmon_areas[0]] = np.array(pcorr_vs_t[mipmon_areas[0]])
            if show_all:
                plt.figure(0)
                plt.plot(all_ts, chi2_vs_t[mipmon_areas[0]], label = mipmon_areas[0] + ' ' + antigen, c = colors[j], lw = 3, linestyle = lstyles[i])
                plt.figure(1)
                plt.plot(all_ts, pcorr_vs_t[mipmon_areas[0]], label = mipmon_areas[0] + ' ' + antigen, c = colors[j], lw = 3, linestyle = lstyles[i])
        chi2_vs_t['Mean'] = (chi2_vs_t['Magude-sede'] + chi2_vs_t['Manhica-Sede'] + chi2_vs_t['Ilha-Josina'])/3.
        pcorr_vs_t['Mean'] = (pcorr_vs_t['Magude-sede'] + pcorr_vs_t['Manhica-Sede'] + pcorr_vs_t['Ilha-Josina'])/3.
        chi2_vs_t['Mean'] = estimations.convolve_ones(chi2_vs_t['Mean'], convolve_size)
        pcorr_vs_t['Mean'] = estimations.convolve_ones(pcorr_vs_t['Mean'], convolve_size)
        chi2_mean[antigen] = chi2_vs_t['Mean']
        pcorr_mean[antigen] = pcorr_vs_t['Mean']
        if convolve_size is None:
            min_chi2[antigen] = min(chi2_vs_t['Mean'])
        else:
            min_chi2[antigen] = min(chi2_vs_t['Mean'][convolve_size:-convolve_size])
        max_pcorr[antigen] = max(pcorr_vs_t['Mean'])
        w = np.where(chi2_vs_t['Mean'] == min_chi2[antigen])[0][0]
        time_lag_chi2[antigen] = all_ts[w]
        w = np.where(pcorr_vs_t['Mean'] == max_pcorr[antigen])[0][0]
        time_lag_pcorr[antigen] = all_ts[w]
        all_chi2_mean += chi2_vs_t['Mean']
        all_pcorr_mean += pcorr_vs_t['Mean']
        plt.figure(0)
        if convolve_size is None:
            plt.plot(all_ts, chi2_vs_t['Mean'], label = 'Mean ' + antigen, c = colors[j], lw = 3, linestyle = '-')
        else:
            plt.plot(all_ts[convolve_size:-convolve_size], \
                     chi2_vs_t['Mean'][convolve_size:-convolve_size], \
                     label = 'Mean ' + antigen, c = colors[j], lw = 3, linestyle = '-')
        plt.figure(1)
        if convolve_size is None:
            plt.plot(all_ts, pcorr_vs_t['Mean'], label = 'Mean ' + antigen, c = colors[j], lw = 3, linestyle = '-')
        else:
            plt.plot(all_ts[convolve_size:-convolve_size], \
                     pcorr_vs_t['Mean'][convolve_size:-convolve_size], \
                     label = 'Mean ' + antigen, c = colors[j], lw = 3, linestyle = '-')
    if show_all_means:
        all_chi2_mean /= (j+1)
        all_pcorr_mean /= (j+1)
        plt.figure(0)
        if convolve_size is None:
            plt.plot(all_ts, all_chi2_mean, label = 'Total mean', c = 'k', lw = 3, linestyle = '-')
        else:
            plt.plot(all_ts[convolve_size:-convolve_size], all_chi2_mean[convolve_size:-convolve_size], label = 'Total mean', c = 'k', lw = 3, linestyle = '-')
        plt.figure(1)
        if convolve_size is None:
            plt.plot(all_ts, all_pcorr_mean, label = 'Total mean', c = 'k', lw = 3, linestyle = '-')
        else:
            plt.plot(all_ts[convolve_size:-convolve_size], all_pcorr_mean[convolve_size:-convolve_size], label = 'Total mean', c = 'k', lw = 3, linestyle = '-')
    plt.figure(0)
    plt.ylim(ymax = ymax_chi2)
    plt.legend()
    plt.xlabel('Time shift (days)')
    plt.ylabel(r'$\chi^2$')
    plt.figure(1)
    plt.legend()
    plt.xlabel('Time shift (days)')
    plt.ylabel(r'Pearson Correlation Coefficient')
    plt.show()
    return min_chi2, max_pcorr, time_lag_chi2, time_lag_pcorr, chi2_mean, pcorr_mean, all_ts

def chi2_vs_timeshift_vs_ags_vs_selection(mipmon, rrs, opd_5, mipmon_selection, all_ts, \
                                          all_mipmon_areas, all_clinic_areas, mipmon_bins, \
                                          clinic_val, antigens, show_all = False, show_all_means = False, \
                                          cutoff = 'FMM', figsize = [12,10], ymax_chi2 = None, \
                                         convolve_size = None):

    min_chi2_test, max_pcorr_test, time_lag_chi2_test, time_lag_pcorr_test, \
    chi2_mean_test, pcorr_mean_test, all_ts_test = {}, {}, {}, {}, {}, {}, {}
    pcorr_vs_t, pcorr025_vs_t, pcorr975_vs_t = {}, {}, {}
    for s in mipmon_selection:
        print(clinic_val, s)
        min_chi2_test[s], max_pcorr_test[s], time_lag_chi2_test[s], time_lag_pcorr_test[s], \
        chi2_mean_test[s], pcorr_mean_test[s], all_ts_test[s] = {}, {}, {}, {}, {}, {}, {}
        pcorr_vs_t[s], pcorr025_vs_t[s], pcorr975_vs_t[s] = {}, {}, {}

        min_chi2_test[s], max_pcorr_test[s], time_lag_chi2_test[s], time_lag_pcorr_test[s], \
        chi2_mean_test[s], pcorr_mean_test[s], all_ts_test[s], \
        pcorr_vs_t[s], pcorr025_vs_t[s], pcorr975_vs_t[s] = chi2_vs_timeshift_vs_ags(mipmon, rrs, opd_5, \
                                                                                         mipmon_selection[s], all_ts, \
                                                                                         all_mipmon_areas, \
                                                                                         all_clinic_areas, mipmon_bins, \
                                                                                         clinic_val, antigens, \
                                                                                         show_all, show_all_means, \
                                                                                         cutoff, figsize, ymax_chi2, \
                                                                                        convolve_size = convolve_size, \
                                                                                        get_CI = True)
    return min_chi2_test, max_pcorr_test, time_lag_chi2_test, time_lag_pcorr_test, \
                chi2_mean_test, pcorr_mean_test, all_ts_test, pcorr_vs_t, pcorr025_vs_t, pcorr975_vs_t

def chi2_vs_timeshift_vs_ags_vs_selection_old(mipmon, rrs, opd_5, mipmon_selection, all_ts, \
                                          all_mipmon_areas, all_clinic_areas, mipmon_bins, \
                                          clinic_val, antigens, show_all = False, show_all_means = False, \
                                          cutoff = 'FMM', figsize = [12,10], ymax_chi2 = None, \
                                         convolve_size = None):

    min_chi2_test, max_pcorr_test, time_lag_chi2_test, time_lag_pcorr_test, \
    chi2_mean_test, pcorr_mean_test, all_ts_test = {}, {}, {}, {}, {}, {}, {}
    for s in mipmon_selection:
        print(clinic_val, s)
        min_chi2_test[s], max_pcorr_test[s], time_lag_chi2_test[s], time_lag_pcorr_test[s], \
        chi2_mean_test[s], pcorr_mean_test[s], all_ts_test[s] = {}, {}, {}, {}, {}, {}, {}
        min_chi2_test[s], max_pcorr_test[s], time_lag_chi2_test[s], time_lag_pcorr_test[s], \
        chi2_mean_test[s], pcorr_mean_test[s], all_ts_test[s] = chi2_vs_timeshift_vs_ags(mipmon, rrs, opd_5, \
                                                                                         mipmon_selection[s], all_ts, \
                                                                                         all_mipmon_areas, \
                                                                                         all_clinic_areas, mipmon_bins, \
                                                                                         clinic_val, antigens, \
                                                                                         show_all, show_all_means, \
                                                                                         cutoff, figsize, ymax_chi2, \
                                                                                        convolve_size = convolve_size)
    return min_chi2_test, max_pcorr_test, time_lag_chi2_test, time_lag_pcorr_test, \
                chi2_mean_test, pcorr_mean_test, all_ts_test

def get_clinic_amplitude_chi2(mipmon_prev, mipmon_err, clinic_bins, clinic_err, get_pcc = False, \
                             mipmon_prev_r = None, clinic_bins_r = None):
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
    clinic_bins_r: np.ndarray
        Mean weekly clinical cases per time bin for each bootstrap resample

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
                #Define clinic data to use in each rand
                if clinic_bins_r is None:
                    clinic_for_ls = clinic_bins
                else:
                    if clinic_bins_r.shape[1] < nrands:
                        print("Warning: not enough subsamples for clinical data for bootstrapping")
                        clinic_for_ls = clinic_bins
                    else:
                        clinic_for_ls = clinic_bins_r[:,r]
                mask = (mipmon_prev_r[:,r] >= 0)&(clinic_for_ls>=0)#TODO change >= to >
                k_r, out = optimization.leastsq(residual, 1., args=(mipmon_prev_r[:,r][mask], mipmon_err[mask], clinic_for_ls[mask], clinic_err[mask]))#TODO test
                pcorr_r.append(sci_stats.pearsonr(mipmon_prev_r[:,r][mask], k_r*clinic_for_ls[mask])[0])#TODO test
            pcorr_r = np.array(pcorr_r)#TODO test
            sorted_pcorrs = np.sort(pcorr_r)
            conf_95 = [sorted_pcorrs[int(nrands*.025)], sorted_pcorrs[int(nrands*.975)]]
            conf_68 = [sorted_pcorrs[int(nrands*.16)], sorted_pcorrs[int(nrands*.84)]]
            return chi2, k, pcorr, conf_68, conf_95, pcorr_r
        else:
            return chi2, k, pcorr
    else:
        return chi2, k

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
    mipmon_prev_r: np.array
        MiPMon prevalence per time bin obtained from the bootstrap resamples
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

def chi2_vs_timeshift_parasito_vs_ags_vs_selection(mipmon, mipmon_selection_para, mipmon_selection_sero, \
                                                   all_ts, all_mipmon_areas, mipmon_bins, mipmon_test, \
                                                   antigens, show_all = False, show_all_means = False, \
                                                   cutoff = 'FMM', figsize = [12,10], ymax_chi2 = None, \
                                                  convolve_size = None, get_CI = False):

    min_chi2_test, max_pcorr_test, time_lag_chi2_test, time_lag_pcorr_test, \
    chi2_mean_test, pcorr_mean_test, all_ts_test, chi2_vs_t, pcorr_vs_t, \
    pcorr025_vs_t, pcorr975_vs_t = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}

    for s in mipmon_selection_sero:
        print(mipmon_test, s)
        min_chi2_test[s], max_pcorr_test[s], time_lag_chi2_test[s], time_lag_pcorr_test[s], \
        chi2_mean_test[s], pcorr_mean_test[s], all_ts_test[s], chi2_vs_t[s], pcorr_vs_t[s], \
        pcorr025_vs_t[s], pcorr975_vs_t[s] = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}

        out = chi2_vs_timeshift_parasito_vs_ags(mipmon, mipmon_selection_para, mipmon_selection_sero[s], \
                                                all_ts, all_mipmon_areas, mipmon_bins, mipmon_test, antigens, \
                                                show_all = show_all, show_all_means = show_all_means, \
                                                cutoff = cutoff, figsize = figsize, ymax_chi2 = ymax_chi2, \
                                                convolve_size = convolve_size, get_CI = get_CI)
        if get_CI:#TODO test
            min_chi2_test[s], max_pcorr_test[s], time_lag_chi2_test[s], time_lag_pcorr_test[s], \
            chi2_mean_test[s], pcorr_mean_test[s], all_ts_test[s], chi2_vs_t[s], pcorr_vs_t[s], \
            pcorr025_vs_t[s], pcorr975_vs_t[s] = out
        else:
            min_chi2_test[s], max_pcorr_test[s], time_lag_chi2_test[s], time_lag_pcorr_test[s], \
            chi2_mean_test[s], pcorr_mean_test[s], all_ts_test[s], chi2_vs_t[s], pcorr_vs_t[s] = out

    if get_CI:#TODO test
        return min_chi2_test, max_pcorr_test, time_lag_chi2_test, time_lag_pcorr_test, \
                chi2_mean_test, pcorr_mean_test, all_ts_test, chi2_vs_t, pcorr_vs_t, \
                pcorr025_vs_t, pcorr975_vs_t
    else:
        return min_chi2_test, max_pcorr_test, time_lag_chi2_test, time_lag_pcorr_test, \
                chi2_mean_test, pcorr_mean_test, all_ts_test, chi2_vs_t, pcorr_vs_t

def chi2_vs_timeshift_parasito_vs_ags_vs_selection_old(mipmon, mipmon_selection_para, mipmon_selection_sero, \
                                                   all_ts, all_mipmon_areas, mipmon_bins, mipmon_test, \
                                                   antigens, show_all = False, show_all_means = False, \
                                                   cutoff = 'FMM', figsize = [12,10], ymax_chi2 = None, \
                                                  convolve_size = None, return_all_areas = False):

    min_chi2_test, max_pcorr_test, time_lag_chi2_test, time_lag_pcorr_test, \
    chi2_mean_test, pcorr_mean_test, all_ts_test = {}, {}, {}, {}, {}, {}, {}

    if return_all_areas:
        all_chi2_vs_t = {}
        all_pcc_vs_t = {}

    for s in mipmon_selection_sero:
        print(mipmon_test, s)
        min_chi2_test[s], max_pcorr_test[s], time_lag_chi2_test[s], time_lag_pcorr_test[s], \
        chi2_mean_test[s], pcorr_mean_test[s], all_ts_test[s] = {}, {}, {}, {}, {}, {}, {}

        if return_all_areas:
            all_chi2_vs_t[s] = {}
            all_pcc_vs_t[s] = {}

        out = chi2_vs_timeshift_parasito_vs_ags_old(mipmon, mipmon_selection_para, mipmon_selection_sero[s], \
                                                all_ts, all_mipmon_areas, mipmon_bins, mipmon_test, antigens, \
                                                show_all = show_all, show_all_means = show_all_means, \
                                                cutoff = cutoff, figsize = figsize, ymax_chi2 = ymax_chi2, \
                                                convolve_size = convolve_size, return_all_areas = return_all_areas)
        if return_all_areas:
            min_chi2_test[s], max_pcorr_test[s], time_lag_chi2_test[s], time_lag_pcorr_test[s], \
        chi2_mean_test[s], pcorr_mean_test[s], all_ts_test[s], all_chi2_vs_t[s], all_pcc_vs_t[s] = out
        else:
            min_chi2_test[s], max_pcorr_test[s], time_lag_chi2_test[s], time_lag_pcorr_test[s], \
        chi2_mean_test[s], pcorr_mean_test[s], all_ts_test[s] = out

    if return_all_areas:
        return min_chi2_test, max_pcorr_test, time_lag_chi2_test, time_lag_pcorr_test, \
                chi2_mean_test, pcorr_mean_test, all_ts_test, all_chi2_vs_t, all_pcc_vs_t
    else:
        return min_chi2_test, max_pcorr_test, time_lag_chi2_test, time_lag_pcorr_test, \
                chi2_mean_test, pcorr_mean_test, all_ts_test

def chi2_vs_timeshift_parasito_vs_ags(mipmon, mipmon_selection_para, mipmon_selection_sero, all_ts, \
                                      all_mipmon_areas, mipmon_bins, mipmon_test, antigens, \
                                      show_all = False, show_all_means = False, cutoff = 'FMM', \
                                      figsize = [12,10], ymax_chi2 = None, convolve_size = None, \
                                      return_all_areas = False, show = True, get_CI = False):
    lstyles = ['-.', '--', ':']
    colors = [cm.turbo((i+1)/float(len(antigens) + 1)) for i in range(len(antigens) + 1)]
    all_chi2_mean = np.zeros(len(all_ts))
    all_pcorr_mean = np.zeros(len(all_ts))
    chi2_mean = {}
    pcorr_mean = {}
    min_chi2 = {}
    max_pcorr = {}
    time_lag_chi2 = {}
    time_lag_pcorr = {}
    #Defining dictionary
    chi2_vs_t = {}
    pcorr_vs_t = {}
    pcorr025_vs_t = {}
    pcorr975_vs_t = {}
    for j,antigen in enumerate(antigens):
        chi2_vs_t[antigen] = {}
        pcorr_vs_t[antigen] = {}
        pcorr025_vs_t[antigen] = {}
        pcorr975_vs_t[antigen] = {}
        for i, mipmon_areas in enumerate(all_mipmon_areas):
            chi2_vs_t[antigen][mipmon_areas[0]] = []
            pcorr_vs_t[antigen][mipmon_areas[0]] = []
            pcorr025_vs_t[antigen][mipmon_areas[0]] = []
            pcorr975_vs_t[antigen][mipmon_areas[0]] = []
        pcorr_vs_t[antigen]['All'] = []
        pcorr025_vs_t[antigen]['All'] = []
        pcorr975_vs_t[antigen]['All'] = []
        chi2_vs_t[antigen]['Mean'] = []
        pcorr_vs_t[antigen]['Mean'] = []
        pcorr025_vs_t[antigen]['Mean'] = []
        pcorr975_vs_t[antigen]['Mean'] = []
    plt.figure(0, figsize = figsize)
    plt.figure(1, figsize = figsize)
    #Running analysis
    for j,antigen in enumerate(antigens):
        for ts in all_ts:
            all_sero_prev = np.array([])
            all_sero_err = np.array([])
            all_para_prev = np.array([])
            all_para_err = np.array([])
            all_mip_sero_prev_r = []
            all_mip_para_prev_r = []
            all_pcorr_r = []
            if ts > 0:
                time_mask_sero = mipmon['visdate'] > mipmon['visdate'].min() + pd.to_timedelta(ts, unit = 'D')
                time_mask_para = mipmon['visdate'] < mipmon['visdate'].max() - pd.to_timedelta(ts, unit = 'D')
            else:
                time_mask_sero = mipmon['visdate'] < mipmon['visdate'].max() + pd.to_timedelta(ts, unit = 'D')
                time_mask_para = mipmon['visdate'] > mipmon['visdate'].min() - pd.to_timedelta(ts, unit = 'D')
            for i, mipmon_areas in enumerate(all_mipmon_areas):
                #Get serological bins
                dates, mip_sero_prev, mip_sero_err, \
                mip_sero_prev_r = get_mipmon_pos_bins(mipmon, mipmon_areas, mipmon_bins, time_shift = 0, \
                                                    mipmon_mask = time_mask_sero&mipmon_selection_sero, \
                                                    mipmon_test = cutoff + '_' + antigen, \
                                                    ret_resamples = True)
                #Get parasitological bins
                dates_para, mip_para_prev, mip_para_err, \
                mip_para_prev_r = get_mipmon_pos_bins(mipmon, mipmon_areas, mipmon_bins, time_shift = ts, \
                                                    mipmon_mask = time_mask_para&mipmon_selection_para, \
                                                    mipmon_test = mipmon_test, \
                                                    ret_resamples = True)
                mask = (mip_sero_prev>0)&(mip_sero_err>0)&(mip_para_prev>0)&(mip_para_err>0)
                #TODO test, make set_trace() and check impact of clinic_bins_r
                chi2, k, pcorr, conf_68, conf_95, pcorr_r  = get_clinic_amplitude_chi2(mip_sero_prev[mask], mip_sero_err[mask], \
                                                                 mip_para_prev[mask], mip_para_err[mask], \
                                                                           get_pcc = True, \
                                                                            mipmon_prev_r = mip_sero_prev_r[mask], \
                                                                            clinic_bins_r = mip_para_prev_r[mask])

                chi2_vs_t[antigen][mipmon_areas[0]].append(chi2)
                pcorr_vs_t[antigen][mipmon_areas[0]].append(pcorr)
                #CI of PCC for each area and time
                pcorr025_vs_t[antigen][mipmon_areas[0]].append(conf_95[0])#TODO test
                pcorr975_vs_t[antigen][mipmon_areas[0]].append(conf_95[1])#TODO test
                #Concatenating all areas
                all_sero_prev = np.concatenate((all_sero_prev, mip_sero_prev)).flatten()
                all_sero_err = np.concatenate((all_sero_err, mip_sero_err)).flatten()
                all_para_prev = np.concatenate((all_para_prev, mip_para_prev)).flatten()
                all_para_err = np.concatenate((all_para_err, mip_para_err)).flatten()
                all_mip_sero_prev_r.append(mip_sero_prev_r)
                all_mip_para_prev_r.append(mip_para_prev_r)
                all_pcorr_r.append(pcorr_r)
            #TODO get pcc and CI for each time for All
            all_mip_sero_prev_r = np.array(all_mip_sero_prev_r)
            all_mip_para_prev_r = np.array(all_mip_para_prev_r)
            all_mip_sero_prev_r = np.reshape(all_mip_sero_prev_r, \
                                        (all_mip_sero_prev_r.shape[0]*all_mip_sero_prev_r.shape[1], \
                                         all_mip_sero_prev_r.shape[2]))
            all_mip_para_prev_r = np.reshape(all_mip_para_prev_r, \
                                        (all_mip_para_prev_r.shape[0]*all_mip_para_prev_r.shape[1], \
                                         all_mip_para_prev_r.shape[2]))
            mask = (all_sero_prev>0)&(all_sero_err>0)&(all_para_prev>0)&(all_para_err>0)
            chi2, k, pcorr, conf_68, conf_95, pcorr_r = get_clinic_amplitude_chi2(all_sero_prev[mask], \
                                                                             all_sero_err[mask], \
                                                                 all_para_prev[mask], all_para_err[mask], \
                                                                           get_pcc = True, \
                                                                            mipmon_prev_r = all_mip_sero_prev_r[mask], \
                                                                            clinic_bins_r = all_mip_para_prev_r[mask])
            pcorr025_vs_t[antigen]['All'].append(conf_95[0])#TODO test
            pcorr975_vs_t[antigen]['All'].append(conf_95[1])#TODO test
            pcorr_vs_t[antigen]['All'].append(pcorr)#TODO test
            pcorr_vs_t[antigen]['Mean'].append((pcorr_vs_t[antigen]['Magude-sede'][-1] + pcorr_vs_t[antigen]['Manhica-Sede'][-1] + \
                                          pcorr_vs_t[antigen]['Ilha-Josina'][-1])/3.)
            #get CI of pcorr Mean
            mean_pcorr_r = np.sum(np.array(all_pcorr_r), axis = 0)/len(all_pcorr_r)#TODO test
            mean_pcorr_r_sorted = np.sort(np.array(mean_pcorr_r))
            pcorr025_vs_t[antigen]['Mean'].append(mean_pcorr_r_sorted[int(.025*len(mean_pcorr_r))])#TODO test
            pcorr975_vs_t[antigen]['Mean'].append(mean_pcorr_r_sorted[int(.975*len(mean_pcorr_r))])#TODO test
        for i, mipmon_areas in enumerate(all_mipmon_areas):
            chi2_vs_t[antigen][mipmon_areas[0]] = np.array(chi2_vs_t[antigen][mipmon_areas[0]])
            pcorr_vs_t[antigen][mipmon_areas[0]] = np.array(pcorr_vs_t[antigen][mipmon_areas[0]])
            pcorr025_vs_t[antigen][mipmon_areas[0]] = np.array(pcorr025_vs_t[antigen][mipmon_areas[0]])
            pcorr975_vs_t[antigen][mipmon_areas[0]] = np.array(pcorr975_vs_t[antigen][mipmon_areas[0]])
            if show_all:
                plt.figure(0)
                plt.plot(all_ts, chi2_vs_t[antigen][mipmon_areas[0]], label = mipmon_areas[0] + ' ' + antigen, c = colors[j], \
                         lw = 3, linestyle = lstyles[i])
                plt.figure(1)
                plt.plot(all_ts, pcorr_vs_t[antigen][mipmon_areas[0]], label = mipmon_areas[0] + ' ' + antigen, c = colors[j], \
                         lw = 3, linestyle = lstyles[i])

        pcorr_vs_t[antigen]['All'] = np.array(pcorr_vs_t[antigen]['All'])
        pcorr025_vs_t[antigen]['All'] = np.array(pcorr025_vs_t[antigen]['All'])
        pcorr975_vs_t[antigen]['All'] = np.array(pcorr975_vs_t[antigen]['All'])
        pcorr_vs_t[antigen]['Mean'] = np.array(pcorr_vs_t[antigen]['Mean'])
        pcorr025_vs_t[antigen]['Mean'] = np.array(pcorr025_vs_t[antigen]['Mean'])
        pcorr975_vs_t[antigen]['Mean'] = np.array(pcorr975_vs_t[antigen]['Mean'])

        chi2_vs_t[antigen]['Mean'] = (chi2_vs_t[antigen]['Magude-sede'] + chi2_vs_t[antigen]['Manhica-Sede'] + chi2_vs_t[antigen]['Ilha-Josina'])/3.
        chi2_vs_t[antigen]['Mean'] = estimations.convolve_ones(chi2_vs_t[antigen]['Mean'], convolve_size)
        pcorr_vs_t[antigen]['Mean'] = estimations.convolve_ones(pcorr_vs_t[antigen]['Mean'], convolve_size)
        pcorr_vs_t[antigen]['All'] = estimations.convolve_ones(pcorr_vs_t[antigen]['All'], convolve_size)

        chi2_mean[antigen] = chi2_vs_t[antigen]['Mean']
        pcorr_mean[antigen] = pcorr_vs_t[antigen]['Mean']
        if convolve_size is None:
            min_chi2[antigen] = min(chi2_vs_t[antigen]['Mean'])
            max_pcorr[antigen] = max(pcorr_vs_t[antigen]['All'])
        else:
            min_chi2[antigen] = min(chi2_vs_t[antigen]['Mean'][convolve_size:-convolve_size])
            max_pcorr[antigen] = max(pcorr_vs_t[antigen]['Mean'][convolve_size:-convolve_size])
            max_pcorr[antigen] = max(pcorr_vs_t[antigen]['All'][convolve_size:-convolve_size])
        w = np.where(chi2_vs_t[antigen]['Mean'] == min_chi2[antigen])[0][0]
        time_lag_chi2[antigen] = all_ts[w]
        w = np.where(pcorr_vs_t[antigen]['All'] == max_pcorr[antigen])[0][0]
        time_lag_pcorr[antigen] = all_ts[w]
        all_chi2_mean += chi2_vs_t[antigen]['Mean']
        all_pcorr_mean += pcorr_vs_t[antigen]['Mean']
        plt.figure(0)
        if convolve_size is None:
            plt.plot(all_ts, chi2_vs_t[antigen]['Mean'], label = antigen, c = colors[j], lw = 3, linestyle = '-')
        else:
            plt.plot(all_ts[convolve_size:-convolve_size], \
                     chi2_vs_t[antigen]['Mean'][convolve_size:-convolve_size], \
                     label = antigen, c = colors[j], lw = 3, linestyle = '-')
        plt.figure(1)
        if convolve_size is None:
            plt.plot(all_ts, pcorr_vs_t[antigen]['All'], label = antigen, c = colors[j], lw = 3, linestyle = '-')
        else:
            plt.plot(all_ts[convolve_size:-convolve_size], \
                     pcorr_vs_t[antigen]['All'][convolve_size:-convolve_size], \
                     label = antigen, c = colors[j], lw = 3, linestyle = '-')
    if show_all_means:
        all_chi2_mean /= (j+1)
        all_pcorr_mean /= (j+1)
        plt.figure(0)
        if convolve_size is None:
            plt.plot(all_ts, all_chi2_mean, label = 'Total mean', c = 'k', lw = 3, linestyle = '-')
        else:
            plt.plot(all_ts[convolve_size:-convolve_size], all_chi2_mean[convolve_size:-convolve_size], label = 'Total mean', c = 'k', lw = 3, linestyle = '-')
        plt.figure(1)
        if convolve_size is None:
            plt.plot(all_ts, all_pcorr_mean, label = 'Total mean', c = 'k', lw = 3, linestyle = '-')
        else:
            plt.plot(all_ts[convolve_size:-convolve_size], all_pcorr_mean[convolve_size:-convolve_size], label = 'Total mean', c = 'k', lw = 3, linestyle = '-')
    plt.figure(0)
    plt.ylim(ymax = ymax_chi2)
    plt.legend()
    plt.xlabel('Time shift (days)')
    plt.ylabel(r'$\chi^2$')
    plt.figure(1)
    plt.legend()
    plt.xlabel('Time shift (days)')
    plt.ylabel(r'Pearson Correlation Coefficient')
    if show:
        plt.show()
    if get_CI:
        return min_chi2, max_pcorr, time_lag_chi2, time_lag_pcorr, chi2_mean, pcorr_mean, all_ts, chi2_vs_t, pcorr_vs_t, pcorr025_vs_t, pcorr975_vs_t
    else:
        return min_chi2, max_pcorr, time_lag_chi2, time_lag_pcorr, chi2_mean, pcorr_mean, all_ts, chi2_vs_t, pcorr_vs_t


def chi2_vs_timeshift_parasito_vs_ags_old(mipmon, mipmon_selection_para, mipmon_selection_sero, all_ts, \
                                      all_mipmon_areas, mipmon_bins, mipmon_test, antigens, \
                                      show_all = False, show_all_means = False, cutoff = 'FMM', \
                                      figsize = [12,10], ymax_chi2 = None, convolve_size = None, return_all_areas = False):
    lstyles = ['-.', '--', ':']
    colors = [cm.turbo((i+1)/float(len(antigens) + 1)) for i in range(len(antigens) + 1)]
    all_chi2_mean = np.zeros(len(all_ts))
    all_pcorr_mean = np.zeros(len(all_ts))
    chi2_mean = {}
    pcorr_mean = {}
    min_chi2 = {}
    max_pcorr = {}
    time_lag_chi2 = {}
    time_lag_pcorr = {}
    all_chi2_vs_t = {}
    all_pcc_vs_t = {}
    plt.figure(0, figsize = figsize)
    plt.figure(1, figsize = figsize)
    for j,antigen in enumerate(antigens):
        chi2_vs_t = {}
        pcorr_vs_t = {}
        for i, mipmon_areas in enumerate(all_mipmon_areas):
            chi2_vs_t[mipmon_areas[0]] = []
            pcorr_vs_t[mipmon_areas[0]] = []
            for ts in all_ts:
                if ts > 0:
                    time_mask_sero = mipmon['visdate'] > mipmon['visdate'].min() + pd.to_timedelta(ts, unit = 'D')
                    time_mask_para = mipmon['visdate'] < mipmon['visdate'].max() - pd.to_timedelta(ts, unit = 'D')
                else:
                    time_mask_sero = mipmon['visdate'] < mipmon['visdate'].max() + pd.to_timedelta(ts, unit = 'D')
                    time_mask_para = mipmon['visdate'] > mipmon['visdate'].min() - pd.to_timedelta(ts, unit = 'D')
                #Get serological bins
                dates, mip_sero_prev, mip_sero_err = get_mipmon_pos_bins(mipmon, mipmon_areas, mipmon_bins, time_shift = 0, \
                                                                         mipmon_mask = time_mask_sero&mipmon_selection_sero, \
                                                                         mipmon_test = cutoff + '_' + antigen, \
                                                                         ret_resamples = False)
                #Get parasitological bins
                dates_para, mip_para_prev, mip_para_err = get_mipmon_pos_bins(mipmon, mipmon_areas, mipmon_bins, time_shift = ts, \
                                                                              mipmon_mask = time_mask_para&mipmon_selection_para, \
                                                                              mipmon_test = mipmon_test, \
                                                                              ret_resamples = False)

                mask = (mip_sero_prev>0)&(mip_sero_err>0)&(mip_para_prev>0)&(mip_para_err>0)
                chi2, k, pcorr = get_clinic_amplitude_chi2(mip_sero_prev[mask], mip_sero_err[mask], \
                                                                 mip_para_prev[mask], mip_para_err[mask], \
                                                                           get_pcc = True)
                chi2_vs_t[mipmon_areas[0]].append(chi2)
                pcorr_vs_t[mipmon_areas[0]].append(pcorr)
            chi2_vs_t[mipmon_areas[0]] = np.array(chi2_vs_t[mipmon_areas[0]])
            pcorr_vs_t[mipmon_areas[0]] = np.array(pcorr_vs_t[mipmon_areas[0]])
            if show_all:
                plt.figure(0)
                plt.plot(all_ts, chi2_vs_t[mipmon_areas[0]], label = mipmon_areas[0] + ' ' + antigen, c = colors[j], \
                         lw = 3, linestyle = lstyles[i])
                plt.figure(1)
                plt.plot(all_ts, pcorr_vs_t[mipmon_areas[0]], label = mipmon_areas[0] + ' ' + antigen, c = colors[j], \
                         lw = 3, linestyle = lstyles[i])
        chi2_vs_t['Mean'] = (chi2_vs_t['Magude-sede'] + chi2_vs_t['Manhica-Sede'] + chi2_vs_t['Ilha-Josina'])/3.
        pcorr_vs_t['Mean'] = (pcorr_vs_t['Magude-sede'] + pcorr_vs_t['Manhica-Sede'] + pcorr_vs_t['Ilha-Josina'])/3.
        chi2_vs_t['Mean'] = estimations.convolve_ones(chi2_vs_t['Mean'], convolve_size)
        pcorr_vs_t['Mean'] = estimations.convolve_ones(pcorr_vs_t['Mean'], convolve_size)

        all_chi2_vs_t[antigen] = chi2_vs_t
        all_pcc_vs_t[antigen] = pcorr_vs_t
        chi2_mean[antigen] = chi2_vs_t['Mean']
        pcorr_mean[antigen] = pcorr_vs_t['Mean']
        if convolve_size is None:
            min_chi2[antigen] = min(chi2_vs_t['Mean'])
        else:
            min_chi2[antigen] = min(chi2_vs_t['Mean'][convolve_size:-convolve_size])
        max_pcorr[antigen] = max(pcorr_vs_t['Mean'])
        w = np.where(chi2_vs_t['Mean'] == min_chi2[antigen])[0][0]
        time_lag_chi2[antigen] = all_ts[w]
        w = np.where(pcorr_vs_t['Mean'] == max_pcorr[antigen])[0][0]
        time_lag_pcorr[antigen] = all_ts[w]
        all_chi2_mean += chi2_vs_t['Mean']
        all_pcorr_mean += pcorr_vs_t['Mean']
        plt.figure(0)
        if convolve_size is None:
            plt.plot(all_ts, chi2_vs_t['Mean'], label = 'Mean ' + antigen, c = colors[j], lw = 3, linestyle = '-')
        else:
            plt.plot(all_ts[convolve_size:-convolve_size], \
                     chi2_vs_t['Mean'][convolve_size:-convolve_size], \
                     label = 'Mean ' + antigen, c = colors[j], lw = 3, linestyle = '-')
        plt.figure(1)
        if convolve_size is None:
            plt.plot(all_ts, pcorr_vs_t['Mean'], label = 'Mean ' + antigen, c = colors[j], lw = 3, linestyle = '-')
        else:
            plt.plot(all_ts[convolve_size:-convolve_size], \
                     pcorr_vs_t['Mean'][convolve_size:-convolve_size], \
                     label = 'Mean ' + antigen, c = colors[j], lw = 3, linestyle = '-')
    if show_all_means:
        all_chi2_mean /= (j+1)
        all_pcorr_mean /= (j+1)
        plt.figure(0)
        if convolve_size is None:
            plt.plot(all_ts, all_chi2_mean, label = 'Total mean', c = 'k', lw = 3, linestyle = '-')
        else:
            plt.plot(all_ts[convolve_size:-convolve_size], all_chi2_mean[convolve_size:-convolve_size], label = 'Total mean', c = 'k', lw = 3, linestyle = '-')
        plt.figure(1)
        if convolve_size is None:
            plt.plot(all_ts, all_pcorr_mean, label = 'Total mean', c = 'k', lw = 3, linestyle = '-')
        else:
            plt.plot(all_ts[convolve_size:-convolve_size], all_pcorr_mean[convolve_size:-convolve_size], label = 'Total mean', c = 'k', lw = 3, linestyle = '-')
    plt.figure(0)
    plt.ylim(ymax = ymax_chi2)
    plt.legend()
    plt.xlabel('Time shift (days)')
    plt.ylabel(r'$\chi^2$')
    plt.figure(1)
    plt.legend()
    plt.xlabel('Time shift (days)')
    plt.ylabel(r'Pearson Correlation Coefficient')
    plt.show()
    if return_all_areas:
        return min_chi2, max_pcorr, time_lag_chi2, time_lag_pcorr, chi2_mean, pcorr_mean, all_ts, all_chi2_vs_t, all_pcc_vs_t
    else:
        return min_chi2, max_pcorr, time_lag_chi2, time_lag_pcorr, chi2_mean, pcorr_mean, all_ts

def get_mipmon_pos_bins(mipmon, mipmon_areas, time_bins, time_shift = 0, mipmon_mask = None, \
                        mipmon_test = 'pcr', ret_resamples = False):
    """
    This methods calculates the mean positivity of MiPMon data in different time bins.

    Parameters:
    -----------
    mipmon: pd.DataFrame
        DataFrame of MiPMon samples
    mipmon_areas: list
        List of area names for MiPMon samples
    time_bins: int
        Number of bins for MiPMon samples
    time_shift: int
        Time shift to apply backwards for clinical cases in number of days
    mipmon_mask: np.array
        Mask to apply to the MiPMon samples
    mipmon_test: str {'pcr', 'rdt'}
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
    mipmon_prev_r: np.array
        MiPMon prevalence per time bin obtained from the bootstrap resamples
    """
    mipmon_areas_mask = utils.get_mipmon_area_mask(mipmon, mipmon_areas)
    if mipmon_mask is None:
        mipmon_mask = mipmon['pcrpos'].notnull().notnull()&mipmon_areas_mask
    else:
        mipmon_mask = mipmon['pcrpos'].notnull().notnull()&mipmon_areas_mask&mipmon_mask

    #Define MiPMon time bin edges
    out, time_bins = pd.cut(mipmon['visdate'][mipmon_mask] + pd.to_timedelta(time_shift, unit = 'D'), time_bins, retbins = True)

    #MiPMon prevalence
    if mipmon_test == 'pcr':
        mip_pos = mipmon['pcrpos']
    elif mipmon_test == 'rdt':
        mip_pos = mipmon['density'] >= 100
    else:
        if mipmon_test in mipmon.columns:
            mip_pos = mipmon[mipmon_test]
            mipmon_mask = mipmon_mask&mipmon[mipmon_test].notnull()
        else:
            print("Wrong mipmon test type assignment: ", mipmon_test)
            print("Using pcrpos test instead")
            mip_pos = mipmon['pcrpos']
    out = stats.mean_prev_time_bins(mipmon['visdate'] + pd.to_timedelta(time_shift, unit = 'D'), \
                                    mip_pos, \
                                    data_mask = mipmon_mask&mip_pos.notnull(), \
                                    nbins = time_bins, \
                                    nrands = 1000, \
                                    ret_resamples = ret_resamples)
    if ret_resamples:
        dates, mipmon_prev, mipmon_err, mipmon_prev_r = out
    else:
        dates, mipmon_prev, mipmon_err = out
    return out

def list2array(str_list):
    final_array = []
    for i in str_list.split(' '):
        if i in ['[', ']', '\n', ' ', '']:
            continue
        elif i[0] in ['[', ']']:
            final_array.append(float(i[1:]))
        elif i[-1] in ['[', ']']:
            final_array.append(float(i[:-1]))
        elif i[-2:] == '\n':
            final_array.append(float(i[:-2]))
        else:
            final_array.append(float(i))
    final_array = np.array(final_array)
    return final_array
def all_list2array(dataframe):
    for s in dataframe:
        for a in dataframe[s].index:
            dataframe[s][a] = list2array(dataframe[s][a])
    return dataframe
