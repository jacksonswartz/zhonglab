import glob
import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from lmfit import Parameters, Model
from lmfit.models import LorentzianModel, ConstantModel, ExponentialModel, LinearModel, VoigtModel, GaussianModel
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap

#Note: use single scan code
date = "7/23"

# for data

#Change t_pump
DATA_DIR = ("/Users/JacksonS/Documents/zhonglab"
            "/07_23_24/10mK_12Amp/pumpnprobe/tr2_196042p746GHz/changing_T_pump/Twait_1p28ms__apump_0p9")

BG_DIR = ("/Users/JacksonS/Documents/zhonglab"
            "/07_23_24/10mK_12Amp/probe/offres")

TEK_HEADER = ["ParamLabel", "ParamVal", "None", "Seconds", "Volts", "None2"]  # hard-coded from TEK oscilloscope


# for peak finding
DISTANCE = 100  # TODO: better way to explicitly calculate this?
PROMINENCE = [0.02, 0.02, 0.02, 0.02, 0.05, 
              0.05, 0.05, 0.08, 0.08, 0.08, 
              0.08, 0.1, 0.1, 0.1, 0.08, 
              0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.1]

PROMINENCE_SCAN = 1

# for fitting of data
LOG_SCALE = True

#for fitting and plotting (from 6/20 code)
SCAN_RANGE = 50  # Unit: MHz
SCAN_TIME = 512e-6  # Unit: s
GAIN = 1e8  # Unit: V/W
EDGE_THRESH = 1  # For finding rising/falling edge of oscilloscope trigger

# for plotting
# plotting parameters
mpl.rcParams.update({'font.size': 12,
                     'figure.figsize': (8, 6)})

# plotting params
CMAP_OFFSET = 0.3
CMAP = cm.Blues
xlim = (-SCAN_RANGE/2, SCAN_RANGE/2)
ylim = (0, 4.5)

LOG_CMAP = True  # use log scale for colormap

# for plotting
# plotting parameters
mpl.rcParams.update({'font.size': 12,
                     'figure.figsize': (8, 6)})
xlim_all_plots = (-1, 11)
PLOT_BG = True
PLOT_DECAY = True

# plotting output control
PLOT_OD = False
PLOT_OD_SPLIT = False

PLOT_ALL_SCANS = False  # plot all scans with background average ##issue: plots ALL found transmissions for particular time, so we get vertical lines
PLOT_ALL_PEAKS = False  # plot all peak transmissions, with fitted double-decay exponential
PLOT_ALL_PEAKS_OD = True
PLOT_ALL_HEIGHTS = False  # plot all individually fitted hole heights, with fitted double-decay
PLOT_STACKED_SCANS = False  # plot all peak transmissions, with color gradient and with no t_wait offset

PLOT_SINGLE_SCAN = False  # plot an individual oscilloscope scan (for troubleshooting)
PLOT_SINGLE_SCAN_OD = False #all oscilloscope scans individually, in OD

PLOT_SINGLE_SCAN_HOLES = False  # plot an individual transmission scan, with fitted hole shapes
# convert holes to OD?

PLOT_LINEWIDTHS = True  # plot fitted linewidth of the hole transmission as a function of time
PLOT_BASELINE = False  # plot fitted transmission baseline as a function of time
PLOT_AREA = True  # plot fitted area of hole as function of time

PLOT_CONTRAST_OD = True

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100): #make cool colors
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


# fit functions
def decay_double(x, amp_fast, amp_slow, tau_fast, tau_slow, offset): #SUM OF TWO EXPONENTIALS plus linear offset
    return amp_fast * np.exp(-x / tau_fast) + amp_slow * np.exp(-x / tau_slow) + offset

def decay_single(x, amp_fast, tau_fast, offset): 
    return amp_fast * np.exp(-x / tau_fast)  + offset

def decay_double_log(x, amp_fast, amp_slow, tau_fast, tau_slow, offset): #e^(decay_double)
    return np.exp(amp_fast * np.exp(-x / tau_fast) + amp_slow * np.exp(-x / tau_slow) + offset)


"""
FILE PROCESSING
"""

print("Gathering files...")

# locate all files
csv_files = glob.glob('*/TEK0000.CSV', recursive=True, root_dir=DATA_DIR)
csv_files_freq = glob.glob('*/TEK0001.CSV', recursive=True, root_dir=DATA_DIR)
csv_paths = [os.path.join(DATA_DIR, file) for file in csv_files]
csv_paths_freq = [os.path.join(DATA_DIR, file) for file in csv_files_freq]

del csv_files[10]
del csv_files_freq[10]
del csv_paths[10]
del csv_paths_freq[10]


# read timing
t_wait = np.zeros(len(csv_files))
for i, path in enumerate(csv_files):
    path = os.path.normpath(path).split(os.sep)
    wait_str = path[-2]
    wait_str = wait_str.replace('p', '.')
    wait_str = wait_str[:-2]  # remove "ms" (only for time case)
    t_wait[i] = float(wait_str)

if LOG_CMAP:
    t_wait_log = np.log10(t_wait) #probably not usable in current setup

# sort
csv_paths = [path for _, path in sorted(zip(t_wait, csv_paths))]
csv_paths_freq = [path for _, path in sorted(zip(t_wait, csv_paths_freq))]
t_wait.sort()

if LOG_CMAP:
    t_wait_log.sort()

# read csvs
dfs = [pd.read_csv(path, names=TEK_HEADER) for path in csv_paths]
dfs_freq = [pd.read_csv(path, names=TEK_HEADER) for path in csv_paths_freq]
print(f"Found {len(dfs)} data files.")
print(f"Found {len(dfs_freq)} frequency files.")

# data for background
if PLOT_BG:
    bg_path = BG_DIR + "/TEK0000.CSV"
    bg_path_freq = BG_DIR + "/TEK0001.CSV"
    df_bg = pd.read_csv(bg_path, names=TEK_HEADER)
    df_bg_freq = pd.read_csv(bg_path_freq, names=TEK_HEADER)


"""
DATA PROCESSING
"""

print("Gathering transmission peaks and background...")

#Obtain background transmission
bg_transmission = df_bg["Volts"]
bg_transmission = (bg_transmission / GAIN) * 1e9  # convert to nW

max_bg = max(bg_transmission)
min_bg = min(bg_transmission)

# read starting times, peaks, and single scan
all_scan_midpoints = []  # note: this is the INDEX of the step in the array
all_scan_start = [] #all start indexes, all stop indexes
all_scan_stop = []
all_scan_transmission = []
all_scan_od = []
all_scan_freq = []

#all_scan_center_times = [] #same as all_peak_times_combine
max_trans = 0
for df, df_freq in zip(dfs, dfs_freq):
    # falling edge case
    if df_freq["Volts"].iloc[-1] < df_freq["Volts"][0]:
        scan_edge = [idx for idx in range(1, len(df_freq["Volts"]))
                     if df_freq["Volts"][idx] - df_freq["Volts"][idx-1] < -EDGE_THRESH]
    else:
        scan_edge = [idx for idx in range(1, len(df_freq["Volts"]))
                     if df_freq["Volts"][idx] - df_freq["Volts"][idx - 1] > EDGE_THRESH]
    if len(scan_edge) > 1:
        warnings.warn("Multiple scan edges found, defaulting to first.")
    center_idx = scan_edge[0]
    all_scan_midpoints.append(center_idx) #rough location of peak!

    time_arr = df_freq["Seconds"]
    center_time = time_arr[center_idx]
    start_time = np.round(center_time - 0.5*SCAN_TIME, 7)
    stop_time = np.round(center_time + 0.5*SCAN_TIME, 7)

    # TODO: why is this necessary?
    #start_time += 0.0000002
    #stop_time += 0.0000002

    start_idx = np.where(time_arr == start_time)[0][0]
    stop_idx = np.where(time_arr == stop_time)[0][0]
    all_scan_start.append(start_idx)
    all_scan_stop.append(stop_idx)

    #all_scan_center_times.append(center_time)

    transmission = df["Volts"][start_idx:stop_idx]
    transmission = (transmission / GAIN) * 1e9  # convert to nW
    all_scan_transmission.append(transmission)
    freq = np.linspace(-SCAN_RANGE/2, SCAN_RANGE/2, stop_idx-start_idx)
    all_scan_freq.append(freq) #important for PLOT_OD (where x-axis is in frequency)

    #Select correct background transmission indices
    bg_trans = bg_transmission[start_idx:stop_idx]

# calculate OD using background (off-res) scan (for now, let's not do this, and just reproduce 11_23 data)
for trans in all_scan_transmission:
    trans_arr = np.array(trans)
    bg_trans_arr = np.array(bg_trans)

    all_scan_od.append(np.log(bg_trans_arr / trans_arr))

# accumulate all peaks (remains from 11_03 hole_decay)
#but, need to change some variable names
all_peaks = [] #indices
all_peaks_combine = []
all_amps_combine = []
all_times_combine = []

all_OD_peaks_combine = []
all_OD_times_combine = []


for i, df in enumerate(dfs): #7/17: adjusted code since peaks do not occur RIGHT at scan edge
    #start_idx = all_starts[i] #start with first scan edge
    time = df["Seconds"].copy() #make a copy of the dataframe (as to not edit the old one)
    time += (t_wait[i]/1e3 - time[all_scan_start[i]])  #convert to seconds? ms? for now, DON'T ADD OFFSET

    ##OD_time = time[all_scan_start[i]:] #for purpose of finding index corresponding to OD peak, cut time to only include indices following all_scan_start[i]

    #EDIT below line of code later: perhaps extract OD?
    #peak_amps = (peak_heights / all_mins[i]).tolist() #get amplitudes of peaks, by diving voltage/height by minimum transmission of spectrum
    #peak_heights = peak_heights.tolist() #convert everything from pandas series to list
    #peak_times = peak_times.tolist() #NOTE: peak times are now defined differently! same peak indices, but "time" list redefined from raw oscilloscope time reading, to raw reading + t_wait - time at start index (basically set first time value to = t_wait)

    #Find voltage peaks
    peaks = find_peaks(df["Volts"], #using scipy "find_peaks" function, and only taking the x-value of the obtained peaks
                       prominence=PROMINENCE[i], distance=DISTANCE)[0] #prominence: increase from adjacent point, #distance: minimum distance between neighboring peaks

    print('peaks number ' + str(i+1))
    if len(peaks) != 1:
        print('Warning: found wrong number of peaks. Got ' + str(len(peaks)))
    
    peak_heights = df["Volts"][peaks] #obtain voltages corresponding to peaks
    peak_times = [time[peaks]] #get times of the peaks


    all_peaks.append(peaks) #all_peaks is just the peak indices (right now it's not in our code)
    all_peaks_combine += [peak_heights] #all_peaks_combine gives the voltages of the peaks
    #all_amps_combine += peak_amps
    all_times_combine += peak_times #only need because time converted by offset
    ##bg_peak_height = df_bg["Volts"].iloc[peaks] #not an actual background peak, just the background voltage at this particular signal peak
    ##all_OD_peaks_combine += [np.log(np.array(bg_peak_height) / np.array(peak_heights))] #take ratio of voltages (equivalent to ratio of transmissions)

    #Find OD peaks
    OD_peaks = find_peaks(-1*all_scan_od[i], #using scipy "find_peaks" function, and only taking the x-value of the obtained peaks
                       prominence=0.2, distance=DISTANCE)[0] #prominence: increase from adjacent point, #distance: minimum distance between neighboring peaks
    
    print('Od time:  ' + str(i+1))
    if len(OD_peaks) != 1:
        print('Warning: found wrong number of OD peaks. Got ' + str(len(OD_peaks)))

    all_OD_peaks_combine += [all_scan_od[i][OD_peaks]] #really, this peaks are TROUGHS due to holeburning

    OD_time = time[OD_peaks + all_scan_start[i]] #shift index by all_scan_start (because time includes all indices)
    all_OD_times_combine += [OD_time]


# fitting #of all holes together (decay of holes)
#there's only one hole for each case
print("Fitting hole peak decay...")

print(t_wait_log)
#0-3: -1 to 0

#4-11: 0 to 1

#12-16: 1 to 2

#17-18: 2 to 3

model = Model(decay_double_log) #utilize lmfit: choose log of double exponential decay
params = Parameters()
params.add('amp_fast', value=0.2, min=0) #5 best-fit parameters
params.add('amp_slow', value=0.2, min=0)
params.add('tau_fast', value=0.0005, min=0)
params.add('tau_slow', value=1, min=0)
params.add('offset', value=0)
result = model.fit(all_peaks_combine, params=params, x=all_times_combine) #fit to peak decay over time #x-values: times #y-values: voltages of peaks
print("")
print("FIT REPORT (peak height)")
print(result.fit_report())

# fitting #of all holes together (decay of holes)
#there's only one hole for each case
print("Fitting OD hole peak decay...")

model2 = Model(decay_double) #utilize lmfit: choose double exponential decay (NOT log, already coverted to log via OD)
#model2 = VoigtModel()
params2 = Parameters()
params2.add('amp_fast', value=0.2) #5 best-fit parameters
params2.add('amp_slow', value=0.2)
params2.add('tau_fast', value=0.0005, min=0)
params2.add('tau_slow', value=1, min=0)
params2.add('offset', value=0)
result2 = model2.fit(all_OD_peaks_combine, params=params2, x=all_OD_times_combine) #fit to peak decay over time #x-values: times #y-values: voltages of peaks
print("")
print("FIT REPORT (OD peak height)")
print(result2.fit_report())


# do fitting of individual holes: one per spectrum
#7/22 edit: convert to OD
print("")
print("Fitting individual holes...")
# model = LorentzianModel() + LinearModel()
model = VoigtModel() + LinearModel() #model is sum of voigt and linear

#model = VoigtModel(prefix='bg_') - VoigtModel(prefix='hole_') + LinearModel() #7/22 temporary change
all_hole_times_2d = [] #no need to be 2D, if just one hole per case
all_hole_centers_2d = []
all_hole_results_2d = []
for i, df in enumerate(dfs):
    print(f"\tFitting holes for scan {i+1}/{len(t_wait)}")
    hole_times = []
    hole_centers = []
    hole_results = []
    #for j, (start_idx, end_idx) in enumerate(zip(all_scan_edges[i][:-1], #take two adjacent scan edge elements (call the indices start and end)
    #                                             all_scan_edges[i][1:])):
        
    #for j, (df, start_idx, end_idx) in enumerate(zip(dfs, all_scan_start, all_scan_stop)):

    time = df["Seconds"][all_scan_start[i]:all_scan_stop[i]]
    hole_times.append(time) #extract all the time readings for which a hole is occurring (all the times between scan edges)

    #trans_data = df["Volts"][all_scan_start[i]:all_scan_stop[i]] #extract transmission readings for hole
    center_guess = float(time[all_scan_midpoints[i]]) #take guess of center of hole to just be location of midpoint (should be somewhat accurate)

    #sigma_guess = 0.00005
    sigma_guess = 1e-5

    amplitude_guess = -1e-6 ##JUL 22 update

    slope_guess = 100 ##
    intercept_guess = 1.5 ##
   
    params = model.make_params()
    params['sigma'].set(min=sigma_guess)

    ##7/25 guesses, may not work
    slope_guess = 1
    intercept_guess = 0
    sigma_guess = 0.00005
    amplitude_guess = 1

    #if LOG_SCALE:
        #result_hole = model.fit(np.log(trans_data), x=time, #fit to log of data, if plotting on log scale
                                #center=center_guess, sigma=sigma_guess)
    #else:
        #result_hole = model.fit(trans_data, x=time,
                                #center=center_guess, sigma=sigma_guess)

    result_hole = model.fit(all_scan_od[i], x=time, center=center_guess, sigma=sigma_guess, amplitude=amplitude_guess, 
                            slope=slope_guess, intercept=intercept_guess)

    hole_results.append(result_hole) #add hole fit results to list
    #result_hole.params.pretty_print(oneline=True)
        
    hole_centers.append(result_hole.params['center'].value) #add hole center location to list

        # # convert linewidth to frequency
        # width_time = result_hole.params['fwhm'].value  # unit: seconds
        # error_time = result_hole.params['fwhm'].stderr  # unit: seconds
        # scaling = SCAN_RANGE / (max(time) - min(time))
        # width = width_time * scaling  # unit: MHz
        # try:
        #     error = error_time * scaling  # unit: MHz
        # except TypeError:
        #     print("Failed to get hole params.")
        #     print("Fit report:")
        #     print(result_hole.fit_report())
        #     # raise Exception()
        # linewidths.append(width)
        # errors.append(error)

    all_hole_times_2d.append(hole_times) #2D array: each element in list is another list, of hole itmes for that spectrum
    all_hole_centers_2d.append(hole_centers) #in 6-20 case, no real need to be 2D.
    all_hole_results_2d.append(hole_results) #cuz only one hole center and result for each case

print("")
print("FIT REPORT (first hole fitting)")
print(all_hole_results_2d[-1][0].fit_report())

# LINEWIDTHS_TO_PRINT = -1
# print("")
# print(f"Linewidths (FWHM) for t_wait = {t_wait[LINEWIDTHS_TO_PRINT]}")
# for i, (lw, err) in enumerate(zip(all_hole_linewidth[LINEWIDTHS_TO_PRINT],
#                                   all_hole_error[LINEWIDTHS_TO_PRINT])):
#     print(f"\t{i+1}: {lw} +/- {err} MHz")
#
# HEIGHTS_TO_PRINT = -7
# print("")
# print(f"Fitted heights for t_wait = {t_wait[HEIGHTS_TO_PRINT]}")
# for i, (h, err) in enumerate(zip(all_hole_amplitudes[HEIGHTS_TO_PRINT],
#                                   all_hole_amp_error[HEIGHTS_TO_PRINT])):
#     print(f"\t{i+1}: {h} +/- {err} (A.U.)")

# reshape all hole data
all_hole_times = [] #all_peaks_combine
all_hole_results = [] #already have all_hole_results_2d
all_hole_centers = [] #already have center indices
for i, df in enumerate(dfs):
    start_idx = all_scan_start[i]
    time_start = df["Seconds"][start_idx]
    # time += (t_wait[i]/1e3 - time[start_idx])  # add offset
    centers = np.array(all_hole_centers_2d[i])
    centers -= time_start #adjust center by time_start
    centers += (t_wait[i] / 1e3) #adjust center by wait time
    centers = centers.tolist() 

    all_hole_times += all_hole_times_2d[i] #combine/accumulate hole data
    all_hole_results += all_hole_results_2d[i]
    all_hole_centers += centers #combine/accumulate hole center times

# fit double exponential of hole height
print("")
print("Fitting hole height...")
model = Model(decay_double)
# model = ExponentialModel() + ConstantModel()
params = Parameters()
params.add('amp_fast', value=0.45, min=0)
params.add('amp_slow', value=0.1, min=0)
params.add('tau_fast', value=0.005, min=0)
params.add('tau_slow', value=10, min=0)
params.add('offset', value=0)
# result_fit_height = model.fit(all_amplitudes_combine, x=all_centers_combine)
print("")
print("FIT REPORT (fitted peak height)")
# print(result_fit_height.fit_report())

# fit double exponential of hole area (SAME AS BEST-FIT VOIGT AMPLITUDE)
print("")
print("Fitting hole area...")
model_area = Model(decay_double)

#model_area = Model(decay_single)

params = Parameters()
params.add('amp_fast', value=-0.4)
params.add('amp_slow', value=-0.1)
params.add('tau_fast', value=0.005, min=0)
params.add('tau_slow', value=10, min=0)
params.add('offset', value=0)
result_fit_area = model_area.fit(list(map(lambda x: x.params['amplitude'].value, all_hole_results[4:])), x=all_hole_centers[4:], params=params)
print("")
print("FIT REPORT (fitted peak area)")
print(result_fit_area.fit_report())

# fit exponential decay of background T_0
bg = list(map(lambda x: x.params['intercept'].value, all_hole_results))
print("")
print("Fitting hole background decay...")
model_bg = ExponentialModel() + ConstantModel()
result_bg = model_bg.fit(bg, x=all_hole_centers)
print("")
print("FIT REPORT (background decay)")
print(result_bg.fit_report())


"""
PLOTTING
"""


##PLOT_OD from adjust t_wait file. Change x-axis to frequency

lines = []

if PLOT_OD:
    plot_lines = all_scan_od
#else:
    #plot_lines = all_scan_transmission

    for freq, trans in zip(all_scan_freq, plot_lines):
        line = np.column_stack((freq, trans))
        lines.append(line)

    cmap = truncate_colormap(CMAP, CMAP_OFFSET, 1)
    line_coll = LineCollection(lines, cmap=cmap)
    if LOG_CMAP:
        line_coll.set_array(t_wait_log)
        line_coll.set_clim(min(t_wait_log), max(t_wait_log))
    else:
        line_coll.set_array(t_wait)
        line_coll.set_clim(0, max(t_wait))

    fig, ax = plt.subplots(figsize=(6, 4))

    im = ax.add_collection(line_coll, autolim=True)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True)

    # labeling
    ax.set_xlabel("Detuning (MHz)")
    #if PLOT_OD:
    ax.set_ylabel("Optical Depth")
    #else:
        #ax.set_ylabel("Transmission (nW)")
    ax.set_title(rf"Pump Time Change - 7/23")

    plt.tight_layout()

    # add colorbar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.82, 0.15, 0.02, 0.7])
    cb = fig.colorbar(im, cax=cbar_ax)
    if LOG_CMAP:
        cb.set_label(r"Log Pump Time $T_{pump}$ (ms)")
    else:
        cb.set_label(r"Time $T_{pump}$ (ms)")
    # axcb = fig.colorbar(line_coll_low, ax=ax1)
    # axcb.set_label("Pump Amplitude")
    # axcb = fig.colorbar(line_coll_high, ax=ax2)
    # axcb.set_label("Pump Amplitude")

    plt.show()

#0-3: -1 to 0

#4-11: 0 to 1

#12-16: 1 to 2

#17-18: 2 to 3

if PLOT_OD_SPLIT:
    fig, ax = plt.subplots(1, 2, figsize=(12, 8))
    axs = ax.ravel()

    plot_lines = all_scan_od

    for freq, trans in zip(all_scan_freq, plot_lines): #create lines
        line = np.column_stack((freq, trans))
        lines.append(line)
    
    cmap = truncate_colormap(CMAP, CMAP_OFFSET, 1)
    line_coll = LineCollection(lines, cmap=cmap)

    if LOG_CMAP:
        line_coll.set_array(t_wait_log)
        line_coll.set_clim(min(t_wait_log), max(t_wait_log))
    else:
        line_coll.set_array(t_wait)
        line_coll.set_clim(0, max(t_wait))

    indices = [0, 9, 18] #lower and upper bounds used for the 4 subplot LISTS (i.e. upper indices off by 1)

    if LOG_CMAP:
        t_wait_log_range = max(t_wait_log) - min(t_wait_log) #normalize range of t_wait_log values

    else:
        t_wait_log_range = max(t_wait) - min(t_wait) #normalize range of t_wait_log values


    for i in range(2):

        #Split cmap
        #new_cmap = truncate_colormap(CMAP, CMAP_OFFSET+(i*0.7/4), CMAP_OFFSET+((i+1)*0.7/4))

        #Divide into 20 (for all 20 t_wait_log values)
        #i.e. Normalize based on passed t_wait_log values

        #This sort of works...problem is it considers 20 things as equally spaced
        #new_cmap = truncate_colormap(CMAP, CMAP_OFFSET+(lower_tier[i]*0.7/20), CMAP_OFFSET+((upper_tier[i-1])*0.7/20))

        if LOG_CMAP:
            new_cmap = truncate_colormap(CMAP, CMAP_OFFSET + (t_wait_log[indices[i]] - min(t_wait_log)) / t_wait_log_range * (1-CMAP_OFFSET),  
                                        CMAP_OFFSET + (t_wait_log[indices[i+1]-1] - min(t_wait_log)) / t_wait_log_range * (1-CMAP_OFFSET))
            
        else:
            new_cmap = truncate_colormap(CMAP, CMAP_OFFSET + (t_wait[indices[i]] - min(t_wait)) / t_wait_log_range * (1-CMAP_OFFSET),  
                                        CMAP_OFFSET + (t_wait[indices[i+1]-1] - min(t_wait)) / t_wait_log_range * (1-CMAP_OFFSET))
            

        #new_cmap = truncate_colormap(CMAP, CMAP_OFFSET+(lower_tier*0.7/20), CMAP_OFFSET+((upper_tier)*0.7/20))

        #original_cmap = cm.get_cmap(cmap)
        #colors = original_cmap(np.linspace(0, 1, 256))
        #new_cmap = ListedColormap(colors)

        #Split line collection into segments
        segments = line_coll.get_segments()
        
        subset = segments[indices[i]:indices[i+1]] #upper bound not included in list
        line_coll_subset = LineCollection(subset, cmap=new_cmap) #set to new_cmap

        if LOG_CMAP:
            line_coll_subset.set_array(t_wait_log[indices[i]:indices[i+1]])
            line_coll_subset.set_clim(min(t_wait_log[indices[i]:indices[i+1]]), max(t_wait_log[indices[i]:indices[i+1]]))
        else:
            line_coll_subset.set_array(t_wait[indices[i]:indices[i+1]])
            line_coll_subset.set_clim(min(t_wait[indices[i]:indices[i+1]]), max(t_wait[indices[i]:indices[i+1]]))

        im = axs[i].add_collection(line_coll_subset, autolim=True)
        axs[i].set_xlim(xlim)
        axs[i].set_ylim(ylim)


        #axs[i].set_ylim(0, max(all_scan_od[indices[i]])+0.1)

        

        """if i < 3: #set unit of subplot title to ms or s
            axs[i].set_title('$T_{wait} = $' + str(10**(i-1)) + ' to ' + str(10**i) + ' ms')

        elif i >= 3:
            axs[i].set_title('$T_{wait} = $' + str(10**(i-1-3)) + ' to ' + str(10**(i-3)) + ' s')"""
        

        #axs[i].set_title('$T_{pump} = $ %0.3f to %0.3f ms' % (t_wait[indices[i]], t_wait[indices[i+1]-1]))

        axs[i].set_title('$T_{pump} = $ ' + f'{t_wait[indices[i]]:.4g}' + ' to ' + f'{t_wait[indices[i+1]-1]:.4g}' + ' ms')

        #axs[i].set_title('$T_{pump} = $  + f{t_wait[indices[i]]:.4g} to f{t_wait[indices[i+1]-1]:.4g} ms')

        axs[i].grid(True)

    
    # labeling
    fig.supxlabel("Detuning (MHz)")
    #if PLOT_OD:
    fig.supylabel("Optical Depth")
    #else:
        #ax.set_ylabel("Transmission (nW)")
    fig.suptitle(rf"Pump Time Change - 7/23")

    #plt.tight_layout()

    # add colorbar
    #im_full = axs[0].add_collection(line_coll, autolim=True) #use FULL colobar, not a split one
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.82, 0.15, 0.02, 0.7])

    
    #cb = fig.colorbar(im_full, cax=cbar_ax)
    if LOG_CMAP:
        cb = fig.colorbar(cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=min(t_wait_log), vmax=max(t_wait_log), clip=False), cmap=cmap), cax=cbar_ax)
        cb.set_label(r"Log Pump Time $T_{pump}$ (ms)")
    else:
        cb = fig.colorbar(cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=min(t_wait), vmax=max(t_wait), clip=False), cmap=cmap), cax=cbar_ax)
        cb.set_label(r"Pump Time $T_{pump}$ (ms)")
    # axcb = fig.colorbar(line_coll_low, ax=ax1)
    # axcb.set_label("Pump Amplitude")
    # axcb = fig.colorbar(line_coll_high, ax=ax2)
    # axcb.set_label("Pump Amplitude")"""

    plt.show()


# for looking at all scans
if PLOT_ALL_SCANS: #plot all the transmission values found at a particular wait time, for all wait times (get vertical lines spaced apart)
    color = 'tab:blue'
    for i, df in enumerate(dfs):
        start_idx = all_scan_start[i]
        time = df["Seconds"][start_idx:]
        transmission = df["Volts"][start_idx:]

        # time *= 1e3  # convert to ms
        time += (t_wait[i]/1e3 - time[start_idx])  # add offset

        if i == 0:
            plt.plot(time, transmission, label="Transmission", color=color)
        else:
            plt.plot(time, transmission, color=color)

    #if PLOT_BG:
        #plt.fill_between(xlim_all_plots, max_bg, min_bg, label="Background",
                         #color='tab:gray', alpha=0.2)

    #plt.xlim(xlim_all_plots)
    plt.title("Hole Transmission Decay (6A B-Field) - 6/19")
    plt.xlabel("Time (s)")
    plt.ylabel("Transmission (A.U.)")
    plt.legend()
    plt.grid('on')

    plt.tight_layout()
    plt.show()


# for looking at all peaks + fit
if PLOT_ALL_PEAKS:
    fig, ax = plt.subplots()
    # color = 'tab:blue'
    # The color below comes from the B-field scan over all transitions in February
    # see previous plotting scripts for its determination.
    color = (0.0, 0.3544953298505101, 0.14229911572472131)

    ax.loglog(all_times_combine, all_peaks_combine, 
                 'o', color=color, label='Data')
    ax.loglog(all_times_combine, result.best_fit,
                 'k--', label='Fit')
    if PLOT_DECAY:
        for i, df in enumerate(dfs):
            time = df["Seconds"][all_scan_start[i]:all_scan_stop[i]]
            transmission = df["Volts"][all_scan_start[i]:all_scan_stop[i]]
            

            # time *= 1e3  # convert to ms
            time += (t_wait[i] / 1e3 - time[all_scan_start[i]])  # add offset

            if i == 0:
                ax.semilogy(time, transmission, label="Transmission",
                             color=color, alpha=0.2)
            else:
                ax.semilogy(time, transmission,
                             color=color, alpha=0.2)
                
    ax.set_title("Hole Transmission Decay - 6/20/24 Data")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Transmission (A.U.)")
    ax.legend()
    ax.grid('on')

    fig.tight_layout()
    fig.show()

    plt.show()
                

                
if PLOT_ALL_PEAKS_OD: #same as plot_all_peaks, but OD instead of raw transmission
    fig, ax = plt.subplots()
    # color = 'tab:blue'
    # The color below comes from the B-field scan over all transitions in February
    # see previous plotting scripts for its determination.
    color = (0.0, 0.3544953298505101, 0.14229911572472131)

    ax.plot(all_OD_times_combine, all_OD_peaks_combine, 
                 'o', color=color, label='Data')
    ax.plot(all_OD_times_combine, result2.best_fit,
                 'k--', label='Fit')
    if PLOT_DECAY:
        for i, df in enumerate(dfs):
            time = df["Seconds"].copy()
            od = all_scan_od[i]

            time += (t_wait[i] / 1e3 - time[all_scan_start[i]])  # add offset

            if i == 0:
                ax.semilogx(time[all_scan_start[i]:all_scan_stop[i]], od, label="Optical Depth",
                             color=color, alpha=0.2)
            else:
                ax.semilogx(time[all_scan_start[i]:all_scan_stop[i]], od,
                             color=color, alpha=0.2)
                

    ax.set_title("Hole OD Decay - 7/23/24 Data")
    ax.set_xlabel("Pump Time (s)")
    ax.set_ylabel("Optical Depth")

    ax.legend()
    #ax.set_xscale('log') #set x-axis to be logarithmic. Keep y-axis the same (hence no loglog)
    ax.grid('on')
    fig.tight_layout
    plt.show()
                
    

    if PLOT_BG:
        print('')
        #ax.fill_between(xlim_all_plots, max_bg, min_bg, label="Background",
                         #color='tab:gray', alpha=0.2)

    # plt.xlim((-0.1, 0.6))
    plt.show()

    if PLOT_BG:
        #Do SECOND plot, but with OD as y-axis (instead of transmission)
        fig, ax = plt.subplots()

        ax.loglog(all_times_combine, all_peaks_combine, 
                 'o', color=color, label='OD Data')
        plt.show()

    # plt.savefig('output_figs/hole_decay/11_01_23/fit_all_decay_loglog.pdf')


# for looking at all fitted heights + fit
if PLOT_ALL_HEIGHTS:
    color = 'tab:purple'
    fig, ax = plt.subplots()

    def get_height(x):
        return x.params['height'].value

    def get_height_err(x):
        return x.params['height'].stderr

    #ax.errorbar(all_hole_centers, list(map(get_height, all_hole_results)),
                #yerr=list(map(get_height_err, all_hole_results)),
                #capsize=10, marker='o', linestyle='', color=color)
    # plt.semilogy(all_centers_combine, result_fit_height.best_fit,
    #              'k--', label='Fit')
    ax.set_xscale('log')

    ax.set_title("Hole Transmission Decay (6A B-Field)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Hole Height Fit (A.U.)")
    ax.grid('on')

    plt.tight_layout()
    plt.show()


# for looking at individual scan decay
#6/20 case: not totally relevant, as only one peak per scan
if PLOT_STACKED_SCANS:
    lines = []
    for i, df in enumerate(dfs):
        peak_amp = df["Volts"][all_peaks[i]]
        times = all_times_combine[i]
        # plt.semilogy(times, peak_amp, '-o')
        line = np.column_stack((times, peak_amp))
        lines.append(line)

    cmap = truncate_colormap(mpl.cm.Blues, 0.3, 1)
    line_coll = LineCollection(lines, cmap=cmap)
    scale = np.log10(t_wait / 1e3)  # convert to s
    line_coll.set_array(scale)

    fig, ax = plt.subplots()
    ax.add_collection(line_coll, autolim=True)
    ax.autoscale_view()
    ax.set_yscale('log')
    axcb = fig.colorbar(line_coll, ax=ax)
    axcb.set_label(r"$\log_{10}(T_{pump})$ (s)")

    ax.set_title("All Peak Transmission Values (6A B-Field)")
    ax.set_xlabel("Time within scan (s)")
    ax.set_ylabel("Transmission (A.U.)")
    ax.grid('on')

    plt.tight_layout()
    plt.show()


# for studying one scan
if PLOT_SINGLE_SCAN:
    print('YAHOO !?!?')
    print('\n\n\n')

    SCAN_TO_PLOT = 0

    fig, ax = plt.subplots()

    color1 = 'tab:blue'
    ax.plot(dfs[SCAN_TO_PLOT]["Seconds"], dfs[SCAN_TO_PLOT]["Volts"],
            color=color1)
    ax.plot(dfs[SCAN_TO_PLOT]["Seconds"][all_peaks[SCAN_TO_PLOT]],
            dfs[SCAN_TO_PLOT]["Volts"][all_peaks[SCAN_TO_PLOT]],
            'x', color=color1)

    """color2 = 'tab:orange'
    ax.plot(dfs_freq[SCAN_TO_PLOT]["Seconds"],
             dfs_freq[SCAN_TO_PLOT]["Volts"],
             color=color2)
    ax.plot(dfs_freq[SCAN_TO_PLOT]["Seconds"][all_peaks[SCAN_TO_PLOT]],
             dfs_freq[SCAN_TO_PLOT]["Volts"][all_peaks[SCAN_TO_PLOT]],
             'x', color=color2)"""
    
    color3 = 'tab:green' #plot background
    ax.plot(df_bg["Seconds"], df_bg["Volts"],
                color=color3)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Voltage')
    plt.tight_layout()
    plt.show()

if PLOT_SINGLE_SCAN_OD: #plot optical depth for one scan (actual signal, background not plotted)
    SCAN_TO_PLOT = 1

    for i, df in enumerate(dfs):
        SCAN_TO_PLOT = i

        time = df["Seconds"].copy() #make a copy of the dataframe (as to not edit the old one)
        time += (t_wait[i]/1e3 - time[all_scan_start[i]])

        fig, ax = plt.subplots()

        color3 = 'tab:green' #plot OD
    
        #Plot OD data at all times within relevant scanning range
        ax.plot(time[all_scan_start[SCAN_TO_PLOT]:all_scan_stop[SCAN_TO_PLOT]], all_scan_od[SCAN_TO_PLOT], color=color3)
        
        #Identify OD critical point; use this to troubleshoot peak finding
        #don't use all_OD_times_combine, use regular time

        #ax.plot((all_OD_times_combine[SCAN_TO_PLOT]+dfs[SCAN_TO_PLOT]["Seconds"][all_scan_start[i]])*1000, all_OD_peaks_combine[SCAN_TO_PLOT], 'x')

        ax.plot(all_OD_times_combine[SCAN_TO_PLOT], all_OD_peaks_combine[SCAN_TO_PLOT], 'x')
    
        ax.set_xlabel('Time')
        ax.set_ylabel('Optical Depth')
        plt.tight_layout()
        plt.show()





#print(zip(all_hole_times[i], all_hole_results[i]))
#print(type(all_hole_results))

#print(all_hole_results[0]) #need to access actual values

#print(all_hole_results[0].params['amplitude'])


#what is res?

# for studying individual hole fits
# 7/22 edit: plot OD, not transmission
if PLOT_SINGLE_SCAN_HOLES:
    for i in range(len(dfs)):
        color1 = 'tab:blue'
        """if LOG_SCALE:
            plt.plot(dfs[i]["Seconds"], #plot actual data
                     np.log(dfs[i]["Volts"]),
                     color=color1, label='Data')
            
        else:
            plt.plot(dfs[i]["Seconds"],
                     dfs[i]["Volts"],
                     color=color1, label='Data')"""
        
        #plt.plot(dfs[i]["Seconds"][all_scan_start[i]:all_scan_stop[i]], 
        #             all_scan_od[i],
        #             color=color1, label='Data')

        plt.plot(all_scan_freq[i], 
                     all_scan_od[i],
                     color=color1, label='Data')
        
        for j in range(len(all_hole_times_2d[i])): #should only be one per trial
            #plt.plot(all_hole_times_2d[i][j], all_hole_results_2d[i][j].best_fit, 'k--', label='Fit') #plot best-fit data (log taken care of)
            plt.plot(all_scan_freq[i], all_hole_results_2d[i][j].best_fit, 'k--', label='Fit') #plot best-fit data (log taken care of)
            

        #wait_time = t_wait[i]
        #plt.title(rf"Hole fitting ($t_{{wait}}$ = {wait_time} ms)")
        #plt.xlabel("Time (s)")
        plt.xlabel("Detuning (MHz)")
        plt.ylabel("Optical Depth")

        plt.title('7/23 Hole: $T_{pump} = $' + str(t_wait[i]) + ' ms')

        plt.grid("on")
        plt.legend()

        plt.tight_layout()
        plt.show()

# for studying fitted hole width
if PLOT_LINEWIDTHS:
    fig, ax = plt.subplots()

    def get_linewidth(x, time):
        width_time = x.params['fwhm'].value  # unit: seconds

        # convert linewidth to frequency
        scaling = SCAN_RANGE / (max(time) - min(time))
        width = width_time * scaling  # unit: MHz

        return width

    def get_linewidth_err(x, time):
        error_time = x.params['fwhm'].stderr  # unit: seconds

        # convert to frequency
        scaling = SCAN_RANGE / (max(time) - min(time))
        error = error_time * scaling  # unit: MHz

        return error

    ax.errorbar(all_hole_centers,
                list(map(get_linewidth, all_hole_results, all_hole_times)),
                yerr=list(map(get_linewidth_err, all_hole_results, all_hole_times)),
                capsize=10, marker='o', linestyle='', color='tab:blue')
    # ax2.plot(all_centers_combine, all_amplitudes_combine,
    #          marker='o', linestyle='', color='tab:purple')
    ax.set_xscale('log')

    ax.set_ylim((3, 8))

    if LOG_SCALE:
        title = "Hole Linewidth (FWHM, Log Scale) vs Pump Time - 7/23"
    else:
        title = "Hole Linewidth (FWHM) vs Pump Time"
    ax.set_title(title)
    ax.set_xlabel("Pump Time (s)")
    ax.set_ylabel("Linewidth (MHz)")
    ax.grid('on')

    plt.tight_layout()
    plt.show()


# for studying fitted hole baseline
if PLOT_BASELINE:
    fig, ax = plt.subplots()

    def get_bg(x):
        return x.params['intercept'].value

    def get_bg_err(x):
        return x.params['intercept'].stderr

    ax.errorbar(all_hole_centers, list(map(get_bg, all_hole_results)),
                yerr=list(map(get_bg_err, all_hole_results)),
                capsize=10, marker='o', linestyle='', color='tab:orange',
                label='Data')
    ax.plot(all_hole_centers, result_bg.best_fit,
            'k--', label='Fit')
    ax.set_xscale('log')

    ax.set_title("Hole Transmission Background versus Time")
    ax.set_xlabel("Time (s)")
    if LOG_SCALE:
        ax.set_ylabel(r"$\log(T_0)$ (A.U.)")
    else:
        ax.set_ylabel(r"$T_0$ (A.U.)")
    ax.grid('on')
    ax.legend()

    plt.tight_layout()
    plt.show()


# for studying fitted hole area
#this case only: remove first four points from fit
if PLOT_AREA:
    fig, ax = plt.subplots()

    def get_area(x):
        return abs(x.params['amplitude'].value)

    def get_area_err(x):
        return x.params['amplitude'].stderr

    ax.errorbar(all_hole_centers, list(map(get_area, all_hole_results)),
                yerr=list(map(get_area_err, all_hole_results)),
                capsize=10, marker='o', linestyle='', color='tab:red')
    ax.plot(all_hole_centers[4:], abs(result_fit_area.best_fit),
            '--k')
    ax.set_xscale('log')

    if LOG_SCALE:
        title = "Hole Area (Log Scale) versus Pump Time - " + str(date)
    else:
        title = "Hole Area versus Pump Time - " + str(date)
    ax.set_title(title)
    ax.set_xlabel("Pump Time (s)")
    ax.set_ylabel("Area (A.U.)")
    ax.grid('on')

    plt.tight_layout()
    plt.show()

if PLOT_CONTRAST_OD:
    fig, ax = plt.subplots()

    def get_height(x):
        return abs(x.params['height'].value)

    def get_height_err(x):
        return x.params['height'].stderr

    ax.errorbar(all_hole_centers, list(map(get_height, all_hole_results)),
                yerr=list(map(get_height_err, all_hole_results)),
                capsize=10, marker='o', linestyle='', color='tab:purple')
    #ax.plot(all_hole_centers, abs(result_fit_area.best_fit),
    #        '--k')
    ax.set_xscale('log')

    if LOG_SCALE:
        title = "Hole Contrast OD (Log Scale) vs Pump Time - " + str(date)
    else:
        title = "Hole Contrast OD vs Pump Time- " + str(date)
    ax.set_title(title)
    ax.set_xlabel("Pump Time (s)")
    ax.set_ylabel("Hole Contrast (A.U.)")
    ax.grid('on')

    plt.tight_layout()
    plt.show()    