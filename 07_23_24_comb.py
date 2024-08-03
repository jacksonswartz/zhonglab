#Alex Kolar's code, slightly modified; exact same as 7/23 comb

import glob
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as colors
import matplotlib.cm as cm
import warnings

date = "08/01"

#2.048ms probe time

# for data (3 datasets: one for each comb)
"""DATA_DIR = ("/Users/JacksonS/Documents/zhonglab"
            "/07_23_24/10mK_12Amp/pumpnprobe/tr2_196042p746GHz/comb")

BG_DIR = ("/Users/JacksonS/Documents/zhonglab"
            "/07_23_24/10mK_12Amp/probe/offres")"""

DATA_DIR = ("/Users/JacksonS/Documents/zhonglab"
            "/08_01_24/10mK_12Amp/pumpnprobe/tr2_196042p753GHz")

BG_DIR = ("/Users/JacksonS/Documents/zhonglab"
            "/08_01_24/10mK_12Amp/probe")


TEK_HEADER = ["ParamLabel", "ParamVal", "None", "Seconds", "Volts", "None2"]  # hard-coded from TEK oscilloscope
SCAN_RANGE = 50  # Unit: MHz
SCAN_TIME = 2.048e-3  # Unit: s
GAIN = 1e8  # Unit: V/W

EDGE_THRESH = 1  # For finding rising/falling edge of oscilloscope trigger

# for plotting
# plotting parameters
mpl.rcParams.update({'font.size': 12,
                     'figure.figsize': (8, 6)})

# plotting params
CMAP_OFFSET = 0.3 #
CMAP = cm.Greens
xlim = (-SCAN_RANGE/2, SCAN_RANGE/2) #x-axis set to match scanning range, where the center is on resonance
ylim = (0, 1)

PLOT_OD = True  # plot as optical depth
LOG_CMAP = True  # use log scale for colormap


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """Function to plot."""
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


"""
FILE PROCESSING
"""

print("Gathering files...")

# locate all files
#csv_files = glob.glob('TEK0000.CSV', recursive=True, root_dir=DATA_DIR) #remove the */ at beginning, since we don't dive any deeper

csv_files = []
csv_files.append(DATA_DIR + "/TEK0000.CSV")
csv_files.append(DATA_DIR + "/TEK0002.CSV")
csv_files.append(DATA_DIR + "/TEK0004.CSV")
csv_files.append(DATA_DIR + "/TEK0006.CSV")
csv_files.append(DATA_DIR + "/TEK0008.CSV")
csv_files.append(DATA_DIR + "/TEK0010.CSV")
csv_files.append(DATA_DIR + "/TEK0012.CSV")



#csv_files.append(DATA_DIR + "/TEK0002.CSV")


#csv_files = glob.glob('TEK0002.CSV', recursive=True, root_dir=DATA_DIR) #remove the */ at beginning, since we don't dive any deeper


#csv_files_freq = glob.glob('TEK0001.CSV', recursive=True, root_dir=DATA_DIR) #0001 file gives frequency

csv_files_freq = []
csv_files_freq.append(DATA_DIR + "/TEK0001.CSV")
csv_files_freq.append(DATA_DIR + "/TEK0003.CSV")
csv_files_freq.append(DATA_DIR + "/TEK0005.CSV")
csv_files_freq.append(DATA_DIR + "/TEK0007.CSV")
csv_files_freq.append(DATA_DIR + "/TEK0009.CSV")
csv_files_freq.append(DATA_DIR + "/TEK0011.CSV")
csv_files_freq.append(DATA_DIR + "/TEK0013.CSV")




csv_paths = [os.path.join(DATA_DIR, file) for file in csv_files] #make list of full paths for each csv file
csv_paths_freq = [os.path.join(DATA_DIR, file) for file in csv_files_freq]

# read csvs
dfs = [pd.read_csv(path, names=TEK_HEADER) for path in csv_paths] #read out each csv in list into pandas datarfame
dfs_freq = [pd.read_csv(path, names=TEK_HEADER) for path in csv_paths_freq]
print(f"Found {len(dfs)} data files.") #Print how many files we got (1 in 6/24 case)
print(f"Found {len(dfs_freq)} frequency files.")

# locate background file
bg_file = os.path.join(BG_DIR, "TEK0000.CSV") #no list: only one value, given by BG_DIR + TEK0000
bg_file_freq = os.path.join(BG_DIR, "TEK0001.CSV")
print("Reading background file...")
df_bg = pd.read_csv(bg_file, names=TEK_HEADER) #read out csv into dataframe
df_bg_freq = pd.read_csv(bg_file_freq, names=TEK_HEADER)


"""
DATA PROCESSING
"""

print("Gathering transmission peaks and background...")

# gather background data
# falling edge case
if df_bg_freq["Volts"].iloc[-1] < df_bg_freq["Volts"][0]: #if last recorded voltage LESS than FIRST voltage
    scan_edge = [idx for idx in range(1, len(df_bg_freq["Volts"])) #consider list of voltages as integer list of inidices (1,2,3...)
                 if df_bg_freq["Volts"][idx] - df_bg_freq["Volts"][idx-1] < -EDGE_THRESH] #only take into list the indices for which the decrease across adjacent voltages (at that index) GREATER than edge threshold (1)
else: #if last recorded voltage GREATER THAN/EQUAL TO first voltage
    scan_edge = [idx for idx in range(1, len(df_bg_freq["Volts"])) #same as before, but if higher values (post-trigger) are after lower values (pre-trigger)
                 if df_bg_freq["Volts"][idx] - df_bg_freq["Volts"][idx - 1] > EDGE_THRESH] #same as before: only take into list the indices for which the increase across adjacent voltages GREATER than edge threshold
if len(scan_edge) > 1:
    warnings.warn("Multiple scan edges found for background, defaulting to first.") #should only find one edge: one trigger
center_idx = scan_edge[0] #take the identified scan edge as the center: zero detuning
time_arr = df_bg_freq["Seconds"]
center_time = time_arr[center_idx]
start_time = np.round(center_time - 0.5*SCAN_TIME, 7) #start and stop times determined by putting center index time as the center
stop_time = np.round(center_time + 0.5*SCAN_TIME, 7) #thus, start time is half of the scan time BEFORE the center time

# TODO: why is this necessary?
#start_time += 0.0000002
#stop_time += 0.0000002
#match actual spectrum with start/stop times of background data (setting trigger placement center)
start_idx = np.where(time_arr == start_time)[0][0] #locate time in time_arr that corresponds to the start time. Gives list that contains index and specified output (None) so add[0][0] (take first value)
stop_idx = np.where(time_arr == stop_time)[0][0] #locate time in time_arr that corresponds to the stop time.
bg_transmission = df_bg["Volts"][start_idx:stop_idx] #only take voltage readings between the start and stop times
bg_transmission = (bg_transmission / GAIN) * 1e9  # convert to nW using Gain (units of V/W)

# read starting times, peaks, and single scan
all_scan_midpoints = []  # note: this is the INDEX of the step in the array
all_scan_start = []
all_scan_stop = []
all_scan_transmission = []
all_scan_od = []
all_scan_freq = []
max_trans = 0
for df, df_freq in zip(dfs, dfs_freq): #zip creates list of tuple pairs, one from each list #Why is this here? Seems redunant to find same indices as in background
    # falling edge case
    if df_freq["Volts"].iloc[-1] < df_freq["Volts"][0]: #same as before:find edge
        scan_edge = [idx for idx in range(1, len(df_freq["Volts"]))
                     if df_freq["Volts"][idx] - df_freq["Volts"][idx-1] < -EDGE_THRESH]
    else:
        scan_edge = [idx for idx in range(1, len(df_freq["Volts"]))
                     if df_freq["Volts"][idx] - df_freq["Volts"][idx - 1] > EDGE_THRESH]
    if len(scan_edge) > 1:
        warnings.warn("Multiple scan edges found, defaulting to first.")
    center_idx = scan_edge[0]
    all_scan_midpoints.append(center_idx)

    time_arr = df_freq["Seconds"]
    center_time = time_arr[center_idx]
    start_time = np.round(center_time - 0.5*SCAN_TIME, 7)
    stop_time = np.round(center_time + 0.5*SCAN_TIME, 7)


    start_idx = np.where(time_arr == start_time)[0][0]
    stop_idx = np.where(time_arr == stop_time)[0][0]
    all_scan_start.append(start_idx)
    all_scan_stop.append(stop_idx)

    transmission = df["Volts"][start_idx:stop_idx]
    transmission = (transmission / GAIN) * 1e9  # convert to nW
    all_scan_transmission.append(transmission)
    freq = np.linspace(-SCAN_RANGE/2, SCAN_RANGE/2, stop_idx-start_idx) #use scan_range to set frequency range. Keep spacing same as in raw data 
    all_scan_freq.append(freq)

# calculate OD using background (off-res) scan
for trans in all_scan_transmission: #only one element in list: but that element is a list itself
    trans_arr = np.array(trans) #convert to np array (works better for calculations)
    all_scan_od.append(np.log(bg_transmission / trans_arr)) #convert transmission to optical depth. Length of two arrays should be the same!


"""
PLOTTING
"""

colour = ['orange', 'green', 'purple']
label = ['Slide1', 'Slide2', 'Slide3', 'Slide4', 'Slide5', 'Slide6', 'Slide7']
ylimz = [[0.1, 0.3], [0.4, 0.7], [0.6, 1.0]]

if PLOT_OD: #if PLOT_OD = True, plot optical depth instead of raw transmission (requires background)
    plot_lines = all_scan_od
else:
    plot_lines = all_scan_transmission

lines = []

for freq, trans in zip(all_scan_freq, plot_lines):
    line = np.column_stack((freq, trans)) #create 2D array with frequency and transmission columns
    lines.append(line) #multiple lines in case of multiple trials


"""cmap = truncate_colormap(CMAP, CMAP_OFFSET, 1)
line_coll = LineCollection(lines, colors=colour)

print(lines)

fig, ax = plt.subplots(figsize=(6, 4))

im = ax.add_collection(line_coll, autolim=True)

ax.set_xlim(xlim)
ax.set_ylim((0.75, 1.25))

ax.grid(True)

# labeling
ax.set_xlabel("Detuning (MHz)")
if PLOT_OD:
    ax.set_ylabel("Optical Depth")
else:
    ax.set_ylabel("Transmission (nW)")
ax.set_title("AFC - " + date )

plt.tight_layout()

    # add colorbar
    #fig.subplots_adjust(right=0.8)
    #cbar_ax = fig.add_axes([0.82, 0.15, 0.02, 0.7])
    #cb = fig.colorbar(im, cax=cbar_ax)
    #if LOG_CMAP:
        #cb.set_label(r"Log Pump Time $T_{wait}$ (ms)")
    #else:
        #cb.set_label(r"Pump Time $T_{wait}$ (ms)")
    ##
    # axcb = fig.colorbar(line_coll_low, ax=ax1)
    # axcb.set_label("Pump Amplitude")
    # axcb = fig.colorbar(line_coll_high, ax=ax2)
    # axcb.set_label("Pump Amplitude")

plt.show()"""


for i in range(len(all_scan_freq)): #separate combs into different plots
    print('yay ' + str(i))
    cmap = truncate_colormap(CMAP, CMAP_OFFSET, 1)
    line_coll = LineCollection([lines[i]], colors='purple')

    print(lines)

    fig, ax = plt.subplots(figsize=(6, 4))

    im = ax.add_collection(line_coll, autolim=True)

    ax.set_xlim(xlim)
    ax.set_ylim((0.92, 1.12))

    ax.grid(True)

    # labeling
    ax.set_xlabel("Detuning (MHz)")
    if PLOT_OD:
        ax.set_ylabel("Optical Depth")
    else:
        ax.set_ylabel("Transmission (nW)")
    ax.set_title("AFC " + date + " - " + str(label[i]))

    plt.tight_layout()

    # add colorbar
    #fig.subplots_adjust(right=0.8)
    #cbar_ax = fig.add_axes([0.82, 0.15, 0.02, 0.7])
    #cb = fig.colorbar(im, cax=cbar_ax)
    #if LOG_CMAP:
        #cb.set_label(r"Log Pump Time $T_{wait}$ (ms)")
    #else:
        #cb.set_label(r"Pump Time $T_{wait}$ (ms)")
    ##
    # axcb = fig.colorbar(line_coll_low, ax=ax1)
    # axcb.set_label("Pump Amplitude")
    # axcb = fig.colorbar(line_coll_high, ax=ax2)
    # axcb.set_label("Pump Amplitude")"""

    plt.show()