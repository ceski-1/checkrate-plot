import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import os
import sys
import tkinter as tk
from tkinter import filedialog

# Plot config
time_lim = [0, 5]                   # seconds

interval_lim = [0, 20]              # milliseconds
interval_tick_step = [5, 1]         # milliseconds (major, minor)
interval_bin_step = 0.2             # milliseconds
interval_histogram_lim = [0, 4]     # min and max

gyro_noise_lim = [-2, 2]            # deg/s
gyro_noise_tick_step = [1, 0.5]     # deg/s (major, minor)
gyro_noise_bin_step = 0.1           # deg/s
gyro_noise_histogram_lim = [0, 6]   # min and max

accel_noise_lim = [-0.2, 0.2]       # m/s^2
accel_noise_tick_step = [0.1, 0.05] # m/s^2 (major, minor)
accel_noise_bin_step = 0.01         # m/s^2
accel_noise_histogram_lim = [0, 60] # min and max

noise_skip_time = 0.5               # seconds (for initial stick movement)

fig_noise_size = [1440, 720]        # pixels
fig_polling_size = [960, 720]       # pixels

# Font config
if os.name == "nt":
    font_family = "Arial"
else:
    font_family = "sans-serif"
plt.rc("font", family=font_family, size=10)
plt.rc("figure", titlesize=10)
plt.rc("axes", titlesize=10, labelsize=10)
plt.rc("xtick", labelsize=10)
plt.rc("ytick", labelsize=10)
plt.rc("legend", fontsize=10)

# Label config
figure_names = ["IMU Sensor Noise", "IMU Polling Consistency"]
window_desc = ["fig_imu_sensor_noise", "fig_imu_polling_consistency"]
sensor_names = ["Gyro", "Accel"]
gyro_titles = ["Pitch Rate", "Yaw Rate", "Roll Rate"]
accel_titles = ["Accel X", "Accel Y", "Accel Z"]
sensor_units = ["Turn Rate [deg/s]", "Acceleration [m/$s^{2}$]"]

# Constants
min_delta= np.uint64(1e9 / 10000)
max_delta = np.uint64(1e9 / 10)
epsilon = np.double(1e-6)
s_per_ns = np.double(1e-9)
ms_per_ns = np.double(1e-6)
ms_per_s = np.double(1000)
deg_per_rad = np.double(180) / np.double(math.pi)
# Switch controllers only
samples_per_packet = 3
samples_dup_window = samples_per_packet * 2

# Explicitly define the data types
log_data_types= {
    "ControllerName"        : "string",
    "LeftStickTime"         : "UInt64",
    "LeftStickX"            : "Int16",
    "LeftStickY"            : "Int16",
    "RightStickTime"        : "UInt64",
    "RightStickX"           : "Int16",
    "RightStickY"           : "Int16",
    "GyroPacketIndex"       : "UInt64",
    "GyroTime"              : "UInt64",
    "GyroPitch"             : "Float64",
    "GyroYaw"               : "Float64",
    "GyroRoll"              : "Float64",
    "AccelPacketIndex"      : "UInt64",
    "AccelTime"             : "UInt64",
    "AccelX"                : "Float64",
    "AccelY"                : "Float64",
    "AccelZ"                : "Float64",
    "LeftTouchpadTime"      : "UInt64",
    "LeftTouchpadTouch"     : "boolean",
    "LeftTouchpadX"         : "Float64",
    "LeftTouchpadY"         : "Float64",
    "LeftTouchpadPressure"  : "Float64",
    "RightTouchpadTime"     : "UInt64",
    "RightTouchpadTouch"    : "boolean",
    "RightTouchpadX"        : "Float64",
    "RightTouchpadY"        : "Float64",
    "RightTouchpadPressure" : "Float64",
}

def InitWorkingDir():
    app_dir = None
    # Is this a compiled executable or a script?
    if getattr(sys, "frozen", False):
        app_dir = os.path.dirname(sys.executable)
    elif __file__:
        app_dir = os.path.dirname(__file__)
    if app_dir:
        app_dir = os.path.abspath(app_dir)
        os.chdir(app_dir)

def GetLogFile():
    if log_file_entry.get() and os.path.isfile(log_file_entry.get()):
        initial_dir = os.path.dirname(log_file_entry.get())
    else:
        initial_dir = os.getcwd()

    filetypes = (("*.csv", "*.csv"), ("*.*", "*.*"))
    log_file = filedialog.askopenfilename(title="Open CSV file",
                                          initialdir=initial_dir,
                                          filetypes=filetypes)
    if log_file:
        log_file = os.path.normpath(log_file)
    return log_file

def CalcIntervalsBasic(timestamp):
    num_samples = len(timestamp)
    time = np.zeros(num_samples, dtype=np.double)
    interval = np.zeros(num_samples, dtype=np.double)

    count = 0
    last_timestamp = timestamp[0]

    for i in range(1, num_samples):
        delta = timestamp[i] - last_timestamp
        if min_delta <= delta <= max_delta:
            time[count] = timestamp[i] * s_per_ns
            interval[count] = delta * ms_per_ns
            count += 1

        last_timestamp = timestamp[i]

    time = time[0:count]
    interval = interval[0:count]

    return time, interval

def CalcIntervalsSwitch(timestamp, x, y, z):
    num_samples = len(timestamp)
    if num_samples < samples_dup_window:
        return np.double(0), np.double(0), np.double(0)

    time = np.empty(num_samples, dtype=np.double)
    interval = np.empty(num_samples, dtype=np.double)
    temp_data = np.empty((num_samples, 3), dtype=np.double)

    count = 0
    last_timestamp = timestamp[samples_per_packet - 1]
    num_found = 0
    unique_x = []
    unique_y = []
    unique_z = []

    for i in range(samples_dup_window - 1, num_samples, samples_per_packet):
        for j in range(samples_per_packet):
            a = i - j
            unique_sample = True
            for k in range(j + 1, samples_dup_window):
                b = i - k
                if (abs(x[a] - x[b]) < epsilon and
                    abs(y[a] - y[b]) < epsilon and
                    abs(z[a] - z[b]) < epsilon):
                    unique_sample = False
                    break
            if unique_sample:
                unique_x.append(x[a])
                unique_y.append(y[a])
                unique_z.append(z[a])
                num_found += 1

        if num_found < 1:
            continue

        delta = np.double(timestamp[i] - last_timestamp) / np.double(num_found)
        if min_delta <= delta <= max_delta:
            for j in range(num_found):
                time[count] = timestamp[i] * s_per_ns
                interval[count] = delta * ms_per_ns
                temp_data[count, 0] = unique_x[j]
                temp_data[count, 1] = unique_y[j]
                temp_data[count, 2] = unique_z[j]
                count += 1

        last_timestamp = timestamp[i]
        num_found = 0
        unique_x = []
        unique_y = []
        unique_z = []

    if count < 1:
        return np.double(0), np.double(0), np.double(0)

    time = time[0:count]
    interval = interval[0:count]
    data = np.empty((count, 3), dtype=np.double)
    for i in range(3):
        data[:, i] = temp_data[0:count, i]

    return time, interval, data

def CalcIntervals(timestamp, x, y, z):
    num_samples = len(timestamp)
    if num_samples < 1:
        return np.double(0), np.double(0), np.double(0)

    time = np.empty(num_samples, dtype=np.double)
    interval = np.empty(num_samples, dtype=np.double)
    temp_data = np.empty((num_samples, 3), dtype=np.double)

    count = 0
    last_timestamp = timestamp[0]
    last_x = x[0]
    last_y = y[0]
    last_z = z[0]

    for i in range(1, num_samples):
        if (abs(x[i] - last_x) < epsilon and
            abs(y[i] - last_y) < epsilon and
            abs(z[i] - last_z) < epsilon):
            continue

        delta = timestamp[i] - last_timestamp
        if min_delta <= delta <= max_delta:
            time[count] = timestamp[i] * s_per_ns
            interval[count] = delta * ms_per_ns
            temp_data[count, 0] = x[i]
            temp_data[count, 1] = y[i]
            temp_data[count, 2] = z[i]
            count += 1

        last_timestamp = timestamp[i]
        last_x = x[i]
        last_y = y[i]
        last_z = z[i]

    if count < 1:
        return np.double(0), np.double(0), np.double(0)

    time = time[0:count]
    interval = interval[0:count]
    data = np.empty((count, 3), dtype=np.double)
    for i in range(3):
        data[:, i] = temp_data[0:count, i]

    return time, interval, data

def IsSwitchController(controller_name, gyro_packet_idx):
    if not "switch" in controller_name.lower():
        return False
    num_packets_to_check = 10
    num_samples_per_packet = 3
    if len(gyro_packet_idx) < num_packets_to_check * num_samples_per_packet:
        return False
    row_idx = 0
    for packet_idx in range(num_packets_to_check):
        for sample_idx in range(num_samples_per_packet):
            if gyro_packet_idx[row_idx] != packet_idx:
                return False
            row_idx += 1
    return True

def StripNaN(x):
    x = x[~np.isnan(x)]
    return x

def CalcPDF(x, mean, std):
    return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * math.pi))

def NoiseFigurePlots(axs, sensor_idx, sensor_time, sensor_data, noise_lim, noise_tick_step, noise_bin_step, noise_histogram_lim, titles):
    col = sensor_idx * 2
    for i in range(3):
        # Scatter plot
        ax = axs[i, col]
        data = sensor_data[:, i]
        data -= np.mean(data)
        ax.scatter(sensor_time, data, s=1)

        ax.set_xlim(time_lim)
        ax.set_ylim(noise_lim)
        ax.yaxis.set_major_locator(MultipleLocator(noise_tick_step[0]))
        ax.yaxis.set_minor_locator(MultipleLocator(noise_tick_step[1]))
        ax.set_title(titles[i])
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(sensor_units[sensor_idx])

        # Distribution plot
        ax = axs[i, col + 1]
        data = sensor_data[:, i]
        count = len(data)
        # Trim the initial stick movement
        trim_data = np.empty(count, dtype=np.double)
        trim_count = 0
        for j in range(count):
            if noise_lim[0] <= data[j] <= noise_lim[1] and sensor_time[j] >= noise_skip_time:
                trim_data[trim_count] = data[j]
                trim_count += 1
        data = trim_data[0:trim_count]
        data -= np.mean(data)

        # Histogram
        bins = np.arange(noise_lim[0], noise_lim[1], noise_bin_step)
        ax.hist(data, bins=bins, range=noise_lim, density=True, edgecolor="black", linewidth=0.5, label="hist")

        # Probability density function
        std = np.std(data)
        pdf_x = np.linspace(noise_lim[0], noise_lim[1], 1000, dtype=np.double)
        pdf = CalcPDF(pdf_x, 0, std)
        ax.plot(pdf_x, pdf, label="pdf")

        ax.set_xlim(noise_lim)
        ax.xaxis.set_major_locator(MultipleLocator(noise_tick_step[0]))
        ax.xaxis.set_minor_locator(MultipleLocator(noise_tick_step[1]))
        ax.set_ylim(noise_histogram_lim)
        ax.set_title(f"{titles[i]} Distribution")
        ax.set_xlabel(sensor_units[sensor_idx])
        ax.set_ylabel("Probability Density")
        ax.legend(loc="upper right")

def PollingFigurePlots(axs, i, sensor_time, sensor_interval, basic_time, basic_interval):
    # Scatter plot
    ax = axs[i, 0]
    ax.scatter(sensor_time, sensor_interval, s=2, label=f"{sensor_names[i]} Interval", zorder=1)
    ax.scatter(basic_time, basic_interval, s=2, label="Polling Interval", zorder=0)

    mean_interval = np.mean(sensor_interval)
    std_interval = np.std(sensor_interval)
    mean_rate = ms_per_s / mean_interval
    rate_text = "{:s} Rate: {:.1f} Hz".format(sensor_names[i], mean_rate)
    interval_text = "{:s} Interval: {:.3f} ± {:.3f} ms".format(sensor_names[i], mean_interval, std_interval)

    ax.set_xlim(time_lim)
    ax.set_ylim(interval_lim)
    ax.yaxis.set_major_locator(MultipleLocator(interval_tick_step[0]))
    ax.yaxis.set_minor_locator(MultipleLocator(interval_tick_step[1]))
    ax.set_title(f"{rate_text}\n{interval_text}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Interval [ms]")
    ax.legend(loc="upper right")

    # Distribution plot
    ax = axs[i, 1]
    bins = np.arange(interval_lim[0], interval_lim[1], interval_bin_step)
    ax.hist(sensor_interval, bins=bins, range=interval_lim, density=True, edgecolor="black", linewidth=0.5)

    ax.set_xlim(interval_lim)
    ax.xaxis.set_major_locator(MultipleLocator(interval_tick_step[0]))
    ax.xaxis.set_minor_locator(MultipleLocator(interval_tick_step[1]))
    ax.set_ylim(interval_histogram_lim)
    ax.set_title(f"{sensor_names[i]} Distribution")
    ax.set_xlabel("Interval [ms]")
    ax.set_ylabel("Probability Density")

def ProcessData(log_file, figure_type):
    log_name = os.path.splitext(os.path.basename(log_file))[0]
    log_data = pd.read_csv(log_file, dtype=log_data_types)
    df = pd.DataFrame(log_data)

    if df["ControllerName"][0]:
        controller_name = str(df["ControllerName"][0])
    else:
        controller_name = "Unknown Controller"

    left_stick_timestamp = np.uint64(StripNaN(df["LeftStickTime"]))
    gyro_packet_idx = np.uint64(StripNaN(df["GyroPacketIndex"]))
    gyro_timestamp = np.uint64(StripNaN(df["GyroTime"]))
    gyro_pitch_rate = np.double(StripNaN(df["GyroPitch"]))
    gyro_yaw_rate = np.double(StripNaN(df["GyroYaw"]))
    gyro_roll_rate = np.double(StripNaN(df["GyroRoll"]))
    accel_timestamp = np.uint64(StripNaN(df["AccelTime"]))
    accel_x = np.double(StripNaN(df["AccelX"]))
    accel_y = np.double(StripNaN(df["AccelY"]))
    accel_z = np.double(StripNaN(df["AccelZ"]))

    # Shared plot title
    title = f"{log_name}\n{controller_name}"

    switch_controller = IsSwitchController(controller_name, gyro_packet_idx)
    if switch_controller:
        title += "\n(Note: Switch controller support is experimental)"

    if figure_type == 0:
        # IMU sensor noise figure
        if switch_controller:
            gyro_time, gyro_interval, gyro_data = CalcIntervalsSwitch(gyro_timestamp, gyro_pitch_rate, gyro_yaw_rate, gyro_roll_rate)
            accel_time, accel_interval, accel_data = CalcIntervalsSwitch(accel_timestamp, accel_x, accel_y, accel_z)
        else:
            gyro_time, gyro_interval, gyro_data = CalcIntervals(gyro_timestamp, gyro_pitch_rate, gyro_yaw_rate, gyro_roll_rate)
            accel_time, accel_interval, accel_data = CalcIntervals(accel_timestamp, accel_x, accel_y, accel_z)
        gyro_data *= deg_per_rad
        fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(fig_noise_size[0], fig_noise_size[1], "px"), layout="tight")
        fig.suptitle(f"{figure_names[0]}\n\n" + title)
        fig.canvas.manager.set_window_title(f"{log_name} {window_desc[0]}")
        NoiseFigurePlots(axs, 0, gyro_time, gyro_data, gyro_noise_lim, gyro_noise_tick_step, gyro_noise_bin_step, gyro_noise_histogram_lim, gyro_titles)
        NoiseFigurePlots(axs, 1, accel_time, accel_data, accel_noise_lim, accel_noise_tick_step, accel_noise_bin_step, accel_noise_histogram_lim, accel_titles)
    else:
        # IMU polling consistency figure
        if switch_controller:
            gyro_time, gyro_interval, gyro_data = CalcIntervalsSwitch(gyro_timestamp, gyro_pitch_rate, gyro_yaw_rate, gyro_roll_rate)
            accel_time, accel_interval, accel_data = CalcIntervalsSwitch(accel_timestamp, accel_x, accel_y, accel_z)
            gyro_time2, gyro_interval2 = CalcIntervalsBasic(left_stick_timestamp)
            accel_time2 = gyro_time2
            accel_interval2 = gyro_interval2
        else:
            gyro_time, gyro_interval, gyro_data = CalcIntervals(gyro_timestamp, gyro_pitch_rate, gyro_yaw_rate, gyro_roll_rate)
            accel_time, accel_interval, accel_data = CalcIntervals(accel_timestamp, accel_x, accel_y, accel_z)
            gyro_time2, gyro_interval2 = CalcIntervalsBasic(gyro_timestamp)
            accel_time2, accel_interval2 = CalcIntervalsBasic(accel_timestamp)
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(fig_polling_size[0], fig_polling_size[1], "px"), layout="tight")
        fig.suptitle(f"{figure_names[1]}\n\n" + title)
        fig.canvas.manager.set_window_title(f"{log_name} {window_desc[1]}")
        PollingFigurePlots(axs, 0, gyro_time, gyro_interval, gyro_time2, gyro_interval2)
        PollingFigurePlots(axs, 1, accel_time, accel_interval, accel_time2, accel_interval2)

    plt.show()

InitWorkingDir()

# Create window
root = tk.Tk()
root.title("checkrate-plot")
root.resizable(False, False)
row = 0

log_file_label = tk.Label(root, text="CSV File:")
log_file_label.grid(row=row, column=0, columnspan=4, padx=10, pady=5, sticky=tk.W)
row += 1

log_file_entry = tk.Entry(root, width=56)
log_file_entry.insert(0, "")
log_file_entry.grid(row=row, column=0, columnspan=3, padx=10, pady=5, sticky=tk.W)

def RunGetLogFile():
    log_file = GetLogFile()
    if log_file:
        log_file_entry.delete(0, tk.END)
        log_file_entry.insert(0, log_file)

log_file_button = tk.Button(root, text="Browse", command=RunGetLogFile, width=8)
log_file_button.grid(row=row, column=3, padx=10, pady=5, sticky=tk.E)
row += 1

figure_type_label = tk.Label(root, text="Figure Type:")
figure_type_label.grid(row=row, column=0, columnspan=4, padx=10, pady=5, sticky=tk.W)
row += 1

radio_text = [f"{figure_names[0]} (select this if the controller was stationary)",
              f"{figure_names[1]} (select this if the controller was moving)"]
radio_var = tk.IntVar(root, 0)

radio_button = []
for i in range(len(radio_text)):
    radio_button.append(tk.Radiobutton(root, text=radio_text[i], variable=radio_var, value=i))
    radio_button[i].grid(row=row, column=0, columnspan=4, padx=10, pady=5, sticky=tk.W)
    row += 1

def RunProcessData():
    if log_file_entry.get() and os.path.isfile(log_file_entry.get()):
        log_file = os.path.abspath(log_file_entry.get())
        ProcessData(log_file, radio_var.get())

run_button = tk.Button(root, text="Generate", command=RunProcessData, width=32, height=2)
run_button.grid(row=row, column=0, columnspan=4, padx=10, pady=10)

root.mainloop()
