import yaml
import numpy as np
import pandas as pd
import seaborn as sea
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import bagpy
from bagpy import bagreader
import os
from rosbag import Bag
import os.path
from data_visualisation.visualise_signal import match_time



class VisualiseSignal(object):
    """
    A class to visualise the data for GPS,
    Mobile Signal Strength from rosbag collected from Hatchgate
    """

    def __init__(self, data_path_, file_type_, bag_name, fig=None, ax=None):
        self.entries_list = None
        self.data_path = data_path_
        self.file_type = file_type_
        self.bag_name = bag_name
        self.INVALID_SIG = -999
        self.sig_max = 0
        self.sig_min = 0
        self.display = False

        self.df_gps_x = []
        self.df_gps_y = []

        self.heatmap_data = None
        self.heatmap_index = None
        self.heatmap_columns = None

        self.show_cbar = False
        if fig:
            self.fig = fig
        if ax:
            self.ax = ax
        else:
            self.fig, self.ax = plt.subplots(1, 1, figsize=(16, 9), sharex=False, sharey=False)

        if not os.path.exists(self.data_path + '/' + self.bag_name):
            self.merge_bag()

    def plot(self, generations):
        for g in generations:     # [2, 3, 4, 5]
            if g < 4:
                self.sig_max = -70
                self.sig_min = -140
            else:
                self.sig_max = 0
                self.sig_min = -20
            self.init_heatmap(self.bag_name, g)

    @staticmethod
    def close_fig():
        plt.close('all')

    def merge_bag(self):
        """
        Merge individual bags into a single bag
        """
        topics = ''
        total_included_count = 0
        total_skipped_count = 0

        self.get_folder_files()
        with Bag(self.data_path + '/' + self.bag_name, 'w') as o:
            for ifile in self.entries_list:
                ifile = self.data_path + '/' + ifile
                matchedtopics = []
                included_count = 0
                skipped_count = 0
                with Bag(ifile, 'r') as ib:
                    for topic, msg, t in ib:
                        # if any(fnmatchcase(topic, pattern) for pattern in topics):
                        #     if topic not in matchedtopics:
                        #         matchedtopics.append(topic)
                        o.write(topic, msg, t)
                        included_count += 1
                        # else:
                        #     skipped_count += 1
                total_included_count += included_count
                total_skipped_count += skipped_count

    def get_folder_files(self):
        entries_list = []
        with os.scandir(self.data_path) as entries:
            for entry in entries:
                if entry.is_file():
                    entries_list.append(entry.name)
        entries_list.sort()
        self.entries_list = entries_list

    def init_heatmap(self, bag_name, sig_g=2):
        """
        Plot heatmap of the signal
        """
        b = bagreader(self.data_path + '/' + bag_name)
        data_sig = b.message_by_topic('/monitors_router/data')
        data_gps = b.message_by_topic('/gps/filtered')
        print("File saved: {}".format(data_gps))
        print("File saved: {}".format(data_sig))

        df_signal = pd.read_csv(data_sig)
        df_gps = pd.read_csv(data_gps)

        match_idx = match_time(df_signal.Time, df_gps.Time)

        self.df_gps_x = []
        self.df_gps_y = []
        for idx in match_idx:
            self.df_gps_x.append(round(df_gps.latitude[idx] * 111139, 1))
            self.df_gps_y.append(round(df_gps.longitude[idx] * 111139, 1))

        max_x = round(max(self.df_gps_x))
        min_x = round(min(self.df_gps_x))
        max_y = round(max(self.df_gps_y))
        min_y = round(min(self.df_gps_y))
        data_sig2 = np.zeros((max_x - min_x,
                              max_y - min_y))

        if sig_g == 2:
            sig = df_signal.data_3
        elif sig_g == 3:
            sig = df_signal.data_4
        elif sig_g == 4:
            sig = df_signal.data_5
        elif sig_g == 5:
            sig = df_signal.data_6
        else:
            raise "Signal generation should be: 2, 3, 4, or 5"

        for i, x in enumerate(self.df_gps_x):
            for j, y in enumerate(self.df_gps_y):
                if i == j:
                    if sig[i] == self.INVALID_SIG:
                        data_sig2[round(x) - min_x - 1, round(y) - min_y - 1] = 0
                    else:
                        data_sig2[round(x) - min_x - 1, round(y) - min_y - 1] = sig[i]
                    break

        df_ht = pd.DataFrame(data_sig2,
                             index=np.linspace(min_x,
                                               max_x - 1,
                                               max_x - min_x,
                                               dtype='int'),
                             columns=np.linspace(min_y,
                                                 max_y - 1,
                                                 max_y - min_y,
                                                 dtype='int'))

        sea.set(font_scale=1.6)

        if self.show_cbar:
            # get sharp grid back by removing rasterized=True, and save fig as svg format
            self.ax = sea.heatmap(df_ht, cbar=True, mask=(df_ht == 0),
                                  vmin=self.sig_min, vmax=self.sig_max, square=True, rasterized=True)
            self.show_cbar = True
        else:
            # get sharp grid back by removing rasterized=True, and save fig as svg format
            self.ax = sea.heatmap(df_ht, cbar=False, mask=(df_ht == 0),
                                  vmin=self.sig_min, vmax=self.sig_max, square=True, rasterized=True)

        self.ax.tick_params(colors='black', left=False, bottom=False)

        plt.rcParams.update({'font.size': 22})
        self.ax.set(xlabel='Node pose x', ylabel='Node pose y')

        # force background to white
        # self.ax.set(facecolor="white")

        # y axis upside down
        self.ax.invert_yaxis()

        # set Axis label
        self.ax.set_xlabel('Longitude (m)', fontsize=16)
        self.ax.set_ylabel('Latitude (m)', fontsize=16)


        # self.fig.canvas.draw()
        #
        # self.fig.tight_layout()
        #
        # self.fig.savefig(self.data_path + "/figs/%iG_Signal_Heatmap" % sig_g + ".pdf")
        #
        # if self.display:
        #     plt.show()



class VisualiseCARV2(object):
    """
    A class to visualise the Call_A_Robot device V2 collected from Hatchgate
    """

    def __init__(self, data_path, data_name, start_time, time_end, df_gps_x, df_gps_y, fig, ax):
        self.data_path = data_path
        self.data_name = data_name
        self.data = None
        self.time_start = start_time  # Unix time
        self.time_end = time_end
        self.gps_x = []
        self.gps_y = []
        self.df_gps_x = df_gps_x      # gps boundary from robot's trajectory
        self.df_gps_y = df_gps_y
        self.status = []
        self.show_cbar = True
        self.invalid_gps = -1.0
        self.gps_y_bound = 48692  # limit the V2 gps to the Hatchgate west, excluding the signal from the east side

        self.display = True

        if fig:
            self.fig = fig
        if ax:
            self.ax = ax
        else:
            self.fig, self.ax = plt.subplots(1, 1, figsize=(16, 8), sharex=False, sharey=False)

        self.get_data()
        self.plot_ht()

    def get_data(self):
        """read data from the log file"""
        with open(self.data_path + "/" + self.data_name, "r") as stream:
            try:
                self.data = stream.readlines()
            except yaml.YAMLError as exc:
                print(exc)

    def plot_ht(self):
        for line in self.data:
            lis = line.split(",")
            if not lis[4] and self.time_start <= int(lis[1]) <= self.time_end \
                    and float(lis[7]) != self.invalid_gps and float(lis[8]) != self.invalid_gps and float(lis[8]) > 0:
                if round(float(lis[8]) * 111139, 1) < self.gps_y_bound:
                    self.gps_x.append(round(float(lis[7]) * 111139, 1))
                    self.gps_y.append(round(float(lis[8]) * 111139, 1))
                    self.status.append(int(lis[5]))

        max_x = round(max(self.gps_x))
        min_x = round(min(self.gps_x))
        max_y = round(max(self.gps_y))
        min_y = round(min(self.gps_y))
        robot_max_x = round(max(self.df_gps_x))
        robot_min_x = round(min(self.df_gps_x))
        robot_max_y = round(max(self.df_gps_y))
        robot_min_y = round(min(self.df_gps_y))


        data_status = np.zeros((robot_max_x - robot_min_x,
                                robot_max_y - robot_min_y))

        for i, x in enumerate(self.gps_x):
            for j, y in enumerate(self.gps_y):
                if i == j:
                    data_status[round(x) - robot_min_x - 1, round(y) - robot_min_y - 1] = self.status[i]
                    break

        df_ht = pd.DataFrame(data_status,
                             index=np.linspace(robot_min_x,
                                               robot_max_x - 1,
                                               robot_max_x - robot_min_x,
                                               dtype='int'),
                             columns=np.linspace(robot_min_y,
                                                 robot_max_y - 1,
                                                 robot_max_y - robot_min_y,
                                                 dtype='int'))

        sea.set(font_scale=1.6)

        myColors = ((0.8, 0.0, 0.0, 1.0), (0.0, 0.0, 0.8, 1.0))
        cmap = LinearSegmentedColormap.from_list('Custom', myColors, len(myColors))


        if self.show_cbar:
            # get sharp grid back by removing rasterized=True, and save fig as svg format
            self.ax = sea.heatmap(df_ht, cmap=cmap, mask=(df_ht == 0), square=True,
                                  rasterized=True, cbar_kws={"shrink": 0.8})
            self.show_cbar = False
        else:
            # get sharp grid back by removing rasterized=True, and save fig as svg format
            self.ax = sea.heatmap(df_ht, cmap=cmap, mask=(df_ht == 0), square=True,
                                  rasterized=True, cbar_kws={"shrink": 0.8})

        self.ax.tick_params(colors='black', left=False, bottom=False)

        # Manually specify colorbar labelling after it's been generated
        colorbar = self.ax.collections[1].colorbar
        colorbar.set_ticks([10.275, 10.75])
        colorbar.set_ticklabels(["State 10", "State 11"])
        plt.rcParams.update({'font.size': 16})

        # y axis upside down
        self.ax.invert_yaxis()

        # set Axis label
        self.ax.set_xlabel('Longitude (m)', fontsize=16)
        self.ax.set_ylabel('Latitude (m)', fontsize=16)

        self.fig.canvas.draw()

        self.fig.tight_layout()

        self.fig.savefig(self.data_path + "/figs/V2_Signal_Heatmap" + ".pdf")

        if self.display:
            plt.show()
