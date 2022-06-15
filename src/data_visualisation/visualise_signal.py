import bagpy
from bagpy import bagreader
import pandas as pd
import seaborn as sea
import matplotlib.pyplot as plt
import numpy as np
import os
from rosbag import Bag
import os.path
from fnmatch import fnmatchcase
from matplotlib import colors


class VisualiseSignal(object):
    """
    A class to visualise the data for GPS,
    Mobile Signal Strength from rosbag collected from Hatchgate
    """

    def __init__(self, data_path_, file_type_, bag_name):
        self.entries_list = None
        self.data_path = data_path_
        self.file_type = file_type_
        self.bag_name = bag_name
        self.INVALID_SIG = -999
        self.sig_max = 0
        self.sig_min = 0
        self.display = False

        self.heatmap_data = None
        self.heatmap_index = None
        self.heatmap_columns = None

        self.show_cbar = True
        self.fig, self.ax = plt.subplots(1, 1, figsize=(16, 9), sharex=False, sharey=False)
        # self.fig3, self.ax3 = plt.subplots(1, 1, figsize=(16, 6), sharex=False, sharey=False)

        # bag_name = 'bag1.bag'
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
            plt.close('all')
            self.fig, self.ax = plt.subplots(1, 1, figsize=(16, 16), sharex=False, sharey=False)

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

        match_idx = self.match_time(df_signal.Time, df_gps.Time)

        df_gps_x = []
        df_gps_y = []
        for idx in match_idx:
            df_gps_x.append(round(df_gps.latitude[idx] * 111139, 1))
            df_gps_y.append(round(df_gps.longitude[idx] * 111139, 1))

        max_x = round(max(df_gps_x))
        min_x = round(min(df_gps_x))
        max_y = round(max(df_gps_y))
        min_y = round(min(df_gps_y))
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

        for i, x in enumerate(df_gps_x):
            for j, y in enumerate(df_gps_y):
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

        # y axis upside down
        self.ax.invert_yaxis()

        # set Axis label
        self.ax.set_xlabel('Longitude (m)', fontsize=16)
        self.ax.set_ylabel('Latitude (m)', fontsize=16)

        self.fig.canvas.draw()

        self.fig.tight_layout()

        self.fig.savefig(self.data_path + "/figs/%iG_Signal_Heatmap" % sig_g + ".pdf")

        if self.display:
            plt.show()

        # Clear the current axes.
        plt.cla()
        # Clear the current figure.
        plt.clf()

        # plot signal line
        # gca stands for 'get current axis'
        ax = plt.gca()

        # force background to white
        ax.set(facecolor="white")
        # ax.patch.set_alpha(1.0)

        G2 = df_signal.data_3.replace(to_replace=-999, value=-140)
        G3 = df_signal.data_4.replace(to_replace=-999, value=-140)
        G4 = df_signal.data_5.replace(to_replace=-999, value=-140)
        G2.plot(kind='line', ax=ax, xlabel='Time stamp', ylabel='Signal strength', label="2G")
        G3.plot(kind='line', ax=ax, xlabel='Time stamp', ylabel='Signal strength', label="3G")
        G4.plot(kind='line', ax=ax, xlabel='Time stamp', ylabel='Signal strength', label="4G")
        plt.legend(loc="best")
        plt.xlabel("Time stamp", fontsize=16, fontweight='bold')
        plt.ylabel("Signal strength (dBm)", fontsize=16, fontweight='bold')
        ax.tick_params(axis='both', labelsize=16)
        # plt.show()

        plt.grid()
        plt.tight_layout()
        plt.savefig(self.data_path + "/figs/Signal" + ".pdf")

        # Clear the current axes.
        plt.cla()
        # Clear the current figure.
        plt.clf()

        del [df_ht]

    def match_time(self, timea, timeb):
        """
        timea and timeb are nx1 array of ROS time stamps: df.Time.
        Find the nearest time stamps between the two time arrays
        """
        if len(timea) > len(timeb):
            time = timea
            timea = timeb
            timeb = time

        match_idx = []
        idx = 0
        for i in range(len(timea)):
            dist_min = 99999
            for j in range(idx + 1, len(timeb)):
                if dist_min > (abs(timea[i] - timeb[j])):
                    dist_min = abs(timea[i] - timeb[j])
                    idx = j
                else:
                    break
            match_idx.append(idx)

        return match_idx
