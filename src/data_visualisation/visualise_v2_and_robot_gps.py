# GPS data of Smart-Trolley Devices are collected from https://lcas.lincoln.ac.uk/car/orders as the rosbags
# collected from Hatchgate (July) failed to get the STD data. Could be caused by MQTT overload or poor signal
# inside the Polytunnels.
# Note: When rosbaging inside the polytunnels, double check both the robot and STD are publishing valid data to
#       the server.


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
import ast
import math

# parameter used to convert GPS to decimeter(1111390)/meters(111139)
GPS2COOR = 1111390


class VisualiseSignal(object):
    """
    A class to visualise the data for GPS
    """

    def __init__(self, data_path_, file_type_, bag_name, car_topic, robot_topic, fig=None, ax=None):
        self.entries_list = None
        self.data_path = data_path_
        self.file_type = file_type_
        self.bag_name = bag_name
        self.INVALID_SIG = -999
        self.sig_max = 0
        self.sig_min = 0
        self.display = False

        self.df_robot_gps = dict(x=[], y=[])
        self.df_car_gps = {}

        self.heatmap_data = None
        self.heatmap_index = None
        self.heatmap_columns = None

        self.car_topic = car_topic
        self.robot_topic = robot_topic

        self.time_start = 0  # Unix time
        self.time_end = 0

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
        for g in generations:  # [2, 3, 4, 5]
            if g < 4:
                self.sig_max = 110
                self.sig_min = 50
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
                if self.file_type == ifile[-3:]:
                    ifile = self.data_path + '/' + ifile
                    # matchedtopics = []
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

    # def get_car_gps(self, df_car_gps):
    #     for line in df_car_gps.data:
    #         lis = line.split(",")
    #         if not lis[4] and self.time_start <= int(lis[1]) <= self.time_end \
    #                 and float(lis[7]) != self.invalid_gps and float(lis[8]) != self.invalid_gps and float(
    #             lis[8]) > 0:
    #             if round(float(lis[8]) * GPS2COOR, 1) < self.gps_y_bound:
    #                 self.gps_x.append(round(float(lis[7]) * GPS2COOR, 1))
    #                 self.gps_y.append(round(float(lis[8]) * GPS2COOR, 1))
    #                 self.status.append(int(lis[5]))

    def init_heatmap(self, bag_name, sig_g=2):
        """
        Plot heatmap of the signal
        """
        b = bagreader(self.data_path + '/' + bag_name)
        data_car_gps = b.message_by_topic(self.car_topic)
        data_robot_gps = b.message_by_topic(self.robot_topic)
        print("File saved: {}".format(data_robot_gps))
        print("File saved: {}".format(data_car_gps))

        # df_car_gps = pd.read_csv(data_car_gps)
        df_robot_gps = pd.read_csv(data_robot_gps)
        long_robot_diff = (max(df_robot_gps.longitude) - min(df_robot_gps.longitude)) * GPS2COOR
        lat_robot_diff = (max(df_robot_gps.latitude) - min(df_robot_gps.latitude)) * GPS2COOR

        self.df_robot_gps["x"] = []
        self.df_robot_gps["y"] = []
        self.df_car_gps = {}
        # for idx in match_idx:
        for idx in range(len(df_robot_gps.latitude)):
            self.df_robot_gps["x"].append(round(df_robot_gps.latitude[idx] * GPS2COOR, 1))
            self.df_robot_gps["y"].append(round(df_robot_gps.longitude[idx] * GPS2COOR, 1))

        max_x = math.ceil(max(self.df_robot_gps["x"]))
        min_x = math.floor(min(self.df_robot_gps["x"]))
        max_y = math.ceil(max(self.df_robot_gps["y"]))
        min_y = math.floor(min(self.df_robot_gps["y"]))

        if max_y-min_y < 2 or max_x-min_x < 2:
            print("Error: Seems robot is not moving: max_x = %d, min_x = %d" % (max_x, min_x))
            print("Error: Seems robot is not moving: max_y = %d, min_y = %d" % (max_y, min_y))
            exit(1)

        data_sig2 = np.zeros((max_x - min_x,
                              max_y - min_y))

        # robot GPS
        for i, x in enumerate(self.df_robot_gps["x"]):
            for j, y in enumerate(self.df_robot_gps["y"]):
                if i == j:
                    data_sig2[int(math.ceil(x)) - min_x - 1, int(math.ceil(y)) - min_y - 1] = self.sig_max * 0.8
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
            self.ax = sea.heatmap(df_ht, cbar=True, mask=(df_ht == 0), cmap='Reds',
                                  vmin=self.sig_min, vmax=self.sig_max, square=True, rasterized=True)
            self.show_cbar = True
        else:
            # get sharp grid back by removing rasterized=True, and save fig as svg format
            self.ax = sea.heatmap(df_ht, cbar=False, mask=(df_ht == 0), cmap='Reds',
                                  vmin=self.sig_min, vmax=self.sig_max, square=True, rasterized=True)

        self.ax.tick_params(colors='black', left=False, bottom=False)

        plt.rcParams.update({'font.size': 22})
        self.ax.set(xlabel='Node pose x', ylabel='Node pose y')

        # force background to white
        # self.ax.set(facecolor="white")

        # y axis upside down
        self.ax.invert_yaxis()

        # set Axis label
        self.ax.set_xlabel('Longitude (dm)', fontsize=16)
        self.ax.set_ylabel('Latitude (dm)', fontsize=16)

        self.fig.canvas.draw()

        self.fig.tight_layout()

        self.fig.savefig(self.data_path + "/figs/robot_GPS_Heatmap" + ".pdf")

        if self.display:
            plt.show()


class VisualiseCARV2(object):
    """
    A class to visualise the Call_A_Robot device V2 collected from Hatchgate
    """

    def __init__(self, user, data_path, data_name, start_time, time_end, df_gps_x, df_gps_y, fig, ax):
        self.user = user
        self.data_path = data_path
        self.data_name = data_name
        self.data = None
        self.time_start = start_time  # Unix time
        self.time_end = time_end
        self.gps_x = {}
        self.gps_y = {}
        self.df_gps_x = df_gps_x  # gps boundary from robot's trajectory
        self.df_gps_y = df_gps_y
        self.status = {}
        self.show_cbar = True
        self.invalid_gps = -1.0
        self.gps_y_bound = 9999999999  # limit the V2 gps to the specified side when necessary

        self.display = True

        if fig:
            self.fig = fig
        if ax:
            self.ax = ax
        else:
            self.fig, self.ax = plt.subplots(1, 1, figsize=(16, 8), sharex=False, sharey=False)

        self.get_data()
        self.plot_ht(self.user)

    def get_data(self):
        """read data from the log file"""
        with open(self.data_path + "/" + self.data_name, "r") as stream:
            try:
                self.data = stream.readlines()
            except yaml.YAMLError as exc:
                print(exc)

    def plot_ht(self, user):
        #TODO: seperate STD based on their names

        data_status = {}
        for u in user:
            self.gps_x[u] = []
            self.gps_y[u] = []
            self.status[u] = []
            data_status[u] = None
        for line in self.data:
            lis = line.split(",")
            # if not lis[4] and self.time_start <= int(lis[1]) <= self.time_end \
            #         and float(lis[7]) != self.invalid_gps and float(lis[8]) != self.invalid_gps and float(lis[8]) > 0:
            #     if round(float(lis[8]) * GPS2COOR, 1) < self.gps_y_bound:
            #         u = lis[3]
            #         self.gps_x[u].append(round(float(lis[7]) * GPS2COOR, 1))
            #         self.gps_y[u].append(round(float(lis[8]) * GPS2COOR, 1))
            #         self.status[u].append(int(u[-2:]))

            # riseholme
            if not lis[4] and self.time_start <= int(lis[1]) <= self.time_end \
                    and float(lis[7]) != self.invalid_gps and float(lis[8]) != self.invalid_gps and float(lis[8]) < 0:
                if round(float(lis[8]) * GPS2COOR, 1) < self.gps_y_bound:
                    u = lis[3]
                    self.gps_x[u].append(round(float(lis[7]) * GPS2COOR, 1))
                    self.gps_y[u].append(round(float(lis[8]) * GPS2COOR, 1))
                    self.status[u].append(int(u[-2:]))

        robot_max_x = math.ceil(max(self.df_gps_x))
        robot_min_x = math.floor(min(self.df_gps_x))
        robot_max_y = math.ceil(max(self.df_gps_y))
        robot_min_y = math.floor(min(self.df_gps_y))

        for u in self.user:
            data_status[u] = np.zeros((robot_max_x - robot_min_x,
                                       robot_max_y - robot_min_y))
            for i, x in enumerate(self.gps_x[u]):
                # if picker is outside of the area of robot, ignore
                if int(math.ceil(x)) > robot_max_x:
                    continue
                for j, y in enumerate(self.gps_y[u]):
                    # if picker is outside of the area of robot, ignore
                    if int(math.ceil(y)) > robot_max_y:
                        continue
                    if i == j:
                        data_status[u][int(math.ceil(x)) - robot_min_x - 1,
                                       int(math.ceil(y)) - robot_min_y - 1] = self.status[u][i]
                        break

        df_ht = {}
        # todo debug
        # self.user = [self.user[0]]
        for u in self.user:
            df_ht[u] = pd.DataFrame(data_status[u],
                                    index=np.linspace(robot_min_x,
                                                      robot_max_x - 1,
                                                      robot_max_x - robot_min_x,
                                                      dtype='int'),
                                    columns=np.linspace(robot_min_y,
                                                        robot_max_y - 1,
                                                        robot_max_y - robot_min_y,
                                                        dtype='int'))

        sea.set(font_scale=1.6)

        # red, blue, yellow
        myColors = ((0.0, 0.0, 0.8, 1.0), (0.8, 0.8, 0, 1.0), (0.9, 0.0, 0.0, 1.0))
        cmap = LinearSegmentedColormap.from_list('Custom', myColors, len(myColors))

        if self.show_cbar:
            # get sharp grid back by removing rasterized=True, and save fig as svg format
            # generate cbar only once
            self.ax = sea.heatmap(df_ht[self.user[0]], cmap=cmap, mask=(df_ht[self.user[0]] == 0), square=True,
                                  rasterized=True, cbar_kws={"shrink": 0.1})
            for u in self.user[1:]:
                self.ax = sea.heatmap(df_ht[u], cmap=cmap, mask=(df_ht[u] == 0), square=True,
                                      rasterized=True, cbar=False)
            self.show_cbar = False
        else:
            # get sharp grid back by removing rasterized=True, and save fig as svg format
            for u in self.user:
                self.ax = sea.heatmap(df_ht[u], cmap=cmap, mask=(df_ht[u] == 0), square=True,
                                      rasterized=True, cbar=False)

        self.ax.tick_params(colors='black', left=False, bottom=False)

        # Manually specify colorbar labelling after it's been generated
        colorbar = self.ax.collections[0].colorbar
        colorbar.set_ticks([72.75, 63.25, 68.05])
        # user_idx = []
        # for u in self.user:
        #     user_idx.append(u[-2:])
        tick_labels = ["Robot_024"] + self.user
        colorbar.set_ticklabels(tick_labels)
        plt.rcParams.update({'font.size': 16})

        # y axis upside down
        self.ax.invert_yaxis()

        # set Axis label
        self.ax.set_xlabel('Longitude (dm)', fontsize=16)
        self.ax.set_ylabel('Latitude (dm)', fontsize=16)

        self.fig.canvas.draw()

        self.fig.tight_layout()

        self.fig.savefig(self.data_path + "/figs/ROBOT_and_STDv2_GPS_Heatmap" + ".pdf")

        if self.display:
            plt.show()
