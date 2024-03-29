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
import simpy

# parameter used to convert GPS to decimeter(1111390)/meters(111139)
GPS2COOR = 111139


class VisualiseCARV2(object):
    """
    A class to visualise the Call_A_Robot device V2 collected from Hatchgate
    """

    def __init__(self, env, user, data_path, data_name, start_time, time_end, df_gps_x, df_gps_y, fig=None, ax=None):
        self.env = env
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
        self.heat_value = {}
        self.loop_time = 0.1
        self.cbar_init = False

        self.display = True

        if fig:
            self.fig = fig
        if ax:
            self.ax = ax
        else:
            self.fig, self.ax = plt.subplots(1, 1, figsize=(16, 8), sharex=False, sharey=False)

        self.min_x, self.max_x, self.min_y, self.max_y = self.specify_bound()
        self.get_data()
        self.get_gps()
        self.fig.canvas.draw()
        self.action = self.env.process(self.update_pos())

    def specify_bound(self):
        """
        Specify the region of the GPS to be plotted
        """
        space = 0
        robot_max_x = math.ceil(max(self.df_gps_x))
        robot_min_x = math.floor(min(self.df_gps_x))
        robot_max_y = math.ceil(max(self.df_gps_y))
        robot_min_y = math.floor(min(self.df_gps_y))

        return robot_min_x-space, robot_max_x+space, robot_min_y-space, robot_max_y+space

    def get_data(self):
        """read data from the log file"""
        with open(self.data_path + "/" + self.data_name, "r") as stream:
            try:
                self.data = stream.readlines()
            except yaml.YAMLError as exc:
                print(exc)

    def get_gps(self):
        """
        Read user GPS from log file
        """
        for user in self.user:
            self.gps_x[user] = []
            self.gps_y[user] = []
            self.status[user] = []
            self.heat_value[user] = None

            for line in self.data:
                lis = line.split(",")
                # read GPS from log, double check the GPS is valid
                if not lis[4] \
                        and self.time_start <= int(lis[1]) <= self.time_end \
                        and float(lis[7]) != self.invalid_gps \
                        and float(lis[8]) != self.invalid_gps \
                        and float(lis[8]) < 0:
                    # GPS filter, restrict to the Riseholme region
                    if self.min_x < float(lis[7]) * GPS2COOR < self.max_x \
                            and self.min_y < float(lis[8]) * GPS2COOR < self.max_y:
                        # select the user
                        if user == lis[3]:
                            self.gps_x[user].append(round(float(lis[7]) * GPS2COOR, 1))
                            self.gps_y[user].append(round(float(lis[8]) * GPS2COOR, 1))
                            # use the last two digits of the PCD name as the heatmap value
                            self.status[user].append(int(lis[3][-2:]))

    def update_pos(self):

        for user in self.user:
            self.heat_value[user] = np.zeros((self.max_x - self.min_x,
                                               self.max_y - self.min_y))
            for i, x in enumerate(self.gps_x[user]):
                # ignore if PCD's GPS is outside of the specified region
                if x > self.max_x or x < self.min_x:
                    continue
                for j, y in enumerate(self.gps_y[user]):
                    # if picker is outside of the area of robot, ignore
                    if y > self.max_y or y < self.min_y:
                        continue
                    if i == j:
                        self.heat_value[user][int(math.ceil(x)) - self.min_x - 1,
                                               int(math.ceil(y)) - self.min_y - 1] = self.status[user][i]
                        self.update_plot()
                        yield self.env.timeout(self.loop_time)
                        break

        yield self.env.timeout(0.1)

    def update_plot(self):
        df_ht = dict()
        for user in self.user:
            df_ht[user] = pd.DataFrame(self.heat_value[user],
                                       index=np.linspace(self.min_x,
                                                         self.max_x - 1,
                                                         self.max_x - self.min_x,
                                                         dtype='int'),
                                       columns=np.linspace(self.min_y,
                                                           self.max_y - 1,
                                                           self.max_y - self.min_y,
                                                           dtype='int'))

        # self.ax.clear()
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
                if not df_ht[u].isnull().values.any():
                    self.ax = sea.heatmap(df_ht[u], cmap=cmap, mask=(df_ht[u] == 0), square=True,
                                          rasterized=True, cbar=False)
            self.show_cbar = False
        else:
            # get sharp grid back by removing rasterized=True, and save fig as svg format
            for u in self.user:
                if not df_ht[u].isnull().values.any():
                    self.ax = sea.heatmap(df_ht[u], cmap=cmap, mask=(df_ht[u] == 0), square=True,
                                          rasterized=True, cbar=False)

        self.ax.tick_params(colors='black', left=False, bottom=False)

        # Manually specify colorbar labelling after it's been generated
        if not self.cbar_init:
            colorbar = self.ax.collections[0].colorbar
            if colorbar is not None:
                colorbar.set_ticks([72.75, 63.25, 68.05])

                tick_labels = ["Robot_024"] + self.user
                colorbar.set_ticklabels(tick_labels)
            plt.rcParams.update({'font.size': 16})
            self.cbar_init = True

            # y axis upside down
            self.ax.invert_yaxis()

            # set Axis label
            self.ax.set_xlabel('Longitude (dm)', fontsize=16)
            self.ax.set_ylabel('Latitude (dm)', fontsize=16)

        self.fig.tight_layout()

        self.fig.canvas.draw()

        # self.fig.tight_layout()

        # self.fig.savefig(self.data_path + "/figs/ROBOT_and_STDv2_GPS_Heatmap" + ".pdf")

        # if self.display:
        #     plt.show()
