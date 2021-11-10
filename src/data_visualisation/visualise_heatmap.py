#!/usr/bin/env python
# ----------------------------------
# @author: ZuyuanZhu
# @email: zuyuanzhu@gmail.com
# @date: 5 Oct 2021
# ----------------------------------


import random
import os
from datetime import datetime
import matplotlib.pyplot
import numpy
import pandas
import seaborn
# import rospy
import yaml
from scandir import scandir

# DEMAND_MAX = 249.1
# DEMAND_MAX = 83.4
DEMAND_MAX = 1


def list_directories(path):
    dir_list = []
    for entry in scandir(path):
        if entry.is_dir() and not entry.is_symlink():
            dir_list.append(entry.path)
            dir_list.extend(list_directories(entry.path))
    return dir_list


def get_data(config_file):
    """read data from the config yaml file"""
    f_handle = open(config_file, "r")
    config_data = yaml.full_load(f_handle)
    f_handle.close()
    return config_data


class VisualiseHeatmap(object):
    def __init__(self, data_path_, file_type_):
        self.data_path = data_path_
        self.entries_list = []
        self.file_type = file_type_
        self.heatmap_data = None
        self.heatmap_index = None
        self.heatmap_columns = None

        self.heatmap_values = []
        self.heatmap_v_mean = []
        self.heatmap_v_mean_unify =[]

        self.heatmap_node_pose_x_list = []
        self.heatmap_node_pose_y_list = []

        self.map_name = ''
        self.trial = None
        self.n_deadlock = []
        self.simulation_time = []

        self.show_cbar = True
        self.fig, self.ax = matplotlib.pyplot.subplots(1, 1, figsize=(16, 6), sharex=False, sharey=False)

        self.f_handle = open(self.data_path + "/" + datetime.now().isoformat().replace(":", "_") + "_heatmap_mean.yaml", "w")

        self.init_plot()

    def close_plot(self):
        """close plot"""
        matplotlib.pyplot.show()
        self.fig.savefig(self.data_path + "/" + datetime.now().isoformat().replace(":", "_") + "redo.pdf")

        # Save just the portion _inside_ the second axis's boundaries
        extent = self.ax.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
        self.fig.savefig(self.data_path + "/" + datetime.now().isoformat().replace(":", "_") + "redo_heatmap.pdf",
                         bbox_inches=extent.expanded(1.29, 1.29))

        self.f_handle.close()

        matplotlib.pyplot.close(self.fig)

    def get_folder_files(self, data_path_):
        entries_list = []
        with os.scandir(data_path) as entries:
            for entry in entries:
                if entry.is_file():
                    entries_list.append(entry.name)
        entries_list.sort()
        self.entries_list = entries_list

    def load_data(self, data_path_, entries_list, file_type_):
        heatmap_data = []
        for file_name in entries_list:
            if file_type_ in file_name:
                heatmap_data.append(get_data(data_path_ + "/" + file_name))
                print(file_name)

        # heatmap_data = sorted(heatmap_data, key=lambda data: data["trial"])
        # self.heatmap_data = heatmap_data
        for heatmap_d in heatmap_data:
            self.heatmap_values.append(heatmap_d["heatmap_values"])
            self.heatmap_node_pose_x_list.append(heatmap_d["heatmap_node_pose_x_list"])
            self.heatmap_node_pose_y_list.append(heatmap_d["heatmap_node_pose_y_list"])
            self.map_name = heatmap_data[0]["map_name"]
            self.trial = len(entries_list)
            self.n_deadlock.append(heatmap_d["n_deadlock"])
            self.simulation_time.append(heatmap_d["simulation_time"])

    def init_plot(self):
        self.get_folder_files(self.data_path)
        self.load_data(self.data_path, self.entries_list, self.file_type)

        max_x = max(self.heatmap_node_pose_x_list[0])
        min_x = min(self.heatmap_node_pose_x_list[0])
        max_y = max(self.heatmap_node_pose_y_list[0])
        min_y = min(self.heatmap_node_pose_y_list[0])
        data = numpy.zeros((max_x - min_x,
                            max_y - min_y))

        heatmap_v = numpy.array(self.heatmap_values)
        heatmap_v_mean = numpy.mean(heatmap_v, axis=0)
        heatmap_v_mean = heatmap_v_mean.tolist()
        self.heatmap_v_mean = heatmap_v_mean
        if DEMAND_MAX == 1:
            self.heatmap_v_mean_unify = self.heatmap_v_mean
        else:
            self.heatmap_v_mean_unify = [x/DEMAND_MAX for x in self.heatmap_v_mean]
        heatmap_v_mean_unify = self.heatmap_v_mean_unify
        for i, x in enumerate(self.heatmap_node_pose_x_list[0]):
            for j, y in enumerate(self.heatmap_node_pose_y_list[0]):
                if i == j:
                    data[x - min_x - 1, y - min_y - 1] = heatmap_v_mean_unify[i]
                    break

        df = pandas.DataFrame(data,
                              index=numpy.linspace(min_x, max_x - 1, max_x - min_x, dtype='int'),
                              columns=numpy.linspace(min_y, max_y - 1, max_y - min_y, dtype='int'))

        seaborn.set(font_scale=1.6)

        # only initialise color bar once, then don't update it anymore
        if self.show_cbar:
            # get sharp grid back by removing rasterized=True, and save fig as svg format
            if DEMAND_MAX == 1:
                self.ax = seaborn.heatmap(df, cbar=True, rasterized=True)
            else:
                self.ax = seaborn.heatmap(df, cbar=True, vmin=0, vmax=1, rasterized=True)
            self.show_cbar = False
        else:
            # get sharp grid back by removing rasterized=True, and save fig as svg format
            self.ax = seaborn.heatmap(df, cbar=False, vmin=0, vmax=1, rasterized=True)
        # matplotlib.rcParams.update({'font.size': 22})
        # self.ax[1].set(xlabel='Node pose y', ylabel='Node pose x')
        self.ax.set_xlabel('Node pose y', fontsize=16)
        self.ax.set_ylabel('Node pose x', fontsize=16)

        self.fig.canvas.draw()

        # save heatmap details
        self.f_handle.writelines("# heatmap details\n")
        self.f_handle.writelines("map_name: %s\n" % self.map_name)
        self.f_handle.writelines("trial: %d\n" % (self.trial - 1))
        self.f_handle.writelines("n_deadlock: %s\n" % self.n_deadlock)
        self.f_handle.writelines("n_deadlock_mean: %0.1f\n" % (sum(self.n_deadlock)/len(self.n_deadlock)))
        self.f_handle.writelines("simulation_time: %s\n" % self.simulation_time)
        self.f_handle.writelines("simulation_time_mean: %0.1f\n" % (sum(self.simulation_time)/len(self.simulation_time)))
        self.f_handle.writelines("heatmap_values_mean: %s\n" % self.heatmap_v_mean)
        self.f_handle.writelines("heatmap_v_mean_unify: %s\n" % self.heatmap_v_mean_unify)
        self.f_handle.writelines("heatmap_values_mean_max: %0.1f\n" % max(self.heatmap_v_mean))
        self.f_handle.writelines("heatmap_values_mean_min: %0.1f\n" % min(self.heatmap_v_mean))
        self.f_handle.writelines("heatmap_node_pose_x_list: %s\n" % self.heatmap_node_pose_x_list[0])
        self.f_handle.writelines("heatmap_node_pose_x_max: %d\n" % max(self.heatmap_node_pose_x_list[0]))
        self.f_handle.writelines("heatmap_node_pose_x_min: %d\n" % min(self.heatmap_node_pose_x_list[0]))
        self.f_handle.writelines("heatmap_node_pose_y_list: %s\n" % self.heatmap_node_pose_y_list[0])
        self.f_handle.writelines("heatmap_node_pose_y_max: %d\n" % max(self.heatmap_node_pose_y_list[0]))
        self.f_handle.writelines("heatmap_node_pose_y_min: %d\n" % min(self.heatmap_node_pose_y_list[0]))

        self.close_plot()


if __name__ == "__main__":

    folder = "heatmap_chf_4_mirror_2"
    data_folder_path = "/home/zuyuan/des_simpy_logs/"
    data_path = data_folder_path + folder
    file_type = 'heatmap.yaml'

    vis = VisualiseHeatmap(data_path, file_type)

