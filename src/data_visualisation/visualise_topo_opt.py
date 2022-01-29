#!/usr/bin/env python

# ----------------------------------
# @author: Zuyuan
# @email: zuyuanzhu@gmail.com
# @date: 19-May-2021
# @info: data visualisation for DES repository
# ----------------------------------

import os
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import yaml


def get_data(config_file):
    """read data from the config yaml file"""
    f_handle = open(config_file, "r")
    config_data = yaml.full_load(f_handle)
    f_handle.close()
    return config_data


class Visualise(object):
    """
    A class to visualise the results of picking and transporting tasks by pickers and robots in rasberry_des
    """

    def __init__(self, data_path, file_type, n_iteration):
        self.data_path = data_path
        self.n_iteration = n_iteration
        self.policy = None   # "uniform_utilisation", "lexicographical", "shortest_distance"
        self.policies = ["uniform_utilisation", "lexicographical", "shortest_distance"]
        self.use_cold_storage = None
        self.cold_storage = None
        self.map_name = None

        self.fig_name_base = data_path

        self.entries_list = []

        self.file_type = file_type  # 'yaml'

        self.trials_data = []

        self.service = {'call_id': [],
                        'n_trays': [],
                        'time_to_picker': [],
                        'time_to_storage': [],
                        'service_distance': []
                        }
        self.services = {}
        self.service_sum = {'n_call': 0,
                            'n_trays': .0,
                            'time_to_picker': .0,
                            'time_to_storage': .0,
                            'service_distance': .0
                            }
        self.service_trails = {'n_call': [],
                               'n_trays': [],
                               'time_to_picker': [],
                               'time_to_storage': [],
                               'service_distance': []
                               }

        self.plot_d = {"sim_finish_time_simpy": [],
                       "map_name": '',
                       "picker_total_working_times_array": [],
                       "picker_wait_for_robot_time_array": [],
                       "robot_total_working_times_array": [],
                       "robot_00_total_working_times_array": [],
                       "n_robots": [],
                       "service": []
                       }
        self.finish_time_array = []
        self.finish_time_array_with_robots = []

        self.font = {'family': 'serif', 'color': 'black', 'weight': 'bold', 'size': 14}
        self.font_over_bar = {'family': 'serif', 'color': 'olive', 'size': 14}
        self.label_size = 14
        self.legend_size = 14
        self.linewidth = 4
        # self.fig1, self.ax1 = plt.subplots(nrows=1, ncols=1, figsize=(9, 9))
        # self.fig2, self.ax2 = plt.subplots(nrows=1, ncols=1, figsize=(9, 9))
        # if n_iteration < 10:
        #     self.fig3, self.ax3 = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
        # else:
        #     self.fig3, self.ax3 = plt.subplots(nrows=1, ncols=1, figsize=(9, 9))
        self.fig4, self.ax4 = plt.subplots(nrows=1, ncols=1, figsize=(9, 9))
        self.init_plot()

    def get_folder_files(self, data_path):
        entries_list = []
        with os.scandir(data_path) as entries:
            for entry in entries:
                if entry.is_file():
                    entries_list.append(entry.name)
        entries_list.sort()
        self.entries_list = entries_list

    def load_data(self, data_path, entries_list, file_type):
        trails_data = []
        for file_name in entries_list:
            if file_type in file_name:
                trails_data.append(get_data(data_path + "/" + file_name))
                print(file_name)
                for policy in self.policies:
                    if policy in file_name:
                        self.policy = policy

        trails_data = sorted(trails_data, key=lambda data: data["sim_details"]["n_robots"])
        self.trials_data = trails_data

    def get_plot_data(self, trails, n_iteration):
        counter = 0
        sim_finish_time_simpy = []
        picker_total_working_times_array = []
        picker_wait_for_robot_time_array = []
        robot_total_working_time_array = []
        robot_00_total_working_times_array = []
        sim_finish_time_clock = []
        for trial in trails:
            counter += 1
            self.map_name = trial["env_details"]["map_name"]
            self.use_cold_storage = trial["sim_details"]["use_cold_storage"]

            self.plot_d["n_pickers"] = trial["sim_details"]["n_pickers"]

            sim_finish_time_simpy.append(trial["sim_details"]["sim_finish_time_simpy"])
            sim_finish_time_clock.append(trial["sim_details"]["sim_finish_time_clock"])
            picker_total_working_time = 0.0
            picker_wait_for_robot_time = 0.0
            for picker in trial["sim_details"]["picker_states"]:
                picker_total_working_time += picker["total_working_time"]
                picker_wait_for_robot_time += picker["wait_for_robot_time"]
            picker_total_working_times_array.append(picker_total_working_time)
            picker_wait_for_robot_time_array.append(picker_wait_for_robot_time)

            robot_total_working_time = 0.0
            if trial["sim_details"]["robot_states"] is not None:
                for robot in trial["sim_details"]["robot_states"]:
                    robot_total_working_time += robot["total_working_time"]

                    # process service info if 'service' is in the data file
                    if 'service' in robot:
                        for service in robot['service']:
                            for key in self.service.keys():
                                self.service[key].append(service[key])
                        self.services[robot['robot_id']] = self.service
                        self.service_sum['n_call'] += len(self.service['call_id'])
                        self.service_sum['n_trays'] += sum(self.service['n_trays'])
                        self.service_sum['time_to_picker'] += sum(self.service['time_to_picker'])
                        self.service_sum['time_to_storage'] += sum(self.service['time_to_storage'])
                        self.service_sum['service_distance'] += sum(self.service['service_distance'])
                        self.service = {key: [] for key in self.service.keys()}

                    # get robot_00 working time
                    if robot['robot_id'] == 'robot_00':
                        robot_00_total_working_times_array.append(robot["total_working_time"])
            else:
                robot_total_working_time = 0
                # robot_00_total_working_time_array.append(0.0)
            robot_total_working_time_array.append(robot_total_working_time)

            for key in self.service_trails.keys():
                self.service_trails[key].append(self.service_sum[key])

            self.service_sum = {key: .0 for key in self.service_sum}

            if counter % n_iteration == 0:
                self.plot_d["sim_finish_time_simpy"].append(sim_finish_time_simpy)
                self.plot_d["picker_total_working_times_array"].append(picker_total_working_times_array)
                self.plot_d["picker_wait_for_robot_time_array"].append(picker_wait_for_robot_time_array)
                self.plot_d["robot_total_working_times_array"].append(robot_total_working_time_array)
                self.plot_d["n_robots"].append(trial["sim_details"]["n_robots"])
                self.plot_d["service"].append(self.service_trails)
                if counter / n_iteration > 1:
                    self.plot_d["robot_00_total_working_times_array"].append(robot_00_total_working_times_array)

                sim_finish_time_simpy = []
                picker_total_working_times_array = []
                picker_wait_for_robot_time_array = []
                robot_total_working_time_array = []
                robot_00_total_working_times_array = []
                self.service_trails = {key: [] for key in self.service_trails}

        if self.use_cold_storage:
            self.cold_storage = 'use_cold_storage'
        else:
            self.cold_storage = 'no_cold_storage'

    def init_plot(self):
        self.get_folder_files(self.data_path)
        self.load_data(self.data_path, self.entries_list, self.file_type)
        self.get_plot_data(self.trials_data, self.n_iteration)

        # Process completion time and picker utilisation
        # self.plot_picker_utilisation()

        self.plot_service()

        # Process completion time and picker wait rate
        # self.plot_picker_wait_time()

        # Robot utilisation
        # self.plot_robot_utilisation()

        # robot_00 utilisation
        # self.plot_robot_00_utilisation()

    def plot_robot_utilisation(self):
        """
        Robot utilisation
        :return: None
        """
        robot_total_working_times_array = np.array((self.plot_d["robot_total_working_times_array"]))
        robot_total_working_times_array = np.rot90(robot_total_working_times_array)
        n_robots = self.plot_d["n_robots"]
        total_time_array = self.finish_time_array * n_robots
        total_time_array[:, 0] = [0.01 for n in range(self.n_iteration)]
        robot_utilisation = 100 * np.divide(robot_total_working_times_array, total_time_array)
        robot_utilisation_mean = np.mean(robot_utilisation, axis=0)
        robot_utilisation_mean = robot_utilisation_mean.tolist()

        x_axis_mean = [i+1 for i in n_robots]
        x_axis = n_robots
        color = 'tab:olive'
        self.ax2.set_xlabel('Number of robots', fontdict=self.font)
        self.ax2.set_ylabel('Robot utilisation (%)', fontdict=self.font)

        boxplot = self.ax2.boxplot(robot_utilisation,
                                   vert=True,  # vertical box alignment
                                   patch_artist=False,
                                   labels=x_axis)  # will be used to label x-ticks
        self.ax2.plot(x_axis_mean, robot_utilisation_mean, linestyle='-.', color=color, linewidth=self.linewidth)
        self.ax2.tick_params(axis='y', labelcolor=color, labelsize=self.label_size)
        self.ax2.tick_params(axis='x', labelsize=self.label_size)
        self.fig2.tight_layout()
        self.fig2.savefig(self.data_path + '_' + self.map_name + '_' + self.policy + '_' + self.cold_storage + '_'
                          + '_robot_utilisation.pdf', format='pdf')

    def plot_robot_00_utilisation(self):
        """
        Robot utilisation only for robot_00
        :return: None
        """
        robot_00_total_working_times_array = np.array((self.plot_d["robot_00_total_working_times_array"]))
        robot_00_total_working_times_array = np.rot90(robot_00_total_working_times_array)
        n_robots = self.plot_d["n_robots"]
        # total_time_array = self.finish_time_array_with_robots * n_robots[1:]
        # total_time_array[:, 0] = [0.01 for n in range(self.n_iteration)]
        robot_00_utilisation = 100 * np.divide(robot_00_total_working_times_array, self.finish_time_array_with_robots)
        robot_00_utilisation_mean = np.mean(robot_00_utilisation, axis=0)
        robot_00_utilisation_mean = robot_00_utilisation_mean.tolist()

        x_axis_mean = [i for i in n_robots[1:]]
        x_axis = n_robots[1:]
        color = 'tab:olive'
        self.ax2.set_xlabel('Number of robots', fontdict=self.font)
        self.ax2.set_ylabel('Robot utilisation (%)', fontdict=self.font)

        boxplot = self.ax2.boxplot(robot_00_utilisation,
                                   vert=True,  # vertical box alignment
                                   patch_artist=False,
                                   labels=x_axis)  # will be used to label x-ticks
        self.ax2.plot(x_axis_mean, robot_00_utilisation_mean, linestyle='-.', color=color, linewidth=self.linewidth)
        self.ax2.tick_params(axis='y', labelcolor=color, labelsize=self.label_size)
        self.ax2.tick_params(axis='x', labelsize=self.label_size)
        self.fig2.tight_layout()
        self.fig2.savefig(self.data_path + '_' + self.map_name + '_' + self.policy + '_' + self.cold_storage + '_'
                          + '_robot_00_utilisation.pdf', format='pdf')

    def plot_picker_wait_time(self):
        """
        Process wait for robot time
        :return: None
        """
        finish_time_array = np.array(self.plot_d["sim_finish_time_simpy"])
        finish_time_array = np.rot90(finish_time_array)
        self.finish_time_array = finish_time_array

        # ignore n_robots == 0
        finish_time_array_with_robots = np.array(self.plot_d["sim_finish_time_simpy"][1:])
        finish_time_array_with_robots = np.rot90(finish_time_array_with_robots)
        self.finish_time_array_with_robots = finish_time_array_with_robots

        finish_time_array_mean = np.mean(finish_time_array, axis=0)
        finish_time_array_mean = finish_time_array_mean.tolist()
        finish_time_array_mean_median = [finish_time_array_mean[0] for x in range(len(finish_time_array_mean))]

        picker_wait_for_robot_time_array = np.array(self.plot_d["picker_wait_for_robot_time_array"])
        picker_wait_for_robot_time_array = np.rot90(picker_wait_for_robot_time_array)

        n_pickers = self.plot_d["n_pickers"]

        picker_wait_for_robot_time_array_mean = np.divide(picker_wait_for_robot_time_array, n_pickers)
        picker_wait_for_robot_time_array_mean_trials = np.mean(picker_wait_for_robot_time_array_mean, axis=0)
        picker_wait_for_robot_time_array_mean_trials = picker_wait_for_robot_time_array_mean_trials.tolist()

        picker_wait_rate = 100 * np.divide(picker_wait_for_robot_time_array, finish_time_array * n_pickers)
        picker_wait_rate_mean = np.mean(picker_wait_rate, axis=0)
        picker_wait_rate_mean = picker_wait_rate_mean.tolist()
        # picker_wait_rate_mean_median = [picker_wait_rate_mean[0] for x in range(len(picker_wait_rate_mean))]

        n_robots = self.plot_d["n_robots"]
        labels = [str(i) for i in n_robots]
        x_axis = [i+1 for i in n_robots]

        # ### Process completion time   ###
        color_c = 'tab:olive'
        self.ax3.set_xlabel('Number of robots', fontdict=self.font)
        self.ax3.set_ylabel('Process completion time (s)', fontdict=self.font)
        # boxplot1 = self.ax3.boxplot(finish_time_array,
        #                             vert=True,  # vertical box alignment
        #                             patch_artist=True,  # fill with color
        #                             labels=labels)  # will be used to label x-ticks
        # self.ax3.plot(x_axis, finish_time_array_mean, linestyle=':', label='Process completion time (s)', color=color_c)
        self.ax3.bar(x_axis, finish_time_array_mean, label='Process completion time (s)', color=color_c)
        # show bar value only when total robot number smaller than 11
        if len(x_axis) < 11:
            for i in range(len(x_axis)):
                finish_time = '%.1f' % finish_time_array_mean[i]
                plt.text(i+1, finish_time_array_mean[i], finish_time, fontdict=self.font_over_bar, ha='center', va='bottom')

        # ### Picker waiting for robot time ###
        color = 'tab:gray'
        self.ax3.bar(x_axis, picker_wait_for_robot_time_array_mean_trials, label='Picker waiting for robot time (s)', color=color)
        # show bar value only when total robot number smaller than 11
        if len(x_axis) < 11:
            for i in range(len(x_axis)):
                wait_time = '%.1f' % picker_wait_for_robot_time_array_mean_trials[i]
                plt.text(i+1,
                         picker_wait_for_robot_time_array_mean_trials[i],
                         wait_time,
                         fontdict=self.font_over_bar, ha='center', va='bottom')

        # self.ax3.boxplot(picker_wait_for_robot_time_array_mean,
        #                  vert=True,
        #                  patch_artist=True,
        #                  labels=labels)
        # self.ax3.plot(x_axis, picker_wait_for_robot_time_array_mean_trials, linestyle='--', label='Picker waiting for robot time (s)', color=color)

        self.ax3.tick_params(axis='y', labelcolor=color_c, labelsize=self.label_size)
        self.ax3.tick_params(axis='x', labelsize=self.label_size)

        # ### Picker waiting rate   ###
        ax3_2 = self.ax3.twinx()

        color = 'tab:blue'
        ax3_2.set_ylabel('Picker waiting rate (%)', fontdict=self.font)  # we already handled the x-label with ax1
        boxplot2 = ax3_2.boxplot(picker_wait_rate,
                                 vert=True,  # vertical box alignment
                                 patch_artist=False,  # fill with color
                                 labels=labels)  # will be used to label x-ticks
        ax3_2.plot(x_axis, picker_wait_rate_mean, linestyle='-.', label='Picker waiting rate (%)', color=color, linewidth=self.linewidth)
        # m = picker_wait_rate_mean_median[0]

        ax3_2.tick_params(axis='y', labelcolor=color, labelsize=self.label_size)

        # fill with colors
        colors1 = ['lightblue']
        colors2 = ['white']

        # fill the box with color
        # for patch, color in zip(boxplot1['boxes'], colors1):
        #     patch.set_facecolor(color)

        self.fig3.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()

        self.fig3.legend(loc='upper right', prop={'size': self.legend_size})  # loc='upper right'

        self.fig3.savefig(self.data_path + '_' + self.map_name + '_' + self.policy + '_' + self.cold_storage + '_'
                          '_process_completion_time_and_picker_wait_rate.pdf',
                          format='pdf')

    def plot_picker_utilisation(self):
        """
        Process completion time and picker utilisation
        :return: None
        """
        finish_time_array = np.array(self.plot_d["sim_finish_time_simpy"])
        finish_time_array = np.rot90(finish_time_array)
        self.finish_time_array = finish_time_array

        finish_time_array_mean = np.mean(finish_time_array, axis=0)
        finish_time_array_mean = finish_time_array_mean.tolist()
        finish_time_array_mean_median = [finish_time_array_mean[0] for x in range(len(finish_time_array_mean))]

        picker_total_working_times_array = np.array(self.plot_d["picker_total_working_times_array"])
        picker_total_working_times_array = np.rot90(picker_total_working_times_array)

        n_pickers = self.plot_d["n_pickers"]
        picker_utilisation = 100 * np.divide(picker_total_working_times_array, finish_time_array * n_pickers)
        picker_utilisation_mean = np.mean(picker_utilisation, axis=0)
        picker_utilisation_mean = picker_utilisation_mean.tolist()
        picker_utilisation_mean_median = [picker_utilisation_mean[0] for x in range(len(picker_utilisation_mean))]

        n_robots = self.plot_d["n_robots"]
        labels = [str(i) for i in n_robots]
        x_axis = [i+1 for i in n_robots]

        color = 'tab:olive'
        self.ax1.set_xlabel('Number of robots', fontdict=self.font)
        self.ax1.set_ylabel('Process completion time (s)', fontdict=self.font)
        boxplot1 = self.ax1.boxplot(finish_time_array,
                                    vert=True,  # vertical box alignment
                                    patch_artist=True,  # fill with color
                                    labels=labels)  # will be used to label x-ticks
        self.ax1.plot(x_axis, finish_time_array_mean, linestyle=':', color=color, linewidth=self.linewidth)
        self.ax1.plot(x_axis, finish_time_array_mean_median,
                      label='Median completion time, no robots (%.1f s)' % finish_time_array_mean_median[0],
                      color=color)
        self.ax1.tick_params(axis='y', labelcolor=color, labelsize=self.label_size)
        self.ax1.tick_params(axis='x', labelsize=self.label_size)
        # ax1.set_title('Process completion time and picker utilisation')

        ax2 = self.ax1.twinx()

        color = 'tab:blue'
        ax2.set_ylabel('Picker utilisation (%)', fontdict=self.font)  # we already handled the x-label with ax1
        boxplot2 = ax2.boxplot(picker_utilisation,
                               vert=True,  # vertical box alignment
                               patch_artist=False,  # fill with color
                               labels=labels)  # will be used to label x-ticks
        ax2.plot(x_axis, picker_utilisation_mean, linestyle='-.', color=color, linewidth=self.linewidth)
        m = picker_utilisation_mean_median[0]
        ax2.plot(x_axis, picker_utilisation_mean_median,
                 label=('Median picker utilisation, no robots (%.1f ' % m) + '%)',
                 color=color)
        ax2.tick_params(axis='y', labelcolor=color, labelsize=self.label_size)

        # fill with colors
        colors1 = ['lightblue']
        colors2 = ['white']
        # for patch, color in zip(boxplot1['boxes'], colors):
        #     patch.set_facecolor(color)

        for patch, color in zip(boxplot1['boxes'], colors1):
            patch.set_facecolor(color)

        self.fig1.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()

        self.fig1.legend(loc='upper right', prop={'size': self.legend_size})  # loc='upper right'

        self.fig1.savefig(self.data_path + '_' + self.map_name + '_' + self.policy + '_' + self.cold_storage + '_'
                          '_process_completion_time_and_picker_utilisation.pdf',
                           format='pdf')

    def plot_service(self):
        """
        plot robot service time to picker, robot service time to storage, robot service distance
        :return: None
        """
        service_distance = []
        for ser in self.plot_d["service"]:
            service_distance.append(ser['service_distance'])
        service_distance_array = np.array(service_distance)
        service_distance_array = np.rot90(service_distance_array)

        x_axis = ['original', 'side + middle station', 'middle station', 'side station']
        self.ax4.set_xlabel('Waiting station', fontdict=self.font)
        self.ax4.set_ylabel('Service distance (m)', fontdict=self.font)

        boxplot = self.ax4.boxplot(service_distance_array,
                                   vert=True,  # vertical box alignment
                                   patch_artist=False,
                                   labels=x_axis)
        self.ax4.tick_params(axis='y', labelsize=self.label_size)
        self.ax4.tick_params(axis='x', labelsize=self.label_size)
        self.fig4.savefig(self.data_path + '_' + self.policy + '_' + self.cold_storage + '_'
                          + '_service_distance.pdf', format='pdf')
        plt.show()




