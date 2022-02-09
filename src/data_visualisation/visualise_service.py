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

# Trials in Sep were with Pickers waiting. There were many calls that were not serviced/cancelled.
# In Oct, we had trials in two days. Day 1 with pickers waiting and Day2 with pickers continuing picking.
# TIME_2_STORAGE = [71.46, 41.28]  # contains noise
TIME_2_PICKER = [81.24, 52.83]
# SERVICE_TIME_TOTAL = [181.45, 125.99]
SERVICE_DISTANCE = [1865.067616, 987.792829]

TIME_2_STORAGE = [85.73, 63.57]             # corrected values
SERVICE_TIME_TOTAL = [184.99, 125.99]       # corrected values


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

        self.fig3, self.ax3 = plt.subplots(nrows=1, ncols=1, figsize=(9, 9))
        self.fig4, self.ax4 = plt.subplots(nrows=1, ncols=1, figsize=(9, 9))
        self.fig5, self.ax5 = plt.subplots(nrows=1, ncols=1, figsize=(9, 9))
        self.fig6, self.ax6 = plt.subplots(nrows=1, ncols=1, figsize=(9, 9))
        self.fig7, self.ax7 = plt.subplots(nrows=1, ncols=1, figsize=(9, 9))

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

        self.plot_service()

    def plot_service(self):
        """
        plot robot service time to picker, robot service time to storage, robot service distance
        :return: None
        """
        service_distance = []
        time_to_picker = []
        time_to_storage = []
        for ser in self.plot_d["service"]:
            service_distance.append(ser['service_distance'])
            time_to_picker.append(ser['time_to_picker'])
            time_to_storage.append(ser['time_to_storage'])
        service_distance_array = np.array(service_distance)
        service_distance_array = np.rot90(service_distance_array)
        time_to_storage_array = np.array(time_to_storage)
        time_to_storage_array = np.rot90(time_to_storage_array)
        time_to_picker_array = np.array(time_to_picker)
        time_to_picker_array = np.rot90(time_to_picker_array)

        ######### robot TOTAL service time #################
        x_axis = ['Day 1', 'Day 2']  # CHF vanity full
        x_axis_line = [i + 1 for i in range(2)]  # CHF vanity full
        self.ax3.set_xlabel('Picking strategy', fontdict=self.font)
        self.ax3.set_ylabel('Total service time (min)', fontdict=self.font)

        boxplot = self.ax3.boxplot(np.divide(time_to_picker_array + time_to_storage_array, 60),
                                   vert=True,  # vertical box alignment
                                   patch_artist=False,
                                   labels=x_axis)
        self.ax3.tick_params(axis='y', labelsize=self.label_size)
        self.ax3.tick_params(axis='x', labelsize=self.label_size)

        time_to_picker_array_total_mean = np.mean(time_to_picker_array + time_to_storage_array, axis=0)
        self.ax3.plot(x_axis_line, time_to_picker_array_total_mean/60, linestyle=':',
                      label="Simulation (min), Day1={}, Day2={}".format(round(time_to_picker_array_total_mean[0] / 60, 1),
                                                                        round(time_to_picker_array_total_mean[1] / 60, 1)),
                                                                        linewidth=self.linewidth)
        self.ax3.plot(x_axis_line, SERVICE_TIME_TOTAL, linestyle='-',
                      label="Real (min), Day1={}, Day2={}".format(round(SERVICE_TIME_TOTAL[0], 1),
                                                                  round(SERVICE_TIME_TOTAL[1], 1)),
                                                                  linewidth=self.linewidth)
        self.fig3.savefig(self.data_path + '_' + self.policy + '_' + self.cold_storage + '_'
                          + '_total_service_time_to_picker.pdf', format='pdf')

        # x_axis = ['original', 'side + middle station', 'middle station', 'side station']   # CHF transportation 4 poly
        # self.ax4.set_xlabel('Waiting station', fontdict=self.font)

        ########## robot service distance to picker############
        # x_axis = ['None', 'Single lane half', 'Double lane half', 'Double lane full']  # CHF vanity full
        # self.ax4.set_xlabel('Cross lanes', fontdict=self.font)
        x_axis = ['Day 1', 'Day 2']
        x_axis_line = [i + 1 for i in range(2)]  # CHF vanity full
        self.ax4.set_xlabel('Picking strategy', fontdict=self.font)
        self.ax4.set_ylabel('Service distance (m)', fontdict=self.font)

        boxplot = self.ax4.boxplot(service_distance_array,
                                   vert=True,  # vertical box alignment
                                   patch_artist=False,
                                   labels=x_axis)
        self.ax4.tick_params(axis='y', labelsize=self.label_size)
        self.ax4.tick_params(axis='x', labelsize=self.label_size)
        service_distance_array_mean = np.mean(service_distance_array, axis=0)
        self.ax4.plot(x_axis_line, service_distance_array_mean, linestyle=':',
                      label="Simulation (m), Day1={}, Day2={}".format(round(service_distance_array_mean[0], 1),
                                                                      round(service_distance_array_mean[1], 1)),
                                                                      linewidth=self.linewidth)
        self.ax4.plot(x_axis_line, SERVICE_DISTANCE, linestyle='-',
                      label="Real (m), Day1={}, Day2={}".format(round(SERVICE_DISTANCE[0], 1),
                                                                round(SERVICE_DISTANCE[1], 1)),
                                                                linewidth=self.linewidth)
        self.fig4.savefig(self.data_path + '_' + self.policy + '_' + self.cold_storage + '_'
                          + '_service_distance.pdf', format='pdf')

        ######### robot service time to picker#################
        x_axis = ['Day 1', 'Day 2']  # CHF vanity full
        x_axis_line = [i + 1 for i in range(2)]  # CHF vanity full
        self.ax5.set_xlabel('Picking strategy', fontdict=self.font)
        self.ax5.set_ylabel('Time to picker (min)', fontdict=self.font)

        boxplot = self.ax5.boxplot(np.divide(time_to_picker_array, 60),   # seconds to minutes
                                   vert=True,  # vertical box alignment
                                   patch_artist=False,
                                   labels=x_axis)
        self.ax5.tick_params(axis='y', labelsize=self.label_size)
        self.ax5.tick_params(axis='x', labelsize=self.label_size)

        time_to_picker_array_mean = np.mean(time_to_picker_array, axis=0)
        self.ax5.plot(x_axis_line, time_to_picker_array_mean/60, linestyle=':',
                      label="Simulation (min), Day1={}, Day2={}".format(round(time_to_picker_array_mean[0] / 60, 1),
                                                                        round(time_to_picker_array_mean[1] / 60, 1)),
                                                                        linewidth=self.linewidth)
        self.ax5.plot(x_axis_line, TIME_2_PICKER, linestyle='-',
                      label="Real (min), Day1={}, Day2={}".format(round(TIME_2_PICKER[0], 1),
                                                                  round(TIME_2_PICKER[1], 1)),
                                                                  linewidth=self.linewidth)
        self.fig5.savefig(self.data_path + '_' + self.policy + '_' + self.cold_storage + '_'
                          + '_service_time_to_picker.pdf', format='pdf')

        ######### robot service time to storage##################
        x_axis = ['Day 1', 'Day 2']  # CHF vanity full
        x_axis_line = [i + 1 for i in range(2)]  # CHF vanity full
        self.ax6.set_xlabel('Picking strategy', fontdict=self.font)
        self.ax6.set_ylabel('Time to storage (min)', fontdict=self.font)
        color = 'tab:blue'

        boxplot = self.ax6.boxplot(np.divide(time_to_storage_array, 60),
                                   vert=True,  # vertical box alignment
                                   patch_artist=False,
                                   labels=x_axis)
        self.ax6.tick_params(axis='y', labelsize=self.label_size)
        self.ax6.tick_params(axis='x', labelsize=self.label_size)
        self.fig6.savefig(self.data_path + '_' + self.policy + '_' + self.cold_storage + '_'
                          + '_service_time_to_storage.pdf', format='pdf')
        time_to_storage_array_mean = np.mean(time_to_storage_array, axis=0)
        self.ax6.plot(x_axis_line, time_to_storage_array_mean / 60, linestyle=':',
                      label="Simulation (min), Day1={}, Day2={}".format(round(time_to_storage_array_mean[0]/60, 1),
                                                                        round(time_to_storage_array_mean[1]/60, 1)),
                      linewidth=self.linewidth)
        self.ax6.plot(x_axis_line, TIME_2_STORAGE, linestyle='-',
                      label="Real (min), Day1={}, Day2={}".format(round(TIME_2_STORAGE[0], 1),
                                                                  round(TIME_2_STORAGE[1], 1)),
                      linewidth=self.linewidth)

        self.ax6.tick_params(axis='y', labelcolor=color, labelsize=self.label_size)

        ######### robot service speed = service distance / service time to storage##################
        x_axis = ['Day 1', 'Day 2']  # CHF vanity full
        x_axis_line = [i + 1 for i in range(2)]  # CHF vanity full
        self.ax7.set_xlabel('Picking strategy', fontdict=self.font)
        self.ax7.set_ylabel('Robot service speed (m/s)', fontdict=self.font)
        color = 'tab:blue'

        service_speed_array = np.divide(service_distance_array, time_to_picker_array)
        boxplot = self.ax7.boxplot(service_speed_array,
                                   vert=True,  # vertical box alignment
                                   patch_artist=False,
                                   labels=x_axis)
        self.ax7.tick_params(axis='y', labelsize=self.label_size)
        self.ax7.tick_params(axis='x', labelsize=self.label_size)
        service_speed_array_mean = np.mean(service_speed_array, axis=0)
        self.ax7.plot(x_axis_line, service_speed_array_mean, linestyle=':',
                      label="Simulation (m/s), Day1={}, Day2={}".format(round(service_speed_array_mean[0], 1),
                                                                        round(service_speed_array_mean[1], 1)),
                                                                        linewidth=self.linewidth)
        time_2_picker_sec = map(lambda x: x * 60, TIME_2_PICKER)
        service_speed_real = [i/j for i, j in zip(SERVICE_DISTANCE, time_2_picker_sec)]
        self.ax7.plot(x_axis_line, service_speed_real, linestyle='-',
                      label="Real (m/s), Day1={}, Day2={}".format(round(service_speed_real[0], 1),
                                                                  round(service_speed_real[1], 1)),
                                                                  linewidth=self.linewidth)
        self.fig7.savefig(self.data_path + '_' + self.policy + '_' + self.cold_storage + '_'
                          + '_service_speed.pdf', format='pdf')

        self.fig3.legend(loc='upper right', prop={'size': self.legend_size})
        self.fig4.legend(loc='upper right', prop={'size': self.legend_size})
        self.fig5.legend(loc='upper right', prop={'size': self.legend_size})
        self.fig6.legend(loc='upper right', prop={'size': self.legend_size})
        self.fig7.legend(loc='upper right', prop={'size': self.legend_size})

        plt.show()




