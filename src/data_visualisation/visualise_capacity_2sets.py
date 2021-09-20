#!/usr/bin/env python

# ----------------------------------
# @author: Zuyuan
# @email: zuyuanzhu@gmail.com
# @date: 15-Sep-2021
# @info: data visualisation for DES repository, using different tray capacity and calling moment
# ----------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt
import yaml
import data_visualisation.visualise as vis


class VisualiseCapacity(vis.Visualise):
    """
    A class to visualise the results of picking and transporting tasks by pickers and robots in rasberry_des
    """

    def __init__(self, data_path, data_path_list, file_type, n_iteration):
        self.data_path_source = data_path
        self.data_path_list = data_path_list
        self.entries_list_all = []
        self.event_data_all = []
        self.plot_data_all = []
        self.colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown',
                       'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        self.fig1_2, self.ax1_2 = plt.subplots(nrows=1, ncols=1, figsize=(9, 9))
        super(VisualiseCapacity, self).__init__(data_path_list, file_type, n_iteration)

    def get_folder_files(self, data_path_list):
        entries_list = []
        for data_path in data_path_list:
            with os.scandir(data_path) as entries:
                for entry in entries:
                    if entry.is_file():
                        entries_list.append(entry.name)
            entries_list.sort()
            self.entries_list_all.append(entries_list)
            entries_list = []

    def load_data(self, data_path_list, entries_list_all, file_type):
        event_data = []
        for idx, entries_list in enumerate(entries_list_all):
            for file_name in entries_list:
                if file_type in file_name:
                    event_data.append(vis.get_data(data_path_list[idx] + file_name))
                    print(file_name)
                    for policy in self.policies:
                        if policy in file_name:
                            self.policy = policy

            event_data = sorted(event_data, key=lambda data: data["sim_details"]["n_robots"])
            self.event_data_all.append(event_data)
            event_data = []

    def get_plot_data(self, event_data_all, n_iteration):
        counter = 0
        sim_finish_time_simpy = []
        picker_total_working_times_array = []
        picker_wait_for_robot_time_array = []
        robot_total_working_time_array = []
        robot_00_total_working_times_array = []

        self.plot_data = []

        for event_data in event_data_all:
            for env in event_data:
                counter += 1
                self.map_name = env["env_details"]["map_name"]
                self.use_cold_storage = env["sim_details"]["use_cold_storage"]

                self.plot_d["n_pickers"] = env["sim_details"]["n_pickers"]

                sim_finish_time_simpy.append(env["sim_details"]["sim_finish_time_simpy"])

                picker_total_working_time = 0.0
                picker_wait_for_robot_time = 0.0
                for picker in env["sim_details"]["picker_states"]:
                    picker_total_working_time += picker["total_working_time"]
                    picker_wait_for_robot_time += picker["wait_for_robot_time"]
                picker_total_working_times_array.append(picker_total_working_time)
                picker_wait_for_robot_time_array.append(picker_wait_for_robot_time)

                robot_total_working_time = 0.0
                if env["sim_details"]["robot_states"] is not None:
                    for robot in env["sim_details"]["robot_states"]:
                        robot_total_working_time += robot["total_working_time"]

                        # get robot_00 working time
                        if robot['robot_id'] == 'robot_00':
                            robot_00_total_working_times_array.append(robot["total_working_time"])
                else:
                    robot_total_working_time = 0
                    # robot_00_total_working_time_array.append(0.0)
                robot_total_working_time_array.append(robot_total_working_time)

                if counter % n_iteration == 0:
                    self.plot_d["sim_finish_time_simpy"].append(sim_finish_time_simpy)
                    self.plot_d["picker_total_working_times_array"].append(picker_total_working_times_array)
                    self.plot_d["picker_wait_for_robot_time_array"].append(picker_wait_for_robot_time_array)
                    self.plot_d["robot_total_working_times_array"].append(robot_total_working_time_array)
                    self.plot_d["n_robots"].append(env["sim_details"]["n_robots"])
                    if counter / n_iteration > 1:
                        self.plot_d["robot_00_total_working_times_array"].append(robot_00_total_working_times_array)

                    sim_finish_time_simpy = []
                    picker_total_working_times_array = []
                    picker_wait_for_robot_time_array = []
                    robot_total_working_time_array = []
                    robot_00_total_working_times_array = []
                    self.plot_data.append(self.plot_d)

            self.plot_data_all.append(self.plot_d)
            self.plot_d = {"sim_finish_time_simpy": [],
                           "map_name": '',
                           "picker_total_working_times_array": [],
                           "picker_wait_for_robot_time_array": [],
                           "robot_total_working_times_array": [],
                           "robot_00_total_working_times_array": [],
                           "n_robots": []
                           }

        if self.use_cold_storage:
            self.cold_storage = 'use_cold_storage'
        else:
            self.cold_storage = 'no_cold_storage'

    def init_plot(self):
        self.get_folder_files(self.data_path_list)
        self.load_data(self.data_path_list, self.entries_list_all, self.file_type)
        self.get_plot_data(self.event_data_all, self.n_iteration)

        # Process completion time and picker utilisation
        self.plot_picker_utilisation()

        self.plot_picker_wait_time()

    def plot_picker_wait_time(self):
        """
        Process wait for robot time
        :return: None
        """
        ax3_2 = None
        n_cap = 1
        n_call = 1
        for idx, plot_data in enumerate(self.plot_data_all):
            finish_time_array = np.array(plot_data["sim_finish_time_simpy"])
            finish_time_array = np.rot90(finish_time_array)
            self.finish_time_array = finish_time_array

            # ignore n_robots == 0
            finish_time_array_with_robots = np.array(plot_data["sim_finish_time_simpy"][1:])
            finish_time_array_with_robots = np.rot90(finish_time_array_with_robots)
            self.finish_time_array_with_robots = finish_time_array_with_robots

            finish_time_array_mean = np.mean(finish_time_array, axis=0)
            finish_time_array_mean = finish_time_array_mean.tolist()
            finish_time_array_mean_median = [finish_time_array_mean[0] for x in range(len(finish_time_array_mean))]

            picker_wait_for_robot_time_array = np.array(plot_data["picker_wait_for_robot_time_array"])
            picker_wait_for_robot_time_array = np.rot90(picker_wait_for_robot_time_array)

            n_pickers = plot_data["n_pickers"]

            picker_wait_for_robot_time_array_mean = np.divide(picker_wait_for_robot_time_array, n_pickers)
            picker_wait_for_robot_time_array_mean_trials = np.mean(picker_wait_for_robot_time_array_mean, axis=0)
            picker_wait_for_robot_time_array_mean_trials = picker_wait_for_robot_time_array_mean_trials.tolist()

            picker_wait_rate = 100 * np.divide(picker_wait_for_robot_time_array, finish_time_array * n_pickers)
            picker_wait_rate_mean = np.mean(picker_wait_rate, axis=0)
            picker_wait_rate_mean = picker_wait_rate_mean.tolist()
            # picker_wait_rate_mean_median = [picker_wait_rate_mean[0] for x in range(len(picker_wait_rate_mean))]

            n_robots = plot_data["n_robots"]
            labels = [str(i) for i in n_robots]
            x_axis = [i for i in n_robots]
            color = self.colors[idx]

            # ### Picker waiting for robot time ###
            self.ax3.plot(x_axis, picker_wait_for_robot_time_array_mean_trials, linestyle=':',
                          label="Waiting time, Tcap={}, Tcall={}".format(n_cap, n_call), color=color, linewidth=self.linewidth)

            self.ax3.plot(x_axis, finish_time_array_mean, linestyle='-.',
                          label="Completion time, Tcap={}, Tcall={}".format(n_cap, n_call), color=color, linewidth=self.linewidth)

            # ONLY FOR ROBOT HIGHWAYS
            if n_call == n_cap:
                n_cap = 4
                n_call = 3

        self.ax3.set_xlabel('Number of robots', fontdict=self.font)
        self.ax3.set_ylabel('Time (s)', fontdict=self.font)

        self.ax3.tick_params(axis='y', labelsize=self.label_size)
        self.ax3.tick_params(axis='x', labelsize=self.label_size)

        self.fig3.tight_layout()  # otherwise the right y-label is slightly clipped

        self.fig3.legend(loc='upper right', prop={'size': self.legend_size})  # loc='upper right'

        self.fig3.savefig(self.data_path_source + '_' + self.map_name + '_' + self.policy + '_' + self.cold_storage +
                          '_process_completion_time_and_picker_wait_rate.eps',
                          format='eps')

    def plot_picker_utilisation(self):
        """
        Process completion time and picker utilisation
        :return: None
        """
        ax2 = None
        n_cap = 1
        n_call = 1
        for idx, plot_data in enumerate(self.plot_data_all):
            finish_time_array = np.array(plot_data["sim_finish_time_simpy"])
            finish_time_array = np.rot90(finish_time_array)
            self.finish_time_array = finish_time_array

            finish_time_array_mean = np.mean(finish_time_array, axis=0)
            finish_time_array_mean = finish_time_array_mean.tolist()
            finish_time_array_mean_median = [finish_time_array_mean[0] for x in range(len(finish_time_array_mean))]

            picker_total_working_times_array = np.array(plot_data["picker_total_working_times_array"])
            picker_total_working_times_array = np.rot90(picker_total_working_times_array)

            n_pickers = plot_data["n_pickers"]
            picker_utilisation = 100 * np.divide(picker_total_working_times_array, finish_time_array * n_pickers)
            picker_utilisation_mean = np.mean(picker_utilisation, axis=0)
            picker_utilisation_mean = picker_utilisation_mean.tolist()
            picker_utilisation_mean_median = [picker_utilisation_mean[0] for x in range(len(picker_utilisation_mean))]

            n_robots = plot_data["n_robots"]
            labels = [str(i) for i in n_robots]
            x_axis = [i for i in n_robots]

            color = self.colors[idx]

            self.ax11.plot(x_axis, finish_time_array_mean,
                           label='Completion time, Tcap={}, Tcall={}'.format(n_cap, n_call),
                           linestyle=':', color=color, linewidth=self.linewidth)

            self.ax11.plot(x_axis, finish_time_array_mean_median,
                           label='No robots({} s), Tcap={}, Tcall={}'.format(round(finish_time_array_mean_median[0], 1), n_cap, n_call),
                           color=color)

            self.ax1_2.plot(x_axis, picker_utilisation_mean, label="Tcap={}, Tcall={}".format(n_cap, n_call),
                            linestyle='-.', color=color, linewidth=self.linewidth)
            m = picker_utilisation_mean_median[0]

            # ONLY FOR ROBOT HIGHWAYS
            if n_call == n_cap:
                n_cap = 4
                n_call = 3

        self.fig11.legend(loc='upper right', prop={'size': self.legend_size})  # loc='upper right'
        self.fig1_2.legend(loc='upper right', prop={'size': self.legend_size})  # loc='upper right'

        self.ax11.set_xlabel('Number of robots', fontdict=self.font)
        self.ax11.set_ylabel('Process completion time (s)', fontdict=self.font)
        self.ax11.tick_params(axis='y', labelcolor='tab:olive', labelsize=self.label_size)
        self.ax11.tick_params(axis='x', labelsize=self.label_size)
        self.ax1_2.set_ylabel('Picker utilisation (%)', fontdict=self.font)  # we already handled the x-label with ax1
        self.ax1_2.set_xlabel('Number of robots', fontdict=self.font)
        self.ax1_2.tick_params(axis='x', labelsize=self.label_size)
        self.ax1_2.tick_params(axis='y', labelcolor='tab:blue', labelsize=self.label_size)
        self.fig11.tight_layout()  # otherwise the right y-label is slightly clipped
        self.fig1_2.tight_layout()

        self.fig11.savefig(
            self.data_path_source + '_' + self.map_name + '_' + self.policy + '_' + self.cold_storage +
            '_process_completion_time.eps',
            format='eps')
        self.fig1_2.savefig(
            self.data_path_source + '_' + self.map_name + '_' + self.policy + '_' + self.cold_storage +
            '_picker_utilisation.eps',
            format='eps')
