#!/usr/bin/env python

# ----------------------------------
# @author: Zuyuan
# @email: zuyuanzhu@gmail.com
# @date: 28-Jan-2022
# @info: Visualise the results of picking and transporting tasks by pickers and robots in rasberry_des
#        Run with different maps/configfile, same n_picker and n_robot. Compare the robot service distance,
#        robot service time etc. in different optimisation strategies.
# data preparation:
#       1. run rasberry_des simpy branch, get data from different topo map
#       2. place all event yaml files into one folder
#       3. run this script with the above folder
# ----------------------------------

import data_visualisation.visualise_service

N_ITERATION = 10


if __name__ == "__main__":

    folder = "2022_Feb_02_17_38_29_clockhouse_transportation_additional_lanes_tmap2_TRO_Oct_Service_times"
    data_folder_path = "/home/zuyuan/des_simpy_logs/"
    data_path = data_folder_path + folder
    file_type = 'yaml'
    vis = data_visualisation.visualise_service.Visualise(data_path, file_type, N_ITERATION)
