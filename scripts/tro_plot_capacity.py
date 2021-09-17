#!/usr/bin/env python

# ----------------------------------
# @author: Zuyuan
# @email: zuyuanzhu@gmail.com
# @date: 15-Sep-2021
# @info: Visualise the results of picking and transporting tasks by pickers and robots in rasberry_des,
#        using different tray capacity and calling moment
# ----------------------------------

import data_visualisation.visualise_capacity
import os

N_ITERATION = 10

if __name__ == "__main__":
    data_path = "/home/zuyuan/des_logs/4Polytun_nTcap_nTcall/"
    data_folder_list = []
    for name in os.listdir(data_path):
        if os.path.isdir(data_path + name):
            for sample_name in os.listdir(data_path + name):
                if os.path.isdir(data_path + name + '/' + sample_name):
                    data_folder_list.append(data_path + name + '/' + sample_name + '/')

    data_folder_list.sort()
    print(data_folder_list)
    file_type = 'yaml'
    visualisation = data_visualisation.visualise_capacity.VisualiseCapacity(data_path, data_folder_list, file_type, N_ITERATION)
