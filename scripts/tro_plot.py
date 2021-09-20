#!/usr/bin/env python

# ----------------------------------
# @author: Zuyuan
# @email: zuyuanzhu@gmail.com
# @date: 19-May-2021
# @info: Visualise the results of picking and transporting tasks by pickers and robots in rasberry_des
# ----------------------------------

import data_visualisation.visualise

N_ITERATION = 10


if __name__ == "__main__":

    folder = "2021_Sep_17_16_52_25"
    data_folder_path = "/home/zuyuan/des_logs/"
    data_path = data_folder_path + folder
    file_type = 'yaml'
    vis = data_visualisation.visualise.Visualise(data_path, file_type, N_ITERATION)
