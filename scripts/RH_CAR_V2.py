#!/usr/bin/env python

# ----------------------------------
# @author: Zuyuan
# @email: zuyuanzhu@gmail.com
# @date: 15-June-2022
# @info: data visualisation for GPS collected from
#        Hatchgate
# ----------------------------------

import data_visualisation.visualise_v2

if __name__ == "__main__":
    folder = "RH_HatchgateWest_CAR_V2"
    data_folder_path = "/home/zuyuan/rasberry_ws/src/data_visualisation/results/"
    data_path = data_folder_path + folder
    data_name = 'car_filtered.log'
    start_time = 1654595910
    end_time = 1654784162
    vis = data_visualisation.visualise_v2.VisualiseCARV2(data_path, data_name, start_time, end_time)
