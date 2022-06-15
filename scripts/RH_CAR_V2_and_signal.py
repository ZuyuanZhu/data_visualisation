#!/usr/bin/env python

# ----------------------------------
# @author: Zuyuan
# @email: zuyuanzhu@gmail.com
# @date: 15-June-2022
# @info: data visualisation for GPS collected from
#        Hatchgate
# ----------------------------------
import data_visualisation.visualise_v2_and_signal
import matplotlib.pyplot as plt


if __name__ == "__main__":

    fig, ax = plt.subplots(1, 1, figsize=(16, 8), sharex=False, sharey=False)

    folder = "robot2_bags"
    data_folder_path = "/home/zuyuan/rasberry_ws/src/RASberry/rasberry_core/new_tmule/"
    data_path = data_folder_path + folder
    file_type = 'bag'
    outputbag = 'merge_bag.bag'
    vis = data_visualisation.visualise_v2_and_signal.VisualiseSignal(data_path, file_type, outputbag, fig, ax)
    generations = [2]
    vis.plot(generations)

    folder = "RH_HatchgateWest_CAR_V2"
    data_folder_path = "/home/zuyuan/rasberry_ws/src/data_visualisation/results/"
    data_path = data_folder_path + folder
    data_name = 'car_filtered.log'
    start_time = 1654595910
    end_time = 1654784162
    vis = data_visualisation.visualise_v2_and_signal.VisualiseCARV2(data_path, data_name, start_time, end_time, fig, ax)
