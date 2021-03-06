#!/usr/bin/env python

# ----------------------------------
# @author: Zuyuan
# @email: zuyuanzhu@gmail.com
# @date: 13-June-2022
# @info: data visualisation for GPS, Mobile Signal Strength from rosbag collected from
#        Hatchgate
# ----------------------------------

import data_visualisation.visualise_signal

if __name__ == "__main__":
    folder = "robot2_bags"
    data_folder_path = "/home/zuyuan/rasberry_ws/src/RASberry/rasberry_core/new_tmule/"
    data_path = data_folder_path + folder
    file_type = 'bag'
    outputbag = 'merge_bag.bag'
    vis = data_visualisation.visualise_signal.VisualiseSignal(data_path, file_type, outputbag)
    generations = [2, 3, 4]
    vis.plot(generations)
    vis.close_fig()
