#!/usr/bin/env python

# ----------------------------------
# @author: Zuyuan
# @email: zuyuanzhu@gmail.com
# @date: 15-June-2022
# @info: data visualisation for V2 and robot GPS collected from
#        Hatchgate west
# ----------------------------------
import data_visualisation.visualise_v2_and_robot_gps
import matplotlib.pyplot as plt

# import data_visualisation.visualise_v2


if __name__ == "__main__":

    fig, ax = plt.subplots(1, 1, figsize=(25, 25), sharex=False, sharey=False)
    #
    # folder = "all_valid"
    # data_folder_path = "/home/zuyuan/rasberry_ws/src/datasets/hatchgate_west/"
    # data_path = data_folder_path + folder
    # file_type = 'bag'
    # outputbag = 'robot_GPS.bag'
    # car_topic = '/car_client/get_gps'
    # robot_topic = '/thorvald_024/gps/filtered'
    #
    # vis = data_visualisation.visualise_v2_and_robot_gps.VisualiseSignal(data_path, file_type, outputbag,
    #                                                                     car_topic, robot_topic,
    #                                                                     fig, ax)
    # generations = [2]
    # vis.plot(generations)

    folder = "picker"
    data_folder_path = "/home/zuyuan/rasberry_ws/src/datasets/riseholme/"
    data_path = data_folder_path + folder
    data_name = 'car_riseholme_0508.log'
    start_time = 1659684214   # 1658406816
    end_time = 1659694010
    # user = ['STD_v2_bcddc2cfcb68', 'STD_v2_246f284a6c94']
    # robot_gps_x = [56924293.0, 56923437.9]
    # robot_gps_y = [486343.8, 485823.1]

    # Riseholme map
    # 53.26872135235208, -0.5243865725205243
    # 53.26874135998218, -0.5240974337695075
    # 53.26830118995692, -0.5243172748033385
    # 53.268332630823295, -0.5239803445232278
    robot_gps_x = [59201857.3, 59202346.5]
    robot_gps_y = [-582798.0, -582346.5]
    user = ['STD_v2_bcddc2cfcb68', 'STD_v2_246f284a6c94']

    vis2 = data_visualisation.visualise_v2_and_robot_gps.VisualiseCARV2(user, data_path, data_name,
                                                                        start_time, end_time,
                                                                        # vis.df_robot_gps["x"], vis.df_robot_gps["y"],
                                                                        robot_gps_x, robot_gps_y,
                                                                        fig, ax)
