#!/usr/bin/env python

# ----------------------------------
# @author: Zuyuan
# @email: zuyuanzhu@gmail.com
# @date: 15-June-2022
# @info: data visualisation for V2 and robot GPS collected from
#        Hatchgate west
# ----------------------------------
import data_visualisation.simpy_robot_picker_GPS
import matplotlib.pyplot as plt
import simpy

SHOW_VIS = True
SIM_RT_FACTOR = 0.01

if __name__ == "__main__":

    # fig, ax = plt.subplots(1, 1, figsize=(15, 15), sharex=False, sharey=False)

    folder = "picker"
    data_folder_path = "/home/zuyuan/rasberry_ws/src/datasets/riseholme/"
    data_path = data_folder_path + folder
    data_name = 'car_riseholme_0508.log'
    start_time = 1659684214
    end_time = 1659694010

    # robot_gps_x = [59201857.3, 59202346.5]   # dm
    # robot_gps_y = [-582798.0, -582346.5]
    robot_gps_x = [5920185.7, 5920234.65]
    robot_gps_y = [-58279.80, -58234.65]   # m
    user = ['STD_v2_bcddc2cfcb68', 'STD_v2_246f284a6c94']

    # env = simpy.RealtimeEnvironment(initial_time=0., factor=SIM_RT_FACTOR, strict=False)
    env = simpy.Environment()

    vis = data_visualisation.simpy_robot_picker_GPS.VisualiseCARV2(env, user, data_path, data_name,
                                                                   start_time, end_time,
                                                                   robot_gps_x, robot_gps_y)

    if SHOW_VIS:
        n = 0
        while n < 1000:
            try:
                env.step()
                n += 1
            except simpy.core.EmptySchedule:
                break
            else:
                pass


