#!/usr/bin/env python

# ----------------------------------
# @author: Zuyuan
# @email: zuyuanzhu@gmail.com
# @date: 17-June-2022
# @info: data visualisation for GPS, Mobile Signal Strength from rosbag collected from
#        Hatchgate
# ----------------------------------

import os
import data_visualisation.visualise_signal
import data_visualisation.visualise_map
import rasberry_des.config_utils_sim


class InitTopoMap(object):

    def __init__(self, config_dir_, config_file_name_, map_dir_, map_file_name_):
        self.config_file = config_dir_ + config_file_name_
        self.config_params = rasberry_des.config_utils_sim.get_mimic_des_params(self.config_file)

        self.map_name = self.config_params["map_name"]
        self.n_polytunnels = self.config_params["n_polytunnels"]
        self.n_farm_rows = self.config_params["n_farm_rows"]
        self.n_topo_nav_rows = self.config_params["n_topo_nav_rows"]
        self.second_head_lane = self.config_params["second_head_lane"]
        self.n_pickers = self.config_params["n_pickers"]
        self.tray_capacity = self.config_params["tray_capacity"]
        self._yield_per_node = self.config_params["yield_per_node"]
        self.n_local_storages = self.config_params["n_local_storages"]
        self.ignore_half_rows = self.config_params["ignore_half_rows"]
        self.pri_head_nodes = self.config_params["pri_head_nodes"]
        self.row_nodes = self.config_params["row_nodes"]
        self.env = False
        self.use_cold_storage = self.config_params["use_cold_storage"]  # set in config_file
        if self.use_cold_storage:
            self.cold_storage_nodes = self.config_params["cold_storage_nodes"]

        self.config_dir = config_dir_
        self.config_file_name = config_file_name_
        self.config_file = self.config_dir + self.config_file_name
        self.map_dir = map_dir_
        self.map_file_name = map_file_name_
        self.topo_map2_file = os.path.join(os.path.expanduser("~"), self.map_dir, self.map_file_name)
        self.single_track = []
        self.VERBOSE = True
        self.debug = True

        self.topo_graph = data_visualisation.visualise_map.LoadMap(self.topo_map2_file,
                                                                   self.n_topo_nav_rows,
                                                                   self.n_polytunnels,
                                                                   self.n_farm_rows)
        self.topo_graph.set_row_info(self.pri_head_nodes, self.row_nodes)

    def get_map(self):
        return self.topo_graph


if __name__ == "__main__":

    folder = "robot2_bags"
    data_folder_path = "/home/zuyuan/rasberry_ws/src/RASberry/rasberry_core/new_tmule/"
    data_path = data_folder_path + folder

    map_dir = '/home/zuyuan/rasberry_ws/src/RASberry/rasberry_core/config/site_files/clockhouse/hatchgate_west/transportation'
    map_file_name = 'tmap.tmap2'
    config_dir = '/home/zuyuan/rasberry_ws/src/RASberry/rasberry_des/config/toptimise_kent/'
    config_file_name = 'RH_clockhouse_hatchgate_west.yaml'
    init_map = InitTopoMap(config_dir, config_file_name, map_dir, map_file_name)
    tmap = init_map.get_map()


    vis_map = data_visualisation.visualise_map.VisualiseMap(tmap, data_path)
    fig = vis_map.fig
    ax = vis_map.ax

    # file_type = 'bag'
    # outputbag = 'merge_bag.bag'
    # vis = data_visualisation.visualise_signal.VisualiseSignal(data_path, file_type, outputbag, fig, ax)
    # generations = [2, 3, 4]
    # vis.plot(generations)
    # vis.close_fig()
