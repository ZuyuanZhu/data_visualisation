import matplotlib.pyplot as plt
import rospy
import datetime
from yaml import safe_load
import seaborn as sea


class LoadMap(object):
    def __init__(self, map_path, n_topo_nav_rows, n_polytunnels, n_farm_rows):
        self.map_path = map_path
        with open(map_path, "r") as f:
            self.tmap2 = safe_load(f)

        self.node_index = {}

        self.row_ids = ["row-%02d" % i for i in range(n_topo_nav_rows)]
        self.n_polytunnels = n_polytunnels
        self.n_farm_rows = n_farm_rows
        self.n_topo_nav_rows = n_topo_nav_rows
        self.head_nodes = {}
        self.row_nodes = {}
        self._row_nodes = []
        self.row_info = {}
        self.local_storage_nodes = {row_id: None for row_id in self.row_ids}

        self.half_rows = set()
        self.set_half_row()

        self.update_node_index()

    def update_node_index(self):
        """once topo_map is received, get the indices of nodes for easy access to node object"""
        for i in range(len(self.tmap2['nodes'])):
            self.node_index[self.tmap2['nodes'][i]['node']['name']] = i

    def get_node(self, node):
        """get_node: Given a node name return its node object.
        Keyword arguments:

        node -- name of the node in topological map"""
        return self.tmap2['nodes'][self.node_index[node]]

    def set_row_info(self, pri_head_nodes, row_nodes):
        """set_row_info: Set information about each row
        {row_id: [pri_head_node, start_node, end_node, local_storage_node, sec_head_node]}

        Also sets
          head_nodes {row_id:[pri_head_node, sec_head_node]}
          row_nodes {row_id:[row_nodes]}

        Keyword arguments:

        pri_head_nodes -- , list
        row_nodes -- , list
        sec_head_nodes -- , list/None
        """
        # TODO: meta information is not queried from the db now.
        # The row and head node names are hard coded now
        # An ugly way to sort the nodes is implemented
        # get_nodes in topological_utils.queries might be useful to get nodes with same tag
        self.head_nodes = {"row-%02d" % (i): [] for i in range(self.n_topo_nav_rows)}
        for i in range(self.n_topo_nav_rows):
            self.head_nodes["row-%02d" % (i)].append(pri_head_nodes[i])

        self.row_nodes = {"row-%02d" % (i): [] for i in range(self.n_topo_nav_rows)}

        rospy.sleep(1)
        for i in range(self.n_topo_nav_rows):
            #            rospy.sleep(0.3)
            for node_name in row_nodes[i]:
                node_found = False
                for node in self.tmap2['nodes']:
                    if node_name == node['node']['name']:
                        self.row_nodes["row-%02d" % (i)].append(node_name)
                        node_found = True
                        break
                if not node_found:
                    msg = "node - %s not found in topological_map topic" % (node_name)
                    rospy.logerr(msg)
                    raise Exception(msg)

        for row_id in self.row_ids:
            # local_storage_nodes should be modified by calling set_local_storages
            self.row_info[row_id] = [self.head_nodes[row_id][0],
                                     self.row_nodes[row_id][0],
                                     self.row_nodes[row_id][-1],
                                     self.local_storage_nodes[row_id]]

    def set_half_row(self):
        # half_rows: rows requiring picking in one direction
        if self.n_polytunnels == 1:
            self.half_rows.add(self.row_ids[0])
            self.half_rows.add(self.row_ids[-1])
        else:
            row_num = 0
            for i in range(self.n_polytunnels):
                row_id = "row-%02d" % row_num
                self.half_rows.add(row_id)
                row_num += self.n_farm_rows[i]
                row_id = "row-%02d" % row_num
                self.half_rows.add(row_id)
                row_num += 1


class VisualiseMap(object):
    """A class to visualise topological map in matplotlib"""

    def __init__(self, topo_graph, data_path, fig=None, ax=None, n_start_row=None, n_end_row=None):
        """initialise the VisualiseMap class

        Keyword arguments:

        topo_graph -- topological map
        n_start_row -- start visualising the map from n_start_row
        n_end_row -- end visualising the map at n_end_row
        """
        self.graph = topo_graph
        self.n_start_row = n_start_row
        self.n_end_row = n_end_row
        self.save_fig = True

        self.linewidth = 0.5
        self.markersize = 1

        self.gps_x0 = 48474.64
        self.gps_y0 = 5692296.92

        if self.save_fig:
            self.fig_name_base = data_path

        if fig:
            self.fig = fig
        else:
            # self.fig = plt.figure(figsize=(16, 10), dpi=100)
            self.fig, self.ax = plt.subplots(1, 1, figsize=(16, 9), sharex=False, sharey=False)

        if ax:
            self.ax = ax
        # else:
        #     self.ax = self.fig.add_subplot(111, frameon=True)

        self.font = {'family': 'serif', 'color': 'red', 'weight': 'bold', 'size': 9, }

        self.static_lines = []

        self.init_plot()

        # show the plot
        plt.show(block=False)

    def close_plot(self):
        """close plot"""
        plt.close(self.fig)

    def init_plot(self):
        """Initialise the plot frame"""
        farm_rows_x, farm_rows_y = [], []
        nav_rows_x, nav_rows_y = [], []
        nav_row_nodes_x, nav_row_nodes_y = [], []
        pri_head_lane_x, pri_head_lane_y = [], []
        pri_head_nodes_x, pri_head_nodes_y = [], []

        local_storage_x, local_storage_y = [], []
        local_storage_nodes = []
        cold_storage_node = None

        for i in range(self.graph.n_topo_nav_rows):
            row_id = self.graph.row_ids[i]
            pri_head_node = self.graph.get_node(self.graph.head_nodes[row_id][0])
            pri_head_nodes_x.append(pri_head_node['node']['pose']['position']['x']+self.gps_x0)
            pri_head_nodes_y.append(pri_head_node['node']['pose']['position']['y']+self.gps_y0)

            for j in range(len(self.graph.row_nodes[row_id])):
                curr_node = self.graph.get_node(self.graph.row_nodes[row_id][j])
                if j == 0:
                    start_node = curr_node
                elif j == len(self.graph.row_nodes[row_id]) - 1:
                    last_node = curr_node
                nav_row_nodes_x.append(curr_node['node']['pose']['position']['x']+self.gps_x0)
                nav_row_nodes_y.append(curr_node['node']['pose']['position']['y']+self.gps_y0)

            nav_rows_x.append(
                (pri_head_node['node']['pose']['position']['x']+self.gps_x0, last_node['node']['pose']['position']['x']+self.gps_x0))
            nav_rows_y.append(
                (pri_head_node['node']['pose']['position']['y']+self.gps_y0, last_node['node']['pose']['position']['y']+self.gps_y0))

            # primary head lane
            #            if (i == 0) or (i == self.graph.n_topo_nav_rows - 1):
            pri_head_lane_x.append(pri_head_node['node']['pose']['position']['x']+self.gps_x0)
            pri_head_lane_y.append(pri_head_node['node']['pose']['position']['y']+self.gps_y0)

            # farm rows
            if i < self.graph.n_topo_nav_rows - 1:
                curr_row_id = self.graph.row_ids[i]
                next_row_id = self.graph.row_ids[i + 1]
                if not (curr_row_id in self.graph.half_rows and next_row_id in self.graph.half_rows):
                    curr_row_start_node = self.graph.get_node(self.graph.row_nodes[curr_row_id][0])
                    curr_row_last_node = self.graph.get_node(self.graph.row_nodes[curr_row_id][-1])
                    next_row_start_node = self.graph.get_node(self.graph.row_nodes[next_row_id][0])
                    next_row_last_node = self.graph.get_node(self.graph.row_nodes[next_row_id][-1])
                    start_node_x = curr_row_start_node['node']['pose']['position']['x']+self.gps_x0 + 0.5 * (
                            next_row_start_node['node']['pose']['position']['x']+self.gps_x0 -
                            (curr_row_start_node['node']['pose']['position']['x']+self.gps_x0))
                    start_node_y = curr_row_start_node['node']['pose']['position']['y']+self.gps_y0 + 0.5 * (
                            next_row_start_node['node']['pose']['position']['y']+self.gps_y0 -
                            (curr_row_start_node['node']['pose']['position']['y']+self.gps_y0))
                    last_node_x = curr_row_last_node['node']['pose']['position']['x']+self.gps_x0 + 0.5 * (
                            next_row_last_node['node']['pose']['position']['x']+self.gps_x0 -
                            (curr_row_last_node['node']['pose']['position']['x']+self.gps_x0))
                    last_node_y = curr_row_last_node['node']['pose']['position']['y']+self.gps_y0 + 0.5 * (
                            next_row_last_node['node']['pose']['position']['y']+self.gps_y0 -
                            (curr_row_last_node['node']['pose']['position']['y']+self.gps_y0))

                    farm_rows_x.append((start_node_x, last_node_x))
                    farm_rows_y.append((start_node_y, last_node_y))

            # if self.graph.local_storage_nodes[row_id] not in local_storage_nodes:
            #     local_storage_nodes.append(self.graph.local_storage_nodes[row_id])
            #     node_obj = self.graph.get_node(local_storage_nodes[-1])
            #     local_storage_x.append(node_obj['node']['pose']['position']['x']+self.gps_x0)
            #     local_storage_y.append(node_obj['node']['pose']['position']['y']+self.gps_y0)

            # if self.graph.cold_storage_node is not None:
            #     cold_storage_node = self.graph.cold_storage_node
            #     node_obj = self.graph.get_node(cold_storage_node)
            #     cold_storage_x = node_obj['node']['pose']['position']['x']+self.gps_x0
            #     cold_storage_y = node_obj['node']['pose']['position']['y']+self.gps_y0

        min_x = min(min(nav_rows_x[0]), min(farm_rows_x[0]))
        max_x = max(max(nav_rows_x[-1]), max(farm_rows_x[-1]))
        min_y = min(min(nav_rows_y[0]), min(farm_rows_y[0]))
        max_y = max(max(nav_rows_y[-1]), max(farm_rows_y[-1]))

        # limits of the axes
        # self.ax.set_xlim(min_x - 5, max_x + 2.5)
        # self.ax.set_ylim(min_y - 2.5, max_y + 7.5)

        # static objects - nodes
        # nav_rows
        for i, item in enumerate(zip(nav_rows_x, nav_rows_y)):
            self.static_lines.append(self.ax.plot(item[0], item[1],
                                                     color="black", linewidth=self.linewidth)[0])
        # farm_rows
        # for i, item in enumerate(zip(farm_rows_x, farm_rows_y)):
        #     self.static_lines.append(self.ax.plot(item, item[1],
        #                                              color="green", linewidth=4)[0])
        # primary head lane
        self.static_lines.append(self.ax.plot(pri_head_lane_x, pri_head_lane_y,
                                                 color="black", linewidth=self.linewidth)[0])

        # nav_row_nodes
        self.static_lines.append(self.ax.plot(nav_row_nodes_x, nav_row_nodes_y,
                                                 color="black", marker="o", markersize=self.markersize,
                                                 linestyle="none")[0])
        # pri_head_lane_nodes
        self.static_lines.append(self.ax.plot(pri_head_nodes_x, pri_head_nodes_y,
                                                 color="black", marker="o", markersize=self.markersize,
                                                 linestyle="none")[0])

        # local storages
        # self.static_lines.append(self.ax.plot(local_storage_x, local_storage_y,
        #                                          color="black", marker="s", markersize=12,
        #                                          markeredgecolor="r", linestyle="none")[0])

        # plt.show()

        self.ax.tick_params(axis='x', labelsize=9)
        self.ax.tick_params(axis='y', labelsize=9)

        # y axis upside down
        # self.ax.invert_yaxis()
        # for static_line in self.static_lines:
        #     # self.ax.add_line(static_line)
        #     # static_line.plot(kind='line', ax=self.ax, label="Map")
        #     sea.lineplot(data=static_line, linewidth=self.linewidth, ax=self.ax)

        self.fig.canvas.draw()
        if self.save_fig:
            self.fig.savefig(
                self.fig_name_base + '/' + datetime.datetime.now().isoformat().replace(":", "_") + ".pdf")

        return self.static_lines
