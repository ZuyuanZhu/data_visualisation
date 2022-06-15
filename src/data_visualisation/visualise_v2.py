import yaml
import numpy as np
import pandas as pd
import seaborn as sea
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


class VisualiseCARV2(object):
    """
    A class to visualise the Call_A_Robot device V2 collected from Hatchgate
    """

    def __init__(self, data_path, data_name, start_time, time_end, fig=None, ax=None):
        self.data_path = data_path
        self.data_name = data_name
        self.data = None
        self.time_start = start_time  # Unix time
        self.time_end = time_end
        self.gps_x = []
        self.gps_y = []
        self.status = []
        self.show_cbar = True
        self.invalid_gps = -1.0
        self.gps_y_bound = 48692   # limit the V2 gps to the Hatchgate west, excluding the signal from the east side

        self.display = True

        if fig:
            self.fig = fig
        if ax:
            self.ax = ax
        else:
            self.fig, self.ax = plt.subplots(1, 1, figsize=(16, 8), sharex=False, sharey=False)

        self.get_data()
        self.plot_ht()

    def get_data(self):
        """read data from the log file"""
        with open(self.data_path + "/" + self.data_name, "r") as stream:
            try:
                self.data = stream.readlines()
            except yaml.YAMLError as exc:
                print(exc)

    def plot_ht(self):
        for line in self.data:
            lis = line.split(",")
            if not lis[4] and self.time_start <= int(lis[1]) <= self.time_end \
                    and float(lis[7]) != self.invalid_gps and float(lis[8]) != self.invalid_gps and float(lis[8]) > 0:
                if round(float(lis[8]) * 111139, 1) < self.gps_y_bound:
                    self.gps_x.append(round(float(lis[7]) * 111139, 1))
                    self.gps_y.append(round(float(lis[8]) * 111139, 1))
                    self.status.append(int(lis[5]))

        max_x = round(max(self.gps_x))
        min_x = round(min(self.gps_x))
        max_y = round(max(self.gps_y))
        min_y = round(min(self.gps_y))
        data_status = np.zeros((max_x - min_x,
                                max_y - min_y))

        for i, x in enumerate(self.gps_x):
            for j, y in enumerate(self.gps_y):
                if i == j:
                    data_status[round(x) - min_x - 1, round(y) - min_y - 1] = self.status[i]
                    break

        df_ht = pd.DataFrame(data_status,
                             index=np.linspace(min_x,
                                               max_x - 1,
                                               max_x - min_x,
                                               dtype='int'),
                             columns=np.linspace(min_y,
                                                 max_y - 1,
                                                 max_y - min_y,
                                                 dtype='int'))

        sea.set(font_scale=1.6)

        myColors = ((0.8, 0.0, 0.0, 1.0), (0.0, 0.8, 0.0, 1.0), (0.0, 0.0, 0.8, 1.0))
        cmap = LinearSegmentedColormap.from_list('Custom', myColors, len(myColors))


        if self.show_cbar:
            # get sharp grid back by removing rasterized=True, and save fig as svg format
            self.ax = sea.heatmap(df_ht, cmap=cmap, mask=(df_ht == 0), square=True,
                                  rasterized=True, cbar_kws={"shrink": 0.4})
            self.show_cbar = False
        else:
            # get sharp grid back by removing rasterized=True, and save fig as svg format
            self.ax = sea.heatmap(df_ht, cmap=cmap, mask=(df_ht == 0), square=True,
                                  rasterized=True, cbar_kws={"shrink": 0.4})

        self.ax.tick_params(colors='black', left=False, bottom=False)

        # Manually specify colorbar labelling after it's been generated
        colorbar = self.ax.collections[0].colorbar
        colorbar.set_ticks([-0.667, 0, 0.667])
        colorbar.set_ticklabels(['B', 'A', 'C'])
        plt.rcParams.update({'font.size': 16})

        # y axis upside down
        self.ax.invert_yaxis()

        # set Axis label
        self.ax.set_xlabel('Longitude (m)', fontsize=16)
        self.ax.set_ylabel('Latitude (m)', fontsize=16)

        self.fig.canvas.draw()

        self.fig.tight_layout()

        self.fig.savefig(self.data_path + "/figs/V2_Signal_Heatmap" + ".pdf")

        if self.display:
            plt.show()
