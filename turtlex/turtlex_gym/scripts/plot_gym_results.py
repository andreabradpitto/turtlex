#!/usr/bin/env python3

import sys
import rospkg
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import itertools
import sys
import argparse
from gym.wrappers.monitor import load_results
from scipy.interpolate import pchip


class LivePlot(object):
    def __init__(self, outdir, data_key='episode_rewards', line_color='blue'):
        """
        Liveplot renders a graph of either episode_rewards or episode_lengths
        Args:
            outdir (outdir): Monitor output file location used to populate the graph
            data_key (Optional[str]): The key in the json to graph (episode_rewards or episode_lengths).
            line_color (Optional[dict]): Color of the plot.
        """

        # data_key can be set to 'episode_lengths'
        self.outdir = outdir
        self._last_data = None
        self.data_key = data_key
        self.line_color = line_color

        # styling options
        matplotlib.rcParams['toolbar'] = 'None'
        plt.style.use('ggplot')
        plt.xlabel("episodes")
        plt.ylabel(self.data_key.replace("_", " "))
        fig = plt.gcf().canvas.manager.set_window_title('OpenAI Gym results plot - ' + str(self.data_key))
        matplotlib.rcParams.update({'font.size': 15})

    def expand(lst, n):
        lst = [[i] * n for i in lst]
        lst = list(itertools.chain.from_iterable(lst))
        return lst

    def plot(self, full=True, dots=False, average=0, interpolated=0):
        results = load_results(self.outdir)
        data =  results[self.data_key]

        if full:
            plt.plot(data, color=self.line_color)

        if dots:
            plt.plot(data, '.', color=self.line_color)

        if average > 0:
            avg_data = []
            average = int(average)
            for i, val in enumerate(data):
                if i % average == 0:
                    if (i + average) < len(data) + average:
                        avg =  sum(data[i : i + average]) / average
                        avg_data.append(avg)

            new_data = self.expand(avg_data,average)
            plt.plot(new_data, color=self.line_color, linewidth=2.5)
 
        if interpolated > 0:
            avg_data = []
            avg_data_points = []
            n = len(data) / interpolated
            if n == 0:
                n = 1
            data_fix = 0
            for i, val in enumerate(data):
                if i % n==0:
                    if (i + n) <= len(data) + n:
                        avg = sum(data[i : i + n]) / n
                        avg_data.append(avg)
                        avg_data_points.append(i)
                if (i + n) == len(data):
                    data_fix = n
            
            x = np.arange(len(avg_data))
            y = np.array(avg_data)

            interp = pchip(avg_data_points, avg_data)
            xx = np.linspace(0, len(data) - data_fix, 1000)
            plt.plot(xx, interp(xx), color=self.line_color, linewidth=3.5)        

        uuid = results['manifests']
        _, _, uuid1, uuid2, _, _ = str(uuid).split(".")
        uuid = uuid1 + "." + uuid2
        return uuid


if __name__ == '__main__':

    # Let the user choose which expriment results to plot
    world_input = input("Type world name [office]: ").lower().strip()
    print("")
    task_input = input("Type task name [nav, arm]: ").lower().strip()
    print("")
    alg_input = input("Type algorithm name [nav -> sac, arm -> ddpg]: ").lower().strip()
    print("")
    user_input = world_input + '_' + task_input + '_' + alg_input

    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('results')
    outdir = pkg_path + '/gym/' + user_input

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--full", action='store_true',\
                         help="print the full data plot with lines")
    parser.add_argument("-d", "--dots", action='store_true',\
                         help="print the full data plot with dots")
    parser.add_argument("-a", "--average", type=int, nargs='?', const=50, metavar="N",\
                         help="plot an averaged graph using N as average size delimiter. Default = 50")
    parser.add_argument("-i", "--interpolated", type=int, nargs='?', const=50, metavar="M",\
                         help="plot an interpolated graph using M as interpolation amount. Default = 50")
    args = parser.parse_args()


    ########## Episode rewards plot
    rew_plotter = LivePlot(outdir)

    if len(sys.argv) == 1:
        # When no arguments are given, plot full data graph
        uuid = rew_plotter.plot(full=True)
    else:
        uuid = rew_plotter.plot(full=args.full, dots=args.dots, average=args.average, interpolated=args.interpolated)

    plt_save_path = outdir + "/" + "gym_plot_ep_rewards_" + uuid + ".png"
    plt.savefig(plt_save_path)
    print ("Saved episode rewards plot in " + plt_save_path)
    print("Opening episode rewards plot in Graphic Tools Window")

    plt.show()


    ########## Episode lengths plot
    len_plotter = LivePlot(outdir, data_key='episode_lengths', line_color='green')

    if len(sys.argv) == 1:
        # When no arguments are given, plot full data graph
        uuid = len_plotter.plot(full=True)
    else:
        uuid = len_plotter.plot(full=args.full, dots=args.dots, average=args.average, interpolated=args.interpolated)

    plt_save_path = outdir + "/" + "gym_plot_ep_lengths_" + uuid + ".png"
    plt.savefig(plt_save_path)
    print ("Saved episode lengths plot in " + plt_save_path)
    print("Opening episode lengths plot in Graphic Tools Window")

    plt.show()
