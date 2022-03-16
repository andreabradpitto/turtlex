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

        # Styling options
        matplotlib.rcParams['toolbar'] = 'None'
        plt.style.use('ggplot')
        plt.xlabel("episodes")
        plt.ylabel(self.data_key.replace("_", " "))
        plt.gcf().canvas.manager.set_window_title('OpenAI Gym results plot - ' + str(self.data_key))
        matplotlib.rcParams.update({'font.size': 15})

    def plot(self, full=True, dots=False):
        results = load_results(self.outdir)
        data =  results[self.data_key]

        if full:
            plt.plot(data, color=self.line_color)

        if dots:
            plt.plot(data, '.', color=self.line_color)

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
    args = parser.parse_args()


    ########## Episode rewards plot
    rew_plotter = LivePlot(outdir)

    if len(sys.argv) == 1:
        # When no arguments are given, plot full data graph
        uuid = rew_plotter.plot(full=True)
    else:
        uuid = rew_plotter.plot(full=args.full, dots=args.dots)

    plt_save_path = outdir + "/" + "gym_plot_ep_rewards_" + uuid + ".png"
    plt.savefig(plt_save_path)
    print ("\nSaved episode rewards plot in " + plt_save_path)
    print("Opening episode rewards plot in Graphic Tools Window\n")

    plt.show()


    ########## Episode lengths plot
    len_plotter = LivePlot(outdir, data_key='episode_lengths', line_color='green')

    if len(sys.argv) == 1:
        # When no arguments are given, plot full data graph
        uuid = len_plotter.plot(full=True)
    else:
        uuid = len_plotter.plot(full=args.full, dots=args.dots)

    plt_save_path = outdir + "/" + "gym_plot_ep_lengths_" + uuid + ".png"
    plt.savefig(plt_save_path)
    print("\nSaved episode lengths plot in " + plt_save_path)
    print("Opening episode lengths plot in Graphic Tools Window\n")

    plt.show()
