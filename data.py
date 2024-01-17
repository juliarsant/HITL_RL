import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imports import data_folder_name, graph_path, demo_name, demo_name_hitl 
import pandas as pd

'''
Julia Santaniello
Started: 06/01/23
Last Updated: 12/21/23

Graphs and saves data from experiments. Can choose the titles for the graph, axes, legends.
Can save data by inserting an experiment code.
'''

"""
Plot all data in the path
"""
def plot():
    color = ["skyblue", "lightgreen", "palered", "dimgrey"]
    # csv files in the path
    name = data_folder_name+demo_name_hitl+".csv"
    name2 = data_folder_name+"train.csv"
    
    data_hitl = pd.read_csv(name)
    data_train = pd.read_csv(name2)


    data_hitl = np.asarray(data_hitl)
    data_train = np.asarray(data_train)
    data_hitl=data_hitl[:,1:]
    data_train=data_train[:,1:]

    assert(len(data_hitl)==len(data_train))

    assert(len(data_hitl[0])==len(data_train[0]))

    x = [i for i in range(len(data_hitl[0]))]

    for i in range(len(data_train)):

        plt.plot(x, data_hitl[i], color = color[1])
        plt.plot(x, data_train[i], color = color[0])
    
        if i == 0:
            plt.title("Average Rewards over Episodes")
            plt.legend(["PG-HITL", "PG"])
        elif i == 1:
            plt.title("Average Steps over Episodes")
            plt.legend(["PG-HITL", "PG"])
        elif i == 2:
            plt.title("Averages over Episodes")
            plt.legend(["PG-HITL", "PG"])

        plt.show()

if __name__=="__main__":
    plot()
