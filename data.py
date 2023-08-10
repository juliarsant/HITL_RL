import numpy as np
import csv
import glob
import pandas as pd
import matplotlib.pyplot as plt

def running_mean(x):
    N= 10
    kernel = np.ones(N)
    conv_len = x.shape[0]-N
    y = np.zeros(conv_len)
    for i in range(conv_len):
        y[i] = kernel @ x[i:i+N]
        y[i] /= N
    return y

def analysis(path, i, versions: bool):

    # csv files in the path
    files = glob.glob(path)

    #graph titles
    if i == 0:
        title = "Rewards Over Episodes"
    if i == 1:
        title = "Steps Over Episodes"
    if i == 2:
        title = "Mean Over Episodes"
    if i == 3:
        title = "Accuracy Over Episodes"

    #empty frames
    data_frame1 = pd.DataFrame()
    data_frame2 = pd.DataFrame()
    if not versions: data_frame3 = pd.DataFrame() #for trainParameters.py


    content = []

    # checking all the csv files in the
    # specified path
    for filename in files:
        # reading content of csv file
        df = pd.read_csv(filename, header=None)
        data_frame1 = data_frame1.append(df.iloc[[0]])
        data_frame2 = data_frame2.append(df.iloc[[1]])
        #for trainParameters.py
        if not versions: data_frame3 = data_frame3.append(df.iloc[[2]])

    # standard deviations in dataframe
    std1_frame = data_frame1.std(axis = 0)
    std2_frame = data_frame2.std(axis = 0)
    # for trainParameters.py
    if not versions: std3_frame = data_frame3.std(axis = 0)

    #dataframe means
    data_frame1 = data_frame1.mean(axis=0)
    data_frame2 = data_frame2.mean(axis=0)
    #for trainParamters.py
    if not versions: data_frame3 = data_frame3.mean(axis=0)

    #x values for fill_between()
    x = [i for i in range(len(data_frame1))]

    #Plotting
    data_frame1.plot()
    plt.fill_between(x, data_frame1 - std1_frame, data_frame1 + std1_frame, color = '#57A9FF', alpha = 0.3)
    data_frame2.plot()
    plt.fill_between(x, data_frame2 - std2_frame, data_frame2 + std2_frame, color = '#FFAC00', alpha = 0.5)
    #for trainParamters.py
    if not versions: 
        data_frame3.plot()
        plt.fill_between(x, data_frame3 - std3_frame, data_frame3 + std3_frame, color = '#008000', alpha = 0.2)
        plt.legend(["0.007", "0.01", "0.013"])

    plt.title(title)
    #for rewards graphs (training and midtraining means)
    if i == 0 or i == 2:
        plt.ylim(-500, 250)

    plt.legend(["MODIFIED", "ORIGINAL"])
    plt.show()


def data_collection(rewards1:list, rewards2:list, rewards3:list, steps1:list, 
           steps2:list, steps3:list, training_data1, training_data2, 
           training_data3, folder_name:str, trial:int):
    
    """
    Collecting the data from trainParameters.py
    and putting them into folders
    """

    #Rewards running mean
    rewards_list1 = np.array(rewards1)
    rewards_list2 = np.array(rewards2)
    rewards_list3 = np.array(rewards3)
    avg_rewards1 = running_mean(rewards_list1)
    avg_rewards2 = running_mean(rewards_list2)
    avg_rewards3 = running_mean(rewards_list3)

    #Steps running mean
    steps_list1 = np.array(steps1)
    steps_list2 = np.array(steps2)
    steps_list3 = np.array(steps3)
    avg_steps1 = running_mean(steps_list1)
    avg_steps2 = running_mean(steps_list2)
    avg_steps3 = running_mean(steps_list3)

    #Training average Rewards
    train_rewards1 = training_data1[1]
    train_rewards2 = training_data2[1]
    train_rewards3 = training_data3[1]

    #Training std
    train_std1 = training_data1[2]
    train_std2 = training_data2[2]
    train_std3 = training_data3[2]

    #Training accuracy
    train_accuracy1 = training_data1[3]
    train_accuracy2 = training_data2[3]
    train_accuracy3 = training_data3[3]

    #For creating future dataframes
    steps_rows = [avg_steps1, avg_steps2, avg_steps3]
    rewards_rows = [avg_rewards1, avg_rewards2, avg_rewards3]
    trainingMean_rows = [train_rewards1, train_rewards2, train_rewards3]
    trainingStd_rows = [train_std1, train_std2, train_std3]
    trainingAcc_rows = [train_accuracy1, train_accuracy2, train_accuracy3]

    #saving csv files
    np.savetxt('./data/{}/steps/stepsData_{}.csv'.format(folder_name, trial), steps_rows, delimiter = ", ")
    np.savetxt('./data/{}/rewards/rewardsData_{}.csv'.format(folder_name, trial), rewards_rows, delimiter = ", ")
    np.savetxt('./data/{}/trainingData/mean/mean_{}.csv'.format(folder_name, trial), trainingMean_rows, delimiter = ", ")
    np.savetxt('./data/{}/trainingData/accuracy/acc_{}.csv'.format(folder_name, trial), trainingAcc_rows, delimiter = ", ")
    np.savetxt('./data/{}/trainingData/std/std_{}.csv'.format(folder_name, trial), trainingStd_rows, delimiter = ", ")



def data_collection_versions(rewards1:list, rewards2:list, steps1:list, 
           steps2:list, training_data1, training_data2, folder_name:str, trial:int):
    
    """
    Collecting data from trainVersions.py and putting them into folders
    """
    
    
    #Rewards running mean
    rewards_list1 = np.array(rewards1)
    rewards_list2 = np.array(rewards2)
    avg_rewards1 = running_mean(rewards_list1)
    avg_rewards2 = running_mean(rewards_list2)

    #Steps running mean
    steps_list1 = np.array(steps1)
    steps_list2 = np.array(steps2)
    avg_steps1 = running_mean(steps_list1)
    avg_steps2 = running_mean(steps_list2)

    #Training average Rewards
    train_rewards1 = training_data1[0]
    train_rewards2 = training_data2[0]

    #Training std
    train_std1 = training_data1[1]
    train_std2 = training_data2[1]

    #Training accuracy
    train_accuracy1 = training_data1[2]
    train_accuracy2 = training_data2[2]

    steps_rows = [avg_steps1, avg_steps2]
    rewards_rows = [avg_rewards1, avg_rewards2]
    trainingMean_rows = [train_rewards1, train_rewards2]
    trainingStd_rows = [train_std1, train_std2]
    trainingAcc_rows = [train_accuracy1, train_accuracy2]

    #saving csv files
    np.savetxt('./data/{}/steps/stepsData_{}.csv'.format(folder_name, trial), steps_rows, delimiter = ", ")
    np.savetxt('./data/{}/rewards/rewardsData_{}.csv'.format(folder_name, trial), rewards_rows, delimiter = ", ")
    np.savetxt('./data/{}/trainingData/mean/mean_{}.csv'.format(folder_name, trial), trainingMean_rows, delimiter = ", ")
    np.savetxt('./data/{}/trainingData/accuracy/acc_{}.csv'.format(folder_name, trial), trainingAcc_rows, delimiter = ", ")
    np.savetxt('./data/{}/trainingData/std/std_{}.csv'.format(folder_name, trial), trainingStd_rows, delimiter = ", ")

def run_analysis(folder, versions):
    paths = ['./data/{}/rewards/*'.format(folder),
             './data/{}/steps/*'.format(folder),
             './data/{}/trainingData/mean/*'.format(folder),
             './data/{}/trainingData/accuracy/*'.format(folder)]
    for i in range(len(paths)):
        analysis(paths[i], i, versions)

if __name__ == '__main__':
    run_analysis(folder = "PreExperiment6", versions = True)

#Train Episodes
#every k Episodes, have agent play game without exploration with policy
#Create test curve with k episodes
#Standard deviation
#plot next to the non moving goal posts