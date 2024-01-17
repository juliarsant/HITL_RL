from hitl_train import run
from train import run_train
from data import plot
from imports import trials, run_demos

"""
Julia Santaniello
Started: 06/01/23
Last Updated: 12/21/23

Main File. Starts demonstrations, runs training, and runs HTIL training.
"""
human = True #HITL?
run_all = True #Run all: Training, Demos, HITL Training

if __name__=="__main__":
    if run_all == True:
        run_train(trials=trials)
        run(trials=trials, run_demos=run_demos)
        plot()
    elif human == True:
        run(trials=trials, run_demos=run_demos)
        plot()
    else:
        run_train(trials=trials)

