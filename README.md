Human-In-The-Loop Reinforcement Learning Using Human-Machine Techniques
By: Julia Santaniello

This program runs multiple RL learning algorithms on various games. It allows for saving human demonstrations and training artificial agents with said demonstrations.

Be prepared to Press "Enter" and play Lunar Lander if demonstrating. Lunar Lander has 4 discrete actions: Up, Down, Left-Engine, Right-Engine. These will be the respective arrow keys.

TO RUN:
python3 main.py

Make sure you have "run_all" set to True for the first time. This will allow you to run demonstrations, HITL-PG and PG training at once.

"Human"= True will allow you to only run HITL-PG

Anything else will run a normal Policy Gradient training on Lunar Lander.

imports.py should have all necessary variables that can be adjusted.

Software:
Python 3.8.5
Anaconda 
Numpy
Pandas
OpenAi Gymnasium
matplotlib.pyplot

