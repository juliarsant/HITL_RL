from datetime import date
from demonstrations import trial

games = ["CAR", "LUNAR", "SNAKE"]

def participant_id(game, game_attempt):
    day = str(date.today())
    P_id = "001_" + day.replace("-","") + "_"
    P_id += game + game_attempt
    print(P_id)

def trial(games):
    for i in games:
        P_id = participant_id(i, 1)
        trial(P_id)

