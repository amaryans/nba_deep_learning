from Game import Game
import argparse
from os import listdir
import os

parser = argparse.ArgumentParser(description='Process arguments about an NBA game.')
parser.add_argument('--path', type=str,
                    help='a path to json file to read the events from',
                    required = True)
parser.add_argument('--event', type=int, default=0,
                    help="""an index of the event to create the animation to
                            (the indexing start with zero, if you index goes beyond out
                            the total number of events (plays), it will show you the last
                            one of the game)""")

args = parser.parse_args()

cur_dir = listdir('/Users/amaryans/Documents/school/spring22/cosc424/projects/NBA-Player-Movements/data/2016.NBA.Raw.SportVU.Game.Logs/')
index = 1
for file in cur_dir:
        if index < 10:
                if file.endswith(".json"):
                        print(file)
                        game = Game(path_to_json='/Users/amaryans/Documents/school/spring22/cosc424/projects/NBA-Player-Movements/data/2016.NBA.Raw.SportVU.Game.Logs/' + str(file), event_index=args.event)
                        game.read_json()
                        os.system("mv " + '/Users/amaryans/Documents/school/spring22/cosc424/projects/NBA-Player-Movements/data/2016.NBA.Raw.SportVU.Game.Logs/' + str(file) + ' /Users/amaryans/Documents/school/spring22/cosc424/projects/NBA-Player-Movements/data/2016.NBA.Raw.SportVU.Game.Logs/used_json')
        else:
                continue
#game.start()
