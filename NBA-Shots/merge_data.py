### Need to open moment_data.csv and seq_all.csv

import csv
import pandas as pd
import numpy as np
from Game import Game


moment_data_path = '/Users/amaryans/Documents/school/spring22/cosc424/projects/final_project/nba_deep_learning/NBA-Shots/merged_shot_data.csv'
seq_data_path = '/Users/amaryans/Documents/school/spring22/cosc424/projects/try2/RNN_basketball/data/seq_all.csv'

# Long winded way of finding when the first datapoint of the shot
# is in the dataframe which has the player data in it

# moment data is the description of the live game data with player data as well as ball data
# Drop duplicates because for some reason some data points are duplicated which slows down searches
# seq_data is the data of the 3 point shots with the x,y,z of the ball data and the outcome
moment_data = pd.read_csv(moment_data_path)
moment_data = moment_data.drop_duplicates()
seq_data = pd.read_csv(seq_data_path)

# Merging the data here based on game clock x y and z, there could be duplicates of the shots in different games,
# but the odds of that are so low based on the accuracy of the positional data, again need to drop duplicates just in case
# Only doing z data here now in order to keep more shot data when doing all of x y and z it would drop a lot of shots
seq_data['game_clock'] = seq_data['game_clock'].astype(str)
seq_data['z'] = seq_data['z'].astype(str)
#moment_data['game_clock'] = moment_data['game_clock'].astype(float)
#moment_data['z'] = moment_data['z'].astype(float)

merged = moment_data.merge(seq_data, on=['game_clock', 'z'], how="inner")
merged = merged.drop_duplicates()
print(merged)
shot_start = merged.loc[merged['rankc'] == 1]
print(shot_start)

# This is how I have figured out how to get the moments before the shot, but right now it is dropping over half of
# the shots. When using get_loc it returns an array of "FALSE", could be because the game clock is that value at many points
conv_input = pd.DataFrame()
for val in shot_start['game_clock']:
    index = moment_data.index.get_loc(val)
    if isinstance(index, int):
        conv_input = conv_input.append(moment_data.iloc[index-60:index])
    else:
        # Fix for there being multiple game_clock values which are the same. Since we are merging based on game clock and z value
        # then only taking from rank = 1 then we can assume that this will give us all of the shots in merged data
        for i in range(len(index)):
            if index[i] == True:
                if moment_data['z'].iloc[i] == shot_start['z'].loc[shot_start['game_clock'] == val].values[0]:
                    conv_input = conv_input.append(moment_data.iloc[i-60:i])
            else:
                pass
        print("next")

print(conv_input)