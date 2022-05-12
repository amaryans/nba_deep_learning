### Need to open moment_data.csv and seq_all.csv

import csv
import pandas as pd
import numpy as np




moment_data_path = '/Users/amaryans/Documents/school/spring22/cosc424/projects/NBA-Player-Movements/moment_data.csv'
seq_data_path = '/Users/amaryans/Documents/school/spring22/cosc424/projects/try2/RNN_basketball/data/seq_all.csv'

# Long winded way of finding when the first datapoint of the shot
# is in the dataframe which has the player data in it

# moment data is the description of the live game data with player data as well as ball data
# Drop duplicates because for some reason some data points are duplicated which slows down searches
# seq_data is the data of the 3 point shots with the x,y,z of the ball data and the outcome
moment_data = pd.read_csv(moment_data_path, index_col=1)
moment_data = moment_data.drop_duplicates()
seq_data = pd.read_csv(seq_data_path)

#print(moment_data)
#print(seq_data)

# Merging the data here based on game clock x y and z, there could be duplicates of the shots in different games,
# but the odds of that are so low based on the accuracy of the positional data, again need to drop duplicates just in case
# Only doing z data here now in order to keep more shot data when doing all of x y and z it would drop a lot of shots
merged = moment_data.merge(seq_data, on=['game_clock', 'z'], how="inner")
merged = merged.drop_duplicates()
print(merged)
shot_start = merged.loc[merged['rankc'] == 1]
print(shot_start)



#merged.to_csv("merged_data.csv", mode='a')