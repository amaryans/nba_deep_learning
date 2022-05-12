import csv
import pandas as pd
import numpy as np
import os
def get_shooter(moment):
    ### Attempting to get the shooter from the moment data

    time = np.array(moment['game_clock'].values, dtype=float)
    ball_x = np.array(moment['x'].values, dtype=float)
    ball_y = np.array(moment['y'].values, dtype=float)
    print(time)
    print(time[0])
    print(time[-1])

    x = np.polyfit(time, ball_x, 2)
    y = np.polyfit(time, ball_y, 2)
    f_x = np.poly1d(x)
    f_y = np.poly1d(y)
    #print(time[-1], time[0])
    prev_time = np.linspace(time[0], time[0] + (time[0] - time[-1]), len(time))
    print(prev_time)
    print(f_x(prev_time))
    print(f_y(prev_time))

    print(moment['player3_x'].values)
    print(moment['player3_y'].values)

    
    pass

df = pd.read_csv("merged_data.csv")

df = df.drop_duplicates()
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

### For each moment need to get the player who shot the ball ###
# This will take some math, just a standard exterpolation of the shot data back to the x and y value of the player #

moments = df.id.unique()

for moment in moments:
    test = df.loc[df['id'] == moment]
    get_shooter(test)
    os.pause(100)

