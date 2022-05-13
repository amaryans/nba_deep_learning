import pandas as pd
from Event import Event
from Team import Team
from Constant import Constant
import json
import os

class Game:
    """A class for keeping info about the games"""
    def __init__(self, path_to_json, event_index):
        # self.events = None
        self.home_team = None
        self.guest_team = None
        self.event = None
        self.path_to_json = path_to_json
        self.event_index = event_index

    def read_json(self):
        data_frame = pd.read_json(self.path_to_json)
        last_default_index = len(data_frame) - 1
        self.event_index = min(self.event_index, last_default_index)
        #print(data_frame['gameid'][0])
        
        ### Has processed this using the json filetype by index. We can work with this and possibly extract the 3-pointer data
        ### The issue is we don't know whether or not the shot goes in or not...
        index = self.event_index
        #print(Constant.MESSAGE + str(last_default_index))
        i = 1
        print("Opening moment_data.csv for writing of first line")
        with open('moment_data.csv', 'w') as f:
            f.write('game_clock,shot_clock,x,y,z,player1_id,player1_x,player1_y,player2_id,player2_x,player2_y,player3_id,player3_x,player3_y,player4_id,player4_x,player4_y,player5_id,player5_x,player5_y,player6_id,player6_x,player6_y,player7_id,player7_x,player7_y,player8_id,player8_x,player8_y,player9_id,player9_x,player9_y,player10_id,player10_x,player10_y'+'\n')
                
        for event in data_frame['events']:
            #event = data_frame['events'][index]
            self.event = Event(event, data_frame['gameid'][0], i)
            i = i + 1
        
        moment_data_path = '/Users/amaryans/Documents/school/spring22/cosc424/projects/NBA-Player-Movements/moment_data.csv'
        seq_data_path = '/Users/amaryans/Documents/school/spring22/cosc424/projects/try2/RNN_basketball/data/seq_all.csv'

        print("Starting to merge the data and appending it to merged_data.csv for future use")
        # moment data is the description of the live game data with player data as well as ball data
        # Drop duplicates because for some reason some data points are duplicated which slows down searches
        # seq_data is the data of the 3 point shots with the x,y,z of the ball data and the outcome
        moment_data = pd.read_csv(moment_data_path, index_col = 0)
        moment_data = moment_data.drop_duplicates()
        seq_data = pd.read_csv(seq_data_path)

        # Merging the data here based on game clock x y and z, there could be duplicates of the shots in different games,
        # but the odds of that are so low based on the accuracy of the positional data, again need to drop duplicates just in case
        # Only doing z data here now in order to keep more shot data when doing all of x y and z it would drop a lot of shots
        merged = moment_data.merge(seq_data, on=['game_clock', 'z'], how="inner")
        merged = merged.drop_duplicates()
        merged.to_csv('merged_shot_data.csv', mode='a')

        shot_start = merged.loc[merged['rankc'] == 1]

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
        
        conv_input.to_csv('before_shot_data.csv', mode='a')

        os.system("rm moment_data.csv")
        print("Removed moment_data.csv after merging the data")
        
        
        self.home_team = Team(event['home']['teamid'])
        self.guest_team = Team(event['visitor']['teamid'])

    def start(self):
        self.event.show()
