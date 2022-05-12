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
        print(data_frame['gameid'][0])
        ### Has processed this using the json filetype by index. We can work with this and possibly extract the 3-pointer data
        ### The issue is we don't know whether or not the shot goes in or not...
        index = self.event_index

        print(Constant.MESSAGE + str(last_default_index))
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
        moment_data = pd.read_csv(moment_data_path, index_col=1)
        seq_data = pd.read_csv(seq_data_path)
        merged = moment_data.merge(seq_data, on=['game_clock','x','y','z'], how="inner")
        merged = merged.drop_duplicates()
        merged.to_csv("merged_data.csv", mode='a')
        os.system("rm moment_data.csv")
        print("Removed moment_data.csv after merging the data")
        
        
        self.home_team = Team(event['home']['teamid'])
        self.guest_team = Team(event['visitor']['teamid'])

    def start(self):
        self.event.show()
