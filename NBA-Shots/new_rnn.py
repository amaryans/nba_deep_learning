import csv
import pandas as pd
import numpy as np
import os
from matplotlib import image
from matplotlib import pyplot as plt
import math

### Need to make these into classes really, but got lazy here and copy pasted a bunch...
def get_shooter(df):

    ### Attempting to get the shooter from the moment data    
    #df["x"] = df['x'].str.replace(' ', '').astype(float)
    x = df['x'].values[:]
    shooter = 0
    try:
        x = [float(val) for val in df['x'].values[:]]
    except:
        print("error")
        print(x)

    y = df['y'].values[:]
    y = [float(val) for val in df['y'].values[:]]
    b_start = [x[0], y[0]]
    b_end = [x[-1], y[-1]]

    p1x = [float(val) for val in df['player1_x'].values[:]]
    p1y = [float(val) for val in df['player1_y'].values[:]]
    p1_start = [p1x[0], p1y[0]]
    p1_end = [p1x[-1], p1y[-1]]
    
    tmp = math.sqrt((p1_end[0] - b_end[0])**2 + (p1_end[1] - b_end[1])**2)
    shooter = 1

    p2x = [float(val) for val in df['player2_x'].values[:]]
    p2y = [float(val) for val in df['player2_y'].values[:]]
    p2_start = [p2x[0], p2y[0]]
    p2_end = [p2x[-1], p2y[-1]]

    if math.sqrt((p2_end[0] - b_end[0])**2 + (p2_end[1] - b_end[1])**2) < tmp:
        tmp = math.sqrt((p2_end[0] - b_end[0])**2 + (p2_end[1] - b_end[1])**2)
        shooter = 2

    p3x = [float(val) for val in df['player3_x'].values[:]]
    p3y = [float(val) for val in df['player3_y'].values[:]]
    p3_start = [p3x[0], p3y[0]]
    p3_end = [p3x[-1], p3y[-1]]

    if math.sqrt((p3_end[0] - b_end[0])**2 + (p3_end[1] - b_end[1])**2) < tmp:
        tmp = math.sqrt((p3_end[0] - b_end[0])**2 + (p3_end[1] - b_end[1])**2)
        shooter = 3

    p4x = [float(val) for val in df['player4_x'].values[:]]
    p4y = [float(val) for val in df['player4_y'].values[:]]
    p4_start = [p4x[0], p4y[0]]
    p4_end = [p4x[-1], p4y[-1]]

    if math.sqrt((p4_end[0] - b_end[0])**2 + (p4_end[1] - b_end[1])**2) < tmp:
        tmp = math.sqrt((p4_end[0] - b_end[0])**2 + (p4_end[1] - b_end[1])**2)
        shooter = 4

    p5x = [float(val) for val in df['player5_x'].values[:]]
    p5y = [float(val) for val in df['player5_y'].values[:]]
    p5_start = [p5x[0], p5y[0]]
    p5_end = [p5x[-1], p5y[-1]]

    if math.sqrt((p5_end[0] - b_end[0])**2 + (p5_end[1] - b_end[1])**2) < tmp:
        tmp = math.sqrt((p5_end[0] - b_end[0])**2 + (p5_end[1] - b_end[1])**2)
        shooter = 5

    p6x = [float(val) for val in df['player6_x'].values[:]]
    p6y = [float(val) for val in df['player6_y'].values[:]]
    p6_start = [p6x[0], p6y[0]]
    p6_end = [p6x[-1], p6y[-1]]

    if math.sqrt((p6_end[0] - b_end[0])**2 + (p6_end[1] - b_end[1])**2) < tmp:
        tmp = math.sqrt((p6_end[0] - b_end[0])**2 + (p6_end[1] - b_end[1])**2)
        shooter = 6

    p7x = [float(val) for val in df['player7_x'].values[:]]
    p7y = [float(val) for val in df['player7_y'].values[:]]
    p7_start = [p7x[0], p7y[0]]
    p7_end = [p7x[-1], p7y[-1]]

    if math.sqrt((p7_end[0] - b_end[0])**2 + (p7_end[1] - b_end[1])**2) < tmp:
        tmp = math.sqrt((p7_end[0] - b_end[0])**2 + (p7_end[1] - b_end[1])**2)
        shooter = 7

    p8x = [float(val) for val in df['player8_x'].values[:]]
    p8y = [float(val) for val in df['player8_y'].values[:]]
    p8_start = [p8x[0], p8y[0]]
    p8_end = [p8x[-1], p8y[-1]]

    if math.sqrt((p8_end[0] - b_end[0])**2 + (p8_end[1] - b_end[1])**2) < tmp:
        tmp = math.sqrt((p8_end[0] - b_end[0])**2 + (p8_end[1] - b_end[1])**2)
        shooter = 8

    p9x = [float(val) for val in df['player9_x'].values[:]]
    p9y = [float(val) for val in df['player9_y'].values[:]]
    p9_start = [p9x[0], p9y[0]]
    p9_end = [p9x[-1], p9y[-1]]

    if math.sqrt((p9_end[0] - b_end[0])**2 + (p9_end[1] - b_end[1])**2) < tmp:
        tmp = math.sqrt((p9_end[0] - b_end[0])**2 + (p9_end[1] - b_end[1])**2)
        shooter = 9

    p10x = [float(val) for val in df['player10_x'].values[:]]
    p10y = [float(val) for val in df['player10_y'].values[:]]
    p10_start = [p10x[0], p10y[0]]
    p10_end = [p10x[-1], p10y[-1]]

    if math.sqrt((p10_end[0] - b_end[0])**2 + (p10_end[1] - b_end[1])**2) < tmp:
        tmp = math.sqrt((p10_end[0] - b_end[0])**2 + (p10_end[1] - b_end[1])**2)
        shooter = 10

    return shooter

### creates an image for the moment
def make_image(df, shooter, save_path):
    x = df['x'].values[:]
    try:
        x = [float(val) for val in df['x'].values[:]]
    except:
        print("error")
        print(x)

    y = df['y'].values[:]
    y = [float(val) for val in df['y'].values[:]]
    b_start = [x[0], y[0]]
    b_end = [x[-1], y[-1]]

    p1x = [float(val) for val in df['player1_x'].values[:]]
    p1y = [float(val) for val in df['player1_y'].values[:]]
    p1_start = [p1x[0], p1y[0]]
    p1_end = [p1x[-1], p1y[-1]] 

    p2x = [float(val) for val in df['player2_x'].values[:]]
    p2y = [float(val) for val in df['player2_y'].values[:]]
    p2_start = [p2x[0], p2y[0]]
    p2_end = [p2x[-1], p2y[-1]]

    p3x = [float(val) for val in df['player3_x'].values[:]]
    p3y = [float(val) for val in df['player3_y'].values[:]]
    p3_start = [p3x[0], p3y[0]]
    p3_end = [p3x[-1], p3y[-1]]

    p4x = [float(val) for val in df['player4_x'].values[:]]
    p4y = [float(val) for val in df['player4_y'].values[:]]
    p4_start = [p4x[0], p4y[0]]
    p4_end = [p4x[-1], p4y[-1]]

    p5x = [float(val) for val in df['player5_x'].values[:]]
    p5y = [float(val) for val in df['player5_y'].values[:]]
    p5_start = [p5x[0], p5y[0]]
    p5_end = [p5x[-1], p5y[-1]]

    p6x = [float(val) for val in df['player6_x'].values[:]]
    p6y = [float(val) for val in df['player6_y'].values[:]]
    p6_start = [p6x[0], p6y[0]]
    p6_end = [p6x[-1], p6y[-1]]

    p7x = [float(val) for val in df['player7_x'].values[:]]
    p7y = [float(val) for val in df['player7_y'].values[:]]
    p7_start = [p7x[0], p7y[0]]
    p7_end = [p7x[-1], p7y[-1]]

    p8x = [float(val) for val in df['player8_x'].values[:]]
    p8y = [float(val) for val in df['player8_y'].values[:]]
    p8_start = [p8x[0], p8y[0]]
    p8_end = [p8x[-1], p8y[-1]]

    p9x = [float(val) for val in df['player9_x'].values[:]]
    p9y = [float(val) for val in df['player9_y'].values[:]]
    p9_start = [p9x[0], p9y[0]]
    p9_end = [p9x[-1], p9y[-1]]

    p10x = [float(val) for val in df['player10_x'].values[:]]
    p10y = [float(val) for val in df['player10_y'].values[:]]
    p10_start = [p10x[0], p10y[0]]
    p10_end = [p10x[-1], p10y[-1]]

    home_color = "#a30508"
    away_color = "#d1a930"
    shooter_color = "#db66e3"
    fig, ax = plt.subplots()
    ax.imshow(data)
    plt.plot(x, y, label = "ball", color = "#508a3e")
    if shooter < 6:
        plt.plot(p1x, p1y, label = "player 1", color = home_color)
        plt.plot(p2x, p2y, label = "player 2", color = home_color)
        plt.plot(p3x, p3y, label = "player 3", color = home_color)
        plt.plot(p4x, p4y, label = "player 4", color = home_color)
        plt.plot(p5x, p5y, label = "player 5", color = home_color)
        plt.plot(p6x, p6y, label = "player 6", color = away_color)
        plt.plot(p7x, p7y, label = "player 7", color = away_color)
        plt.plot(p8x, p8y, label = "player 8", color = away_color)
        plt.plot(p9x, p9y, label = "player 9", color = away_color)
        plt.plot(p10x, p10y, label = "player 10", color = away_color)
    else:
        plt.plot(p1x, p1y, label = "player 1", color = away_color)
        plt.plot(p2x, p2y, label = "player 2", color = away_color)
        plt.plot(p3x, p3y, label = "player 3", color = away_color)
        plt.plot(p4x, p4y, label = "player 4", color = away_color)
        plt.plot(p5x, p5y, label = "player 5", color = away_color)
        plt.plot(p6x, p6y, label = "player 6", color = home_color)
        plt.plot(p7x, p7y, label = "player 7", color = home_color)
        plt.plot(p8x, p8y, label = "player 8", color = home_color)
        plt.plot(p9x, p9y, label = "player 9", color = home_color)
        plt.plot(p10x, p10y, label = "player 10", color = home_color)

    if shooter == 1:
        plt.plot(p1x, p1y, label = "player 1", color = shooter_color)
    elif shooter == 2:
        plt.plot(p2x, p2y, label = "player 2", color = shooter_color)
    elif shooter == 3:
        plt.plot(p3x, p3y, label = "player 3", color = shooter_color)
    elif shooter == 4:
        plt.plot(p4x, p4y, label = "player 4", color = shooter_color)
    elif shooter == 5:
        plt.plot(p5x, p5y, label = "player 5", color = shooter_color)
    elif shooter == 6:
        plt.plot(p6x, p6y, label = "player 6", color = shooter_color)
    elif shooter == 7:
        plt.plot(p7x, p7y, label = "player 7", color = shooter_color)
    elif shooter == 8:
        plt.plot(p8x, p8y, label = "player 8", color = shooter_color)
    elif shooter == 9:
        plt.plot(p9x, p9y, label = "player 9", color = shooter_color)
    elif shooter == 10:
        plt.plot(p10x, p10y, label = "player 10", color = shooter_color)

    
    #plt.scatter(p10_start[0], p10_start[1], marker='o', color = 'white')
    #plt.scatter(p10_end[0], p10_end[1], marker='x', color = 'white')

    ax = plt.gca()
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    plt.axis("off")
    #ax.margins(94, 50)
    ax.set(xlim=(-2, 96), ylim=(-2, 52))
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()




df = pd.read_csv("before_shot_data.csv")
ball_position = [df['x'].values[1:60], df['y'].values[1:60]]
player_1 = [df['player1_x'].values[1:60], df['player1_y'].values[1:60]]
data = image.imread('court.png')

for i in range(int(len(df) / 60)):
    if i < 14520:
        continue
    else:
        path = "/Users/amaryans/Documents/school/spring22/cosc424/projects/final_project/nba_deep_learning/NBA-Shots/moment_images/moment_" + str(i) + ".png"
        shooter = get_shooter(df[i*60 + 1: 60*(i+1)])
        make_image(df[i*60 + 1: 60*(i+1)], shooter= shooter, save_path=path)


