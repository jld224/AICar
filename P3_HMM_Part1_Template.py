# Python template for HMM Part1
# figure out where the stationary "hidden" car is.
# The car and your agent (car) live in a nxn periodic grid world.
# assume a shape of car is square, length is 1

import numpy as np
import pandas as pd
import sys
from scipy.stats import norm
import math


# print the values stored on grid just in case you are interested in the numbers
def printGrid(grid):
    grid = grid[::-1]
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]-1):
            print(f"{grid[i][j]:.3f}", ", ", end='')
        print(f"{grid[i][grid.shape[1] - 1]:.3f}")


# Function: Get Belief
# Updates beliefs based on the distance observation and your car's (agent's) position.
# Returns your belief of the probability that the hidden car is in each tile. Your
# belief probabilities should sum to 1.
def getBelief(observation, gridSize, carLength):
    std = carLength * 2. / 3
    carPosMap = np.zeros((gridSize, gridSize))  # space holder
    rowC, colC = (0, 0)  # space holder for most probable location of the hidden car
    # Your code


    # carPosMap.sum should be 1
    return rowC, colC, carPosMap


# No need to change the main function.
def main():
    # Example command line arguments: 10 3 stationaryCarReading10.csv
    gridSize, reportingTime, microphoneReadingFileName,  = sys.argv[1:]
    gridSize = int(gridSize)
    reportingTime = int(reportingTime)
    carLength = 1
    print(gridSize, reportingTime, microphoneReadingFileName)

    data = pd.read_csv(microphoneReadingFileName, nrows=reportingTime)
    # print(data.head())  # take a peak of your data

    df = pd.DataFrame(data, columns=['agentX', 'agentY', 'eDist'])
    rowC, colC, carPosBelief = getBelief(df, gridSize, carLength)  # return numpy array of probabilities

    # printGrid(carPosBelief)
    print("Most probable location (row#, column#): (", str(rowC), ",", str(colC), ")")
    df = pd.DataFrame(carPosBelief, columns=np.array(list(range(gridSize))))
    df.to_csv("probMap" + str(gridSize)+"_atTime" + str(reportingTime) + ".csv", sep=',')


if __name__ == '__main__':
    main()


