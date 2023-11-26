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
def getBelief(data, gridSize, carLength):
    std = carLength * 2. / 3
    carPosMap = np.ones((gridSize, gridSize))  # Initialize beliefs as 1 to prevent zero probabilities.

    for index, row in data.iterrows():
        agentX = row['agentX']
        agentY = row['agentY']
        eDist = row['eDist']

        for x in range(gridSize):
            for y in range(gridSize):
                dx, dy = abs(agentX - x), abs(agentY - y)
                dx, dy = min(dx, gridSize - dx), min(dy, gridSize - dy)
                d = math.sqrt(dx ** 2 + dy ** 2)
                pdf = norm.pdf(eDist, loc=d, scale=std)
                carPosMap[x, y] *= pdf

        # Normalize (prevent underflow and keep sum = 1)
        carPosMap /= carPosMap.sum()

    # Find car's most probable location
    rowC, colC = np.unravel_index(np.argmax(carPosMap), carPosMap.shape)

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
    print(data.head())  # take a peak of your data

    df = pd.DataFrame(data, columns=['agentX', 'agentY', 'eDist'])
    rowC, colC, carPosBelief = getBelief(df, gridSize, carLength)  # return numpy array of probabilities

    printGrid(carPosBelief)
    print("Most probable location (row#, column#): (", str(rowC), ",", str(colC), ")")
    df = pd.DataFrame(carPosBelief, columns=np.array(list(range(gridSize))))
    df.to_csv("probMap" + str(gridSize)+"_atTime" + str(reportingTime) + ".csv", sep=',')


if __name__ == '__main__':
    main()