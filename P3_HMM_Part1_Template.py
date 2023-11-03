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


def distance(agent_position, car_position, grid_size):
    """
    Function that calculates the periodic distance from agent to car.
    """
    diff = np.abs(np.array(agent_position) - np.array(car_position))
    dist = np.minimum(diff, grid_size - diff)
    return np.sqrt((dist ** 2).sum())



# Function: Get Belief
# Updates beliefs based on the distance observation and your car's (agent's) position.
# Returns your belief of the probability that the hidden car is in each tile. Your
# belief probabilities should sum to 1.
def getBelief(df, gridSize, carLength):
    # standard deviation for the Gaussian distribution
    std = carLength * 2. / 3
    # initialize the belief map - all grid cells have equal beliefs initially
    carPosMap = np.ones((gridSize, gridSize)) / (gridSize ** 2)

    # iterate over all observations in the data
    for _, row in df.iterrows():
        agent_position = (row["agentX"], row["agentY"])
        eDist = row["eDist"]

        # update belief of each grid cell
        for i in range(gridSize):
            for j in range(gridSize):
                car_position = (i, j)
                # calculate real distance
                true_distance = distance(agent_position, car_position, gridSize)
                # calculate emission probability
                emission_prob = norm.pdf(eDist, true_distance, std)
                # update belief
                carPosMap[i, j] *= emission_prob

        # normalize beliefs so they sum to 1
        tot_belief = carPosMap.sum()
        carPosMap /= tot_belief

    # return grid cell with maximum belief
    max_index = np.unravel_index(np.argmax(carPosMap, axis=None), carPosMap.shape)
    rowC, colC = max_index

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

    # printGrid(carPosBelief)
    print("Most probable location (row#, column#): (", str(rowC), ",", str(colC), ")")
    df = pd.DataFrame(carPosBelief, columns=np.array(list(range(gridSize))))
    df.to_csv("probMap" + str(gridSize)+"_atTime" + str(reportingTime) + ".csv", sep=',')


if __name__ == '__main__':
    main()


