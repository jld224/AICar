# Python program to HMM Part2 - template.
# Part2: figure out where the moving "hidden" car is.
# The car and your agent (car) live in a nxn periodic grid world.
# assume a shape of car is square, length is 1

import numpy as np
import pandas as pd
import sys
from scipy.stats import norm
import math


# Function: Get Belief
# ---------------------
# Updates beliefs based on recorded distance, transition prob, and your car's (agent's) pos.
# @input: gridSize, recording/observation, transition prob, car length = 1
# @return: your belief of the prob the "hidden" car is at each tile at each time step.
# Note: Your belief probabilities should sum to 1. (belief probabilities = posterior prob)

def getBeliefwMovingObj(N, observation, transitionP, carLength=1):
    std = carLength / 3.

    timeSteps = observation.shape[0]

    carTrackingFrames = np.zeros((timeSteps + 1, N, N))
    carTrackingFrames[0] = 1. / (N * N)

    for t in range(1, timeSteps + 1):
        agentX, agentY, eDist = observation.iloc[t - 1]

        # compute prior
        prior = np.zeros((N, N))
        for _, row in transitionP.iterrows():
            x, y, n, e, s, w = int(row.X), int(row.Y), row.N, row.E, row.S, row.W
            prior[(x-1)%N, y] += n * carTrackingFrames[t-1, x, y]
            prior[(x+1)%N, y] += s * carTrackingFrames[t-1, x, y]
            prior[x, (y-1)%N] += w * carTrackingFrames[t-1, x, y]
            prior[x, (y+1)%N] += e * carTrackingFrames[t-1, x, y]

        # compute emission probabilities
        emissionP = np.zeros((N, N))
        for x in range(N):
            for y in range(N):
                dx, dy = abs(agentX - x), abs(agentY - y)
                dx, dy = min(dx, N - dx), min(dy, N - dy)
                dist = math.sqrt(dx ** 2 + dy ** 2)
                emissionP[x, y] = norm.pdf(eDist, dist, std)

        # update beliefs
        posterior = emissionP * prior
        carTrackingFrames[t] = posterior / posterior.sum()

    return carTrackingFrames[1:]



# No need to change the main function.
def main():
    # example: 10 20 movingCarReading10.csv transitionProb10.csv
    gridSize, reportingTime, microphoneReadingFileName, transitionProbFileName = sys.argv[1:]
    gridSize, reportingTime = int(gridSize), int(reportingTime)

    transitionP = pd.read_csv(transitionProbFileName)
    readings = pd.read_csv(microphoneReadingFileName, nrows=reportingTime)
    readings_df = pd.DataFrame(readings, columns=['agentX', 'agentY', 'eDist'])

    print("Shape of transitionP:", transitionP.shape)

    probMapWithTime = getBeliefwMovingObj(gridSize, readings_df, transitionP)

    mostProbableCarPosWithTime = np.zeros([reportingTime, 2])
    for t in range(reportingTime):
        mostProbableCarPosWithTime[t] = np.unravel_index(np.argmax(probMapWithTime[t], axis=None), probMapWithTime[t].shape)

    df = pd.DataFrame(mostProbableCarPosWithTime, columns=['carX', 'carY'], dtype=np.int32)
    df.to_csv("most probable location with time" + str(gridSize) + "_tillTime" + str(reportingTime) + ".csv", index=False)
    print("Most probable location (row#, column#) at time", reportingTime, ":", tuple(df.iloc[-1]))


if __name__ == '__main__':
    main()
