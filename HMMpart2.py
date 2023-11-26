# Python program to HMM Part2 - template.
# Part2: figure out where the moving "hidden" car is.
# The car and your agent (car) live in a nxn periodic grid world.
# assume a shape of car is square, length is 1

import numpy as np
import pandas as pd
import sys
from scipy.stats import norm
import math

# Function: getTopTwoBeliefs
# ---------------------
# Utility function to return top 2 beliefs from getBelief function.
def getTopTwoBeliefs(carTrackingFrames):
    flatten_frames = carTrackingFrames.flatten()
    first_idx = np.argmax(flatten_frames)
    flatten_frames[first_idx] = -1  # Exclude the first max from the next search
    second_idx = np.argmax(flatten_frames)
    top_two_indices = [np.unravel_index(first_idx, carTrackingFrames.shape), np.unravel_index(second_idx, carTrackingFrames.shape)]
    return top_two_indices

# Function: Get Belief
# ---------------------
# Updates beliefs based on recorded distance, transition prob, and your car's (agent's) pos.
# @input: gridSize, recording/observation, transition prob, car length = 1
# @return: your belief of the prob the "hidden" car is at each tile at each time step.
# Note: Your belief probabilities should sum to 1. (belief probabilities = posterior prob)

def getBeliefwMovingObj(N, observation, transitionP, carLength=1):
    std = (2.0 * carLength) / 3.0  # Given from supplemental data

    timeSteps = observation.shape[0]
    carTrackingFrames = np.zeros((timeSteps + 1, N, N))
    carTrackingFrames[0] = 1. / (N * N)
    top_two_locations = np.zeros((timeSteps, 2, 2), dtype=int)  # To store top two locations

    for t in range(1, timeSteps + 1):
        agentX, agentY, eDist = observation.iloc[t - 1]

        # compute prior
        prior = np.zeros((N, N))
        for _, row in transitionP.iterrows():
            x, y, n, e, s, w = int(row.X), int(row.Y), row.N, row.E, row.S, row.W
            prior[(x-1)%N, y] += carTrackingFrames[t-1, (x-1)%N, y] * n
            prior[(x+1)%N, y] += carTrackingFrames[t-1, (x+1)%N, y] * s
            prior[x, (y-1)%N] += carTrackingFrames[t-1, x, (y-1)%N] * w
            prior[x, (y+1)%N] += carTrackingFrames[t-1, x, (y+1)%N] * e

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

        top_two_locations[t-1] = getTopTwoBeliefs(carTrackingFrames[t])

    # Output top two locations at time t
    print(f"Top two most likely locations of the hidden car at time {timeSteps}: {top_two_locations[-1]}")

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
