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

def getBeliefwMovingObj(grid_size, sensor_data, prob_matrix, grid_unit=1):
    deviation = grid_unit / 3.0
    num_steps = sensor_data.shape[0]
    belief_matrix = np.zeros((num_steps + 1, grid_size, grid_size))
    belief_matrix[0] = 1. / (grid_size ** 2)

    for step in range(1, num_steps + 1):
        for x in range(grid_size):
            for y in range(grid_size):
                # Summation of probability based on prior belief and transition model
                prob_sum = 0
                for prev_x in range(grid_size):
                    for prev_y in range(grid_size):
                        # Calculate grid distance
                        delta_x = min(abs(x - prev_x), grid_size - abs(x - prev_x))
                        delta_y = min(abs(y - prev_y), grid_size - abs(y - prev_y))
                        probabilities = prob_matrix.loc[(prob_matrix['X'] == prev_x) & (prob_matrix['Y'] == prev_y),
                                                        ['N', 'E', 'S', 'W']].values.flatten()
                        # Determine direction and update belief accordingly
                        if delta_x == abs(x - prev_x):
                            direction = 0 if (y - prev_y) > 0 else 2
                            direction = direction if delta_y != 0 else 0
                            prob_sum += belief_matrix[step - 1, prev_x, prev_y] * probabilities[direction]
                        else:
                            direction = 1 if (x - prev_x) > 0 else 3
                            prob_sum += belief_matrix[step - 1, prev_x, prev_y] * probabilities[direction]

                # Factoring in sensor measurements
                observer_x, observer_y, expected_dist = sensor_data.iloc[step - 1][['agentX', 'agentY', 'eDist']]
                sensor_delta_x = min(abs(x - observer_x), grid_size - abs(x - observer_x))
                sensor_delta_y = min(abs(y - observer_y), grid_size - abs(y - observer_y))
                actual_dist = np.sqrt(sensor_delta_x ** 2 + sensor_delta_y ** 2)

                # Emission probability computation
                sensor_prob = norm.pdf(expected_dist, actual_dist, deviation)

                # Finalizing belief update for current cell
                belief_matrix[step, x, y] = prob_sum * sensor_prob

        # Probability normalization for current time step
        current_total = np.sum(belief_matrix[step])
        if current_total > 0:
            belief_matrix[step] /= current_total

    return belief_matrix[1:]

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
