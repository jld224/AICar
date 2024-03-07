# Hidden Markov Model (HMM) for Car Tracking

This Python template implements HMM Part1 and Part2 to track the hidden position of a car in a periodic grid world. In Part1, the stationary hidden car's position is estimated, while in Part2, the moving hidden car's position is estimated.

## Part1: Stationary Car Tracking

### Features
- **Grid World**: The car and the agent (your car) reside in an nxn periodic grid world.
- **Sensor Data**: Recorded distance observations and your car's position are used to update beliefs.
- **Belief Calculation**: Beliefs about the probability of the hidden car's position in each tile are calculated and normalized.

### Functionality
- `getBelief(data, gridSize, carLength)`: Updates beliefs based on distance observations and agent's position.
- `printGrid(grid)`: Prints the values stored on the grid.

### Usage
1. Run the program with command line arguments: `python Part1.py gridSize reportingTime microphoneReadingFileName`.
2. View the most probable location of the hidden car.

## Part2: Moving Car Tracking

### Features
- **Grid World**: Similar to Part1, the car and agent reside in an nxn periodic grid world.
- **Sensor Data & Transition Probabilities**: Recorded distance observations, transition probabilities, and agent's position are used for belief updates.
- **Belief Calculation**: Beliefs about the probability of the hidden car's position in each tile at each time step are calculated and normalized.

### Functionality
- `getBeliefwMovingObj(grid_size, sensor_data, prob_matrix, grid_unit=1)`: Updates beliefs based on recorded distance, transition probabilities, and agent's position.
- `getTopTwoBeliefs(carTrackingFrames)`: Returns the top two beliefs from the belief matrix.

### Usage
1. Run the program with command line arguments: `python Part2.py gridSize reportingTime microphoneReadingFileName transitionProbFileName`.
2. View the most probable location of the hidden car at each time step.

## Contributors
- [Jake Darida](https://github.com/jld224)
