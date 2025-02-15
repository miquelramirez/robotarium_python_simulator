# Code Listing 1 from the Robotarium guide

import numpy as np
import time

import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

def main() -> None:

    N: int = 1                  # number of robots
    iterations: int = 450       # run the simulation/experiment for 450 steps
    line_width: float = 5.0     # width of the line

    init_poses: np.array = np.array([-1.3, 0.0, 0.0]).reshape(3, 1)
    p_vec: np.array = np.empty((2,0)) # sequence of positions the robot visits

    sim = robotarium.Robotarium(number_of_robots=N,
                                show_figure=True,
                                initial_conditions=init_poses,
                                sim_in_real_time=True)
    # draw reference line
    sim.axes.plot([-1.6, -1.6], [0, 0], linewidth=line_width, color='k')

    for t in range(iterations):
        # Get the poses of the robots
        x_t = sim.get_poses()
        # Define the robot speed
        dxu_t = np.array([0.15, 0.0]).reshape(2, 1)
        # Set the velocities via corresponding commands
        sim.set_velocities(np.arange(N), dxu_t)

        # Plot robot true trajectory
        p_vec = np.append(p_vec, x_t[:2], axis=1)
        if t == iterations-1:
            sim.axes.scatter(p_vec[0,:], p_vec[1,:], s=1, linewidth=line_width, color='r', linestyle='dashed')
        # Move sim one step fwd
        sim.step()

    time.sleep(5)

    # Print debugging information
    sim.call_at_scripts_end()

if __name__ == '__main__':
    main()