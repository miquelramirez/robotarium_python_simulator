# From original example `leader_follower_static/leader_follower.py`
import numpy as np
from argparse import ArgumentParser, Namespace
from dataclasses import *

import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.graph import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *


@dataclass
class Parameters(object):
    # Number of robots
    N: int = 4
    # Time steps for integrator
    dt: float = 0.033
    # Waypoint tolerance
    wpt_tol: float = 0.03
    # Linear speed limit
    max_lin_speed: float = 0.15
    # Gains of formation control
    formation_control_gain: float = 10.0
    desired_distance: float = 0.3


def process_cmd_line_args() -> Namespace:
    """
    Processes command line arguments
    :return:
    """
    parser = ArgumentParser(description="Leader Follower demonstration, n=4")
    parser.add_argument("-T", "--num-iterations",
                        type=int,
                        default=5000,
                        help="Number of iterations")
    return parser.parse_args()


def main(opt: Namespace) -> None:
    """
    Main function
    :param opt:
    :return:
    """
    params: Parameters = Parameters()

    # waypoints for the leader to follow
    # (-1, 0.8), (-1, -0.8), (1, -0.8), (1.0, 0.8)
    waypoints = np.array([[-1, -1, 1, 1],[0.8, -0.8, -0.8, 0.8]])

    # Create the desired Laplacian
    followers = -completeGL(params.N - 1)
    L: np.ndarray = np.zeros(shape=(params.N, params.N))
    L[1:params.N, 1:params.N] = followers
    L[1, 1] = L[1, 1] + 1
    L[1, 0] = -1

    rows, cols = np.where(L == 1)

    # Initialize velocity vector
    dxi = np.zeros(shape=(2, params.N))
    # Initialize leader controller state
    # we have as many states as waypoints
    q: int = 0
    # Initialize plant states
    x0 = np.array([[0, 0.5, 0.3, -0.1], [0.5, 0.5, 0.2, 0], [0, 0, 0, 0]])

    # Setup the simulator
    sim = robotarium.Robotarium(number_of_robots=params.N,
                                show_figure=True,
                                initial_conditions=x0,
                                sim_in_real_time=True)

    # Get refs to helper functions to map single-integrator to unicycle states and
    # handle collision avoidance
    _, uni_to_si_states = create_si_to_uni_mapping()
    si_to_uni_dyn = create_si_to_uni_dynamics()

    # Barrier certificates
    si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()
    # Position controller
    leader_control = create_si_position_controller(velocity_magnitude_limit=0.1)

    # Run simulation
    for t in range(opt.num_iterations):

        # Get state update
        x_t = sim.get_poses()
        #x_t = uni_to_si_states(q_t)

        # Control algorithms
        ## Followers control
        for i in range(1, params.N):
            # Zero agent velociiteis and get topological neighbour of i-th agent
            dxi[:, [i]] = np.zeros(shape=(2, 1))
            neighbors = topological_neighbors(L, i)

            for j in neighbors:
                l_ij: np.array = x_t[:2, [j]] - x_t[:2, [i]]
                dist_ij: float = np.linalg.norm(l_ij)
                error: float = dist_ij**2 - params.desired_distance**2
                dxi[:, [i]] += params.formation_control_gain * error * l_ij
        ## Leader control
        waypoint = waypoints[:, q].reshape((2, 1))
        dxi[:, [0]] = leader_control(x_t[:2, [0]], waypoint)
        ## Leader control state update
        if np.linalg.norm(dxi[:, [0]]) < params.wpt_tol:
            q = (q+1)%4

        # Enforce velocity constraints
        norms = np.linalg.norm(dxi, 2, 0)
        idx_to_normalize = (norms > params.max_lin_speed)
        dxi[:, idx_to_normalize] *= params.max_lin_speed/norms[idx_to_normalize]

        # Apply barriers
        dxi = si_barrier_cert(dxi, x_t[:2, :])
        # Map control to unicycle control space
        dxu = si_to_uni_dyn(dxi, x_t)

        # Set velocities
        sim.set_velocities(np.arange(params.N), dxu)

        # One step formard in time
        sim.step()

    sim.call_at_scripts_end()


if __name__ == '__main__':
    main(process_cmd_line_args())