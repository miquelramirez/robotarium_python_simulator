import numpy as np


def create_si_to_uni_dynamics(linear_velocity_gain: float = 1, angular_velocity_limit: float = np.pi):
    """ Returns a function mapping from single-integrator to unicycle dynamics with angular velocity magnitude restrictions.

        linear_velocity_gain: Gain for unicycle linear velocity
        angular_velocity_limit: Limit for angular velocity (i.e., |w| < angular_velocity_limit)

        -> function
    """
    #Check user input types
    if not isinstance(linear_velocity_gain, float):
        msg = ("In the function create_si_to_uni_dynamics, the linear velocity gain (linear_velocity_gain) must be an "
               f"integer or float. Received type {type(linear_velocity_gain.__name__)}.")
        raise TypeError(msg)
    if not isinstance(angular_velocity_limit, float):
        msg = ("In the function create_si_to_uni_dynamics, the angular velocity limit (angular_velocity_limit) must be "
               f"an integer or float. Received type {type(angular_velocity_limit).__name__}.")
        raise TypeError(msg)

    #Check user input ranges/sizes
    if linear_velocity_gain <= 0:
        msg: str = (f"In the function create_si_to_uni_dynamics, the linear velocity gain (linear_velocity_gain) must be "
                    f"positive. Received {linear_velocity_gain}.")
        raise ValueError(msg)
    if angular_velocity_limit < 0:
        msg: str= (f"In the function create_si_to_uni_dynamics, the angular velocity limit (angular_velocity_limit) must "
                   f"not be negative. Received {angular_velocity_limit}.")
        raise ValueError(msg)
    

    def si_to_uni_dyn(dxi: np.ndarray, poses: np.ndarray):
        """A mapping from single-integrator to unicycle dynamics.

        dxi: 2xN numpy array with single-integrator control inputs
        poses: 2xN numpy array with single-integrator poses

        -> 2xN numpy array of unicycle control inputs
        """

        #Check user input types
        assert isinstance(dxi, np.ndarray), ("In the si_to_uni_dyn function created by the create_si_to_uni_dynamics "
                                             "function, the single integrator velocity inputs (dxi) must be a numpy "
                                             "array. Received type %r.") % type(dxi).__name__
        assert isinstance(poses, np.ndarray), ("In the si_to_uni_dyn function created by the create_si_to_uni_dynamics "
                                               "function, the current robot poses (poses) must be a numpy array. Received "
                                               "type %r.") % type(poses).__name__

        #Check user input ranges/sizes
        assert dxi.shape[0] == 2, ("In the si_to_uni_dyn function created by the create_si_to_uni_dynamics function, the "
                                   "dimension of the single integrator velocity inputs (dxi) must be 2 ([x_dot;y_dot]). "
                                   "Received dimension %r.") % dxi.shape[0]
        assert poses.shape[0] == 3, ("In the si_to_uni_dyn function created by the create_si_to_uni_dynamics function, "
                                     "the dimension of the current pose of each robot must be 3 ([x;y;theta]). Received "
                                     "dimension %r.") % poses.shape[0]
        assert dxi.shape[1] == poses.shape[1], ("In the si_to_uni_dyn function created by the create_si_to_uni_dynamics "
                                                "function, the number of single integrator velocity inputs must be equal "
                                                "to the number of current robot poses. Received a single integrator "
                                                "velocity input array of size %r x %r and current pose array of "
                                                "size %r x %r.") % (dxi.shape[0], dxi.shape[1], poses.shape[0], poses.shape[1])

        M, N = np.shape(dxi)

        a = np.cos(poses[2, :])
        b = np.sin(poses[2, :])

        dxu = np.zeros((2, N))
        dxu[0, :] = linear_velocity_gain*(a*dxi[0, :] + b*dxi[1, :])
        dxu[1, :] = angular_velocity_limit*np.arctan2(-b*dxi[0, :] + a*dxi[1, :], dxu[0, :])/(np.pi/2)

        return dxu

    return si_to_uni_dyn


def create_si_to_uni_dynamics_with_backwards_motion(linear_velocity_gain: float = 1,
                                                    angular_velocity_limit: float = np.pi):
    """ Returns a function mapping from single-integrator dynamics to unicycle dynamics. This implementation of 
    the mapping allows for robots to drive backwards if that direction of linear velocity requires less rotation.

        linear_velocity_gain: Gain for unicycle linear velocity
        angular_velocity_limit: Limit for angular velocity (i.e., |w| < angular_velocity_limit)

    """

    #Check user input types
    if not isinstance(linear_velocity_gain, float):
        msg = ("In the function create_si_to_uni_dynamics, the linear velocity gain (linear_velocity_gain) must be an "
               f"integer or float. Received type {type(linear_velocity_gain.__name__)}.")
        raise TypeError(msg)
    if not isinstance(angular_velocity_limit, float):
        msg = ("In the function create_si_to_uni_dynamics, the angular velocity limit (angular_velocity_limit) must be "
               f"an integer or float. Received type {type(angular_velocity_limit).__name__}.")
        raise TypeError(msg)
    #Check user input ranges/sizes
    assert linear_velocity_gain > 0, "In the function create_si_to_uni_dynamics, the linear velocity gain (linear_velocity_gain) must be positive. Received %r." % linear_velocity_gain
    assert angular_velocity_limit >= 0, "In the function create_si_to_uni_dynamics, the angular velocity limit (angular_velocity_limit) must not be negative. Received %r." % angular_velocity_limit
    

    def si_to_uni_dyn(dxi, poses):
        """A mapping from single-integrator to unicycle dynamics.

        dxi: 2xN numpy array with single-integrator control inputs
        poses: 2xN numpy array with single-integrator poses

        -> 2xN numpy array of unicycle control inputs
        """

        #Check user input types
        assert isinstance(dxi, np.ndarray), "In the si_to_uni_dyn function created by the create_si_to_uni_dynamics_with_backwards_motion function, the single integrator velocity inputs (dxi) must be a numpy array. Received type %r." % type(dxi).__name__
        assert isinstance(poses, np.ndarray), "In the si_to_uni_dyn function created by the create_si_to_uni_dynamics_with_backwards_motion function, the current robot poses (poses) must be a numpy array. Received type %r." % type(poses).__name__

        #Check user input ranges/sizes
        assert dxi.shape[0] == 2, "In the si_to_uni_dyn function created by the create_si_to_uni_dynamics_with_backwards_motion function, the dimension of the single integrator velocity inputs (dxi) must be 2 ([x_dot;y_dot]). Received dimension %r." % dxi.shape[0]
        assert poses.shape[0] == 3, "In the si_to_uni_dyn function created by the create_si_to_uni_dynamics_with_backwards_motion function, the dimension of the current pose of each robot must be 3 ([x;y;theta]). Received dimension %r." % poses.shape[0]
        assert dxi.shape[1] == poses.shape[1], "In the si_to_uni_dyn function created by the create_si_to_uni_dynamics_with_backwards_motion function, the number of single integrator velocity inputs must be equal to the number of current robot poses. Received a single integrator velocity input array of size %r x %r and current pose array of size %r x %r." % (dxi.shape[0], dxi.shape[1], poses.shape[0], poses.shape[1])

        M,N = np.shape(dxi)

        a = np.cos(poses[2, :])
        b = np.sin(poses[2, :])

        dxu = np.zeros((2, N))
        dxu[0, :] = linear_velocity_gain*(a*dxi[0, :] + b*dxi[1, :])
        dxu[1, :] = angular_velocity_limit*np.arctan2(-b*dxi[0, :] + a*dxi[1, :], dxu[0, :])/(np.pi/2)

        return dxu

    return si_to_uni_dyn

def create_si_to_uni_mapping(projection_distance=0.05, angular_velocity_limit = np.pi):
    """Creates two functions for mapping from single integrator dynamics to 
    unicycle dynamics and unicycle states to single integrator states. 
    
    This mapping is done by placing a virtual control "point" in front of 
    the unicycle.

    projection_distance: How far ahead to place the point
    angular_velocity_limit: The maximum angular velocity that can be provided

    -> (function, function)
    """

    #Check user input types
    assert isinstance(projection_distance, (int, float)), "In the function create_si_to_uni_mapping, the projection distance of the new control point (projection_distance) must be an integer or float. Received type %r." % type(projection_distance).__name__
    assert isinstance(angular_velocity_limit, (int, float)), "In the function create_si_to_uni_mapping, the maximum angular velocity command (angular_velocity_limit) must be an integer or float. Received type %r." % type(angular_velocity_limit).__name__
    
    #Check user input ranges/sizes
    assert projection_distance > 0, "In the function create_si_to_uni_mapping, the projection distance of the new control point (projection_distance) must be positive. Received %r." % projection_distance
    assert projection_distance >= 0, "In the function create_si_to_uni_mapping, the maximum angular velocity command (angular_velocity_limit) must be greater than or equal to zero. Received %r." % angular_velocity_limit

    def si_to_uni_dyn(dxi, poses):
        """Takes single-integrator velocities and transforms them to unicycle
        control inputs.

        dxi: 2xN numpy array of single-integrator control inputs
        poses: 3xN numpy array of unicycle poses

        -> 2xN numpy array of unicycle control inputs
        """

        #Check user input types
        assert isinstance(dxi, np.ndarray), "In the si_to_uni_dyn function created by the create_si_to_uni_mapping function, the single integrator velocity inputs (dxi) must be a numpy array. Received type %r." % type(dxi).__name__
        assert isinstance(poses, np.ndarray), "In the si_to_uni_dyn function created by the create_si_to_uni_mapping function, the current robot poses (poses) must be a numpy array. Received type %r." % type(poses).__name__

        #Check user input ranges/sizes
        assert dxi.shape[0] == 2, "In the si_to_uni_dyn function created by the create_si_to_uni_mapping function, the dimension of the single integrator velocity inputs (dxi) must be 2 ([x_dot;y_dot]). Received dimension %r." % dxi.shape[0]
        assert poses.shape[0] == 3, "In the si_to_uni_dyn function created by the create_si_to_uni_mapping function, the dimension of the current pose of each robot must be 3 ([x;y;theta]). Received dimension %r." % poses.shape[0]
        assert dxi.shape[1] == poses.shape[1], "In the si_to_uni_dyn function created by the create_si_to_uni_mapping function, the number of single integrator velocity inputs must be equal to the number of current robot poses. Received a single integrator velocity input array of size %r x %r and current pose array of size %r x %r." % (dxi.shape[0], dxi.shape[1], poses.shape[0], poses.shape[1])


        M,N = np.shape(dxi)

        cs = np.cos(poses[2, :])
        ss = np.sin(poses[2, :])

        dxu = np.zeros((2, N))
        dxu[0, :] = (cs*dxi[0, :] + ss*dxi[1, :])
        dxu[1, :] = (1/projection_distance)*(-ss*dxi[0, :] + cs*dxi[1, :])

        #Impose angular velocity cap.
        dxu[1,dxu[1,:]>angular_velocity_limit] = angular_velocity_limit
        dxu[1,dxu[1,:]<-angular_velocity_limit] = -angular_velocity_limit 

        return dxu

    def uni_to_si_states(poses):
        """Takes unicycle states and returns single-integrator states

        poses: 3xN numpy array of unicycle states

        -> 2xN numpy array of single-integrator states
        """

        _,N = np.shape(poses)

        si_states = np.zeros((2, N))
        si_states[0, :] = poses[0, :] + projection_distance*np.cos(poses[2, :])
        si_states[1, :] = poses[1, :] + projection_distance*np.sin(poses[2, :])

        return si_states

    return si_to_uni_dyn, uni_to_si_states

def create_uni_to_si_dynamics(projection_distance=0.05):
    """Creates two functions for mapping from unicycle dynamics to single 
    integrator dynamics and single integrator states to unicycle states. 
    
    This mapping is done by placing a virtual control "point" in front of 
    the unicycle.

    projection_distance: How far ahead to place the point

    -> function
    """

    #Check user input types
    assert isinstance(projection_distance, (int, float)), "In the function create_uni_to_si_dynamics, the projection distance of the new control point (projection_distance) must be an integer or float. Received type %r." % type(projection_distance).__name__
    
    #Check user input ranges/sizes
    assert projection_distance > 0, "In the function create_uni_to_si_dynamics, the projection distance of the new control point (projection_distance) must be positive. Received %r." % projection_distance
    

    def uni_to_si_dyn(dxu, poses):
        """A function for converting from unicycle to single-integrator dynamics.
        Utilizes a virtual point placed in front of the unicycle.

        dxu: 2xN numpy array of unicycle control inputs
        poses: 3xN numpy array of unicycle poses
        projection_distance: How far ahead of the unicycle model to place the point

        -> 2xN numpy array of single-integrator control inputs
        """

        #Check user input types
        assert isinstance(dxu, np.ndarray), "In the uni_to_si_dyn function created by the create_uni_to_si_dynamics function, the unicycle velocity inputs (dxu) must be a numpy array. Received type %r." % type(dxi).__name__
        assert isinstance(poses, np.ndarray), "In the uni_to_si_dyn function created by the create_uni_to_si_dynamics function, the current robot poses (poses) must be a numpy array. Received type %r." % type(poses).__name__

        #Check user input ranges/sizes
        assert dxu.shape[0] == 2, "In the uni_to_si_dyn function created by the create_uni_to_si_dynamics function, the dimension of the unicycle velocity inputs (dxu) must be 2 ([v;w]). Received dimension %r." % dxu.shape[0]
        assert poses.shape[0] == 3, "In the uni_to_si_dyn function created by the create_uni_to_si_dynamics function, the dimension of the current pose of each robot must be 3 ([x;y;theta]). Received dimension %r." % poses.shape[0]
        assert dxu.shape[1] == poses.shape[1], "In the uni_to_si_dyn function created by the create_uni_to_si_dynamics function, the number of unicycle velocity inputs must be equal to the number of current robot poses. Received a unicycle velocity input array of size %r x %r and current pose array of size %r x %r." % (dxu.shape[0], dxu.shape[1], poses.shape[0], poses.shape[1])

        
        M,N = np.shape(dxu)

        cs = np.cos(poses[2, :])
        ss = np.sin(poses[2, :])

        dxi = np.zeros((2, N))
        dxi[0, :] = (cs*dxu[0, :] - projection_distance*ss*dxu[1, :])
        dxi[1, :] = (ss*dxu[0, :] + projection_distance*cs*dxu[1, :])

        return dxi

    return uni_to_si_dyn