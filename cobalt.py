#!/usr/bin/env python3

import abc
import matplotlib.pyplot as plt
import numpy as np

"""
## Planning: the Traveling Robot Problem

Visit a collection of points in the shortest path you can find. 
The catch? You have to "go home to recharge" every so often. 

We want fast approximations rather than a brute force perfect solution.
Your solution will be judged on:
* the length of path it produces
* fast runtime
* code quality and maintainability

### Details

* There are 5000 points distributed uniformly in [0, 1]
* The recharge station is located at (.5, .5)
* You cannot travel more than 3 units of distance before recharging
* You must start and end at the recharge station
* Skeleton code provided in Python. Python and C++ are acceptable
"""
#############################
home = np.array([0.5, 0.5]) # home is the recharging station

def visualize_pts(pts: list):
    # kathia # visualize #undos the vstack
    x = []
    y = []
    for i, val in enumerate (pts):
        x.append(val[0])
        y.append(val[1])
        
    plt.scatter (x, y)
    plt.show()
    pt = pts[1, :]
    last = pts[0, :] 
    d = np.linalg.norm(pt - last)

def check_order(pts, max_charge: float, order):
    """Check whether a given order of points is valid, and prints the total 
    length. You start and stop at the charging station.
    pts: np array of points to visit, prepended by the location of home
    order: array of pt indicies to visit, where 0 is home
    i.e. order = [0, 1, 0, 2, 0, 3, 0]"""
    pass

    print("Checking order")
        # assert statements checking for immediate errors 
    assert(pts.shape == (len(pts), 2)) # nothing weird
    assert(order[0] == 0) # start path at home
    assert(order[-1] == 0) # end path at home
    assert(set(order) == set(range(len(pts)))) # all pts visited

    print ("Assertions passed")

    # traverse path
    total_d = 0 # holds total distance travelled by robot
    charge = max_charge 
    last = pts[0, :] # home

    for idx in order:
        pt = pts[idx, :] #single point
        d = np.linalg.norm(pt - last) # distance

    # update totals
    total_d += d # add to distance traveled so far 
    charge -= d # subtract to the amount of charge we have 

    assert(charge > 0) # out of battery

    # did we recharge?
    if idx == 0:
        charge = max_charge

    # moving to next point
    last = pt

    def draw_path(pts, order):
        """Draw the path to the screen"""
        path = pts[order, :]
        plt.plot(path[:, 0], path[:, 1])
        plt.show()

    # We made it to end! path was valid
    print ("Valid path!")
    print (total_d)
    draw_path(pts, order)

#############################
# Your code goes here
# Read the "pts" array
# generate a valid order, starting and ending with 0, the recharging station
#############################

from scipy.spatial import distance

class Route(object):
    """
    Simple route data structure with a distance value and list of waypoints.

    Attributes:
        distance    float=0     distance of this Route.
        waypoints   list=[]     waypoints of this Route.
    """

    def __init__(self, distance: float=0, waypoints: list=[]):
        """
        Constructs a new Route instance with an initial distance value and list 
        of waypoints

        :param distance: distance of this Route.
        :param waypoints: waypoints of this Route.
        """
        self.distance = distance
        self.waypoints = waypoints

class Solution():
    """
    Base abstract Solution class.

    Attributes:
        pts         list        points of this Solution.
        total_dst   float=0     total distance of the waypoints of this solution
        waypoints   list=[0]    waypoints of this solution
    """

    def __init__(self, pts: list):
        """
        Constructs a new Solution instance with an initial list of points.

        :param pts: the points list to use for this solution.
        """

        # points store of this Solution
        self.pts = pts
        # total distance traveled by the robot for this Solution
        self.total_dst = 0
        # array of waypoints for this Solutiion
        self.waypoints = [0]
        
        # precalculate the distance of all points from home base
        # very minor performance improvement by O(N/c), where c is random
        # based on how points are selected at runtime.
        self.dsts_from_home = distance.cdist([home], pts)[0]

        # points that have been used from self.pts
        self.used_pts = [0]

        # Masks used to to access pts and dsts_from_home
        self.pts_mask = np.zeros(self.pts.shape, bool)
        self.dst_mask = np.zeros(self.dsts_from_home.shape, bool)

    @abc.abstractmethod
    def find_next_route(self, max_charge: float):
        """
        Finds the next valid route that starts and ends at home and returns the
        distance traveled and waypoints of that route.
        Extending classes MUST implement this abstract method and return a Route 
        instance.

        :param max_charge: the max charge allowed for the robot measured in 
        distance.
        :return: a Route instance.
        """
        pass

    def find_and_append_next_route(self, max_charge: float):
        """
        Convenience method for automatically appending the next route.
        
        :param max_charge: the max charge allowed for the robot measured in distance.
        """

        route: Route = self.find_next_route(max_charge)
        self.total_dst += route.distance
        self.waypoints += route.waypoints

    def generate_waypoints(self, max_charge: float, verbose: bool=False):
        """
        Generates the full route of subroutes for the robot to traverse all waypoints
        in 'self.pts'. This extendeding class may override this method, but it will 
        sacrifice the standard output logic of this default implementation.

        :param max_charge: the max charge allowed for the robot measured in distance.
        :param verbose: specify 'True' to output verbose messages to stdout. Default
        value is 'False'.
        :return: a list of waypoints indices pointing to elements in 'self.pts'.
        """
        while len(self.used_pts) < len(self.pts):
            self.find_and_append_next_route(max_charge)
        
        if verbose:
            print('Waypoints length:\t%s' % len(self.waypoints))
            print('Distance traveled:\t%.2f' % self.total_dst)
            number_of_routes = (len(self.waypoints) - len(self.pts))
            print('# of routes traveled:\t %s' % number_of_routes)
            print('Average dist/route:\t %.2f' % (self.total_dst / number_of_routes))
        return self.waypoints

class SolutionA(Solution):
    """ 
    - Most efficient implementation in terms of developmental complexity.
    - Computational complexity is exponential O(N * (N - c)), where c is the number
    of points already visited
    - Point traversal complexity is linear O(N + c), where c is number of times 
    returning home
    - Scalability is not ideal for magnitudes where N > 1,000,000 - a multiprocessing 
    approach would improve computational performance by a linear factor, 
    O((N * N - c) / t), where c is the number of points already visited, and t is
    the number of threads used for multiprocessing.
    """

    def find_next_loc_from(self, loc, max_charge: float, dst_traveled: float=0, waypoints=[]):
        """
        Finds the next location from a starting location and appends the new location,
        if it is traversable, to passed in existing distance traveled as well as waypoints.
        
        :param max_charge: the max charge allowed for the robot measured in distance.
        :param dst_traveled: the distance already traveled thus far.
        :param waypoints: the waypoints already traveled.
        :return: a 4-d tuple containing the new location traveled to, the location index in
        pts, the new total distance traveled thus far, the llist of waypoints traveled.
        """

        self.pts_mask[self.used_pts] = True
        self.dst_mask[self.used_pts] = True

        # available points
        self.unused_pts = np.ma.MaskedArray(self.pts, self.pts_mask)
        # available point distances to home
        self.unused_dhs = np.ma.MaskedArray(self.dsts_from_home, self.dst_mask) 

        # distances to current location
        self.dsts_from_loc = distance.cdist([loc], self.unused_pts)[0]
        # index reference to dsts_from_loc
        self.dsts_from_loc_mask = np.ma.MaskedArray(self.dsts_from_loc, self.dst_mask) 

        # index of closest point to loc
        idx = self.dsts_from_loc_mask.argmin() 
        loc_idx = idx
        
        # target location
        target_loc = self.unused_pts[idx].compressed()  
        # distance from current location to target location
        target_dst = self.dsts_from_loc[idx]
        # distance of target location from home
        target_home_dst = self.unused_dhs[idx]

        # travel to target if dist traveled + distance to target + 
        # distance of target from home is within range, then go home
        if (dst_traveled + target_dst + target_home_dst) == max_charge:
            self.used_pts.append(idx) # point is now used
            waypoints.append(idx) # add point to order list
            waypoints.append(0) # add home to the order list
            loc = home # go to home 
            loc_idx = 0
            dst_traveled += target_dst + target_home_dst

        # travel to target if dist traveled + distance to target + 
        # distance of target from home is within range
        elif (dst_traveled + target_dst + target_home_dst) < max_charge:
            self.used_pts.append(idx) # point is now used
            waypoints.append(idx) # add point to order list
            loc = target_loc
            loc_idx = idx
            dst_traveled += target_dst

        # travel home if dist traveled + distance to target + 
        # distance of target from home out of range
        elif (dst_traveled + target_dst + target_home_dst) > max_charge:
            waypoints.append(0) # add home to the order list
            loc = home # go home
            loc_idx = 0
            dst_traveled += self.dsts_from_loc[0]

        else:
            # This edge case will never actually be run
            pass

        # if all points have been visited, return home
        if len(self.used_pts) >= len(self.pts):
            waypoints.append(0) # add home to the order list
            loc = home # go home
            loc_idx = 0
            dst_traveled = 0 # reset distance

        return (loc, loc_idx, dst_traveled, waypoints)

    def find_next_route(self, max_charge: float):
        loc = home
        dst_traveled = 0
        waypoints = []
        (loc, loc_idx, dst_traveled, waypoints) = self.find_next_loc_from(loc, max_charge, dst_traveled, waypoints)
        while loc_idx != 0:
            (loc, loc_idx, dst_traveled, waypoints) = self.find_next_loc_from(loc, max_charge, dst_traveled, waypoints)
        return Route(dst_traveled, waypoints)

class SolutionB(Solution):
    """
    Template Solution B for POC example should another route 
    selection process be considered. See SolutionB_try.py which 
    should it's business logic be a good candidate for
    another edge case, it could eventually be implemented
    here.
    """

    def find_next_route(self, max_charge: float):
        # TODO - custom route logic goes here.
        return Route(0, [0])

def pfchk(func, *args, **kwargs):
    """ Runs a performance check on the specified routine """

    import time
    start = time.time()
    return_values = func(*args, **kwargs)
    end = time.time()
    print('Took %.2fs to execute' % (end - start))
    return return_values

def main():
    """
    Handles CLI passed arguments to allow for flexible testing with variable 
    parameter including and N parameter specfifying the scale size
    of the problem.
    """

    import argparse

    parser = argparse.ArgumentParser(description='Solves a robot point traversal problem.')
    parser.add_argument('-m', '--max-charge', type=float, default=3.0, help='max charge allowed for robot. default is 3.0')
    parser.add_argument('-n', '-N', type=int, default=5000, help='an integer to specify N points to be randomly generated. default is 5000')
    parser.add_argument('-v', '--verbose', action='store_true', help='output to stdout interesting metrics')

    args = parser.parse_args()

    # parse arguments
    N = args.n
    max_charge = args.max_charge
    verbose = args.verbose
    if N == None or N < 0:
        N = 5000

    if verbose:
        print('Generating %s points' % N)
    pts = np.vstack((home, np.random.rand(N, 2)))

    sol = SolutionA(pts)
    waypoints = pfchk(sol.generate_waypoints, max_charge=max_charge, verbose=verbose)
    check_order(pts, max_charge, waypoints)

if __name__ == "__main__":
    main()