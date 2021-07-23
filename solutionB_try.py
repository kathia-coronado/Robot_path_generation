#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

"""
## Planning: The Traveling Robot Problem

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

max_charge = 3.0
#############################

# generate the points to visit uniformly in [0, 1]
# recharging station is index 0
N = 5000
pts = np.vstack((home, np.random.rand(N, 2))) 

def check_order(pts, order):
	"""Check whether a given order of points is valid, and prints the total 
	length. You start and stop at the charging station.
	pts: np array of points to visit, prepended by the location of home
	order: array of pt indicies to visit, where 0 is home
	i.e. order = [0, 1, 0, 2, 0, 3, 0]"""

	print ("Checking order")
    # assert statements checking for immediate errors 
	assert(pts.shape == (len(pts), 2)) # nothing weird
	assert(order[0] == 0) # start path at home
	assert(order[-1] == 0) # end path at home
	assert(set(order) == set(range(len(pts)))) # all pts visited

	print ("Assertions passed")

	# traverse path
	total_d = 0 # holds total distance travelled by vaccum
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

	# We made it to end! path was valid
	print ("Valid path!")
	print (total_d)
	draw_path(pts, order)

def draw_path(pts, order):
	"""Draw the path to the screen"""
	path = pts[order, :]

	plt.plot(path[:, 0], path[:, 1])
	plt.show()

#############################
# Your code goes here
# Read the "pts" array
# generate a valid order, starting and ending with 0, the recharging station
#############################
# known 

# most effienct in time to develope 
# computationally its average  ( could be beter )
# scalability not the best 
total_d = 0 # holds total distance travelled by vaccum
#charge = max_charge 
dh = distance.cdist([home], pts)[0]

used_pts = [0]
order = [0] 
current_loc = home
i = 0

# solution B

def find_return_points_between(pt, rpt, unused_pts, max_charge: float, dst_traveled: float=0):

    unused_mask = []

    for _pt in unused_pts:
        d = np.linalg.norm(pt - _pt)
        b = not (
            d > 0 and # ignore self point
            _pt[0] >= np.amin([pt[0], rpt[0]]) and
            _pt[0] <= np.amax([pt[0], rpt[0]]) and
            _pt[1] >= np.amin([pt[1], rpt[1]]) and
            _pt[1] <= np.amax([pt[1], rpt[1]]) and
           dst_traveled + d < max_charge
            )
        unused_mask.append(np.array([b, b]))
    unused_pts = np.ma.MaskedArray(unused_pts, unused_mask) 

    return unused_pts

def find_return_routes_from(pt, unused_pts, max_charge: float, dst_traveled: float=0):

    routes = [(np.linalg.norm(pt - home), [0])]
    pts_between = find_return_points_between(pt, home, unused_pts, max_charge, dst_traveled)
    #print('Returnable points %s: ' % pts_between)

    #return [(np.linalg.norm(pt - home), [0])]

    if pts_between.count() > 0:
        #print(np.array([pts_between.compressed()]))
        for n in range(len(pts_between)):
            tv = np.ma.getmask(pts_between[n])
            if np.array_equal(tv, np.array([True, True])):
                continue
            rtp = pts_between[n]
            d = np.linalg.norm(pt - rtp)
            #print(rtp)
            subroutes = list(map(lambda sr: (sr[0] + d, [n] + sr[1]),find_return_routes_from(rtp, unused_pts, max_charge, dst_traveled)))
            #print('Subroutes: %s' % subroutes)
            routes += subroutes

    return routes

def find_best_return_route_from(pt, unused_pts, max_charge: float, dst_traveled: float=0):

    routes = find_return_routes_from(pt, unused_pts, max_charge, dst_traveled)

    max_points = 0
    best_route = None
    for route in routes:
        if dst_traveled + route[0] <= max_charge and len(route[1]) > max_points:
            max_points = len(route[1])
            best_route = route
    
    return best_route
    
def find_points_from(pt, avail_pts, max_charge: float, dst_traveled: float=0, quad: int=None):

    unused_mask = []
     
    if quad == 1:
        for _pt in avail_pts:
            d = np.linalg.norm(pt - _pt)
            b = not (_pt[0] > pt[0] and _pt[1] > pt[1] and dst_traveled + d < max_charge)
            unused_mask.append(np.array([b, b]))
        avail_pts = np.ma.MaskedArray(avail_pts, unused_mask) # all points in forward quadrant 
    
    elif quad == 2: 
        for _pt in avail_pts:
            d = np.linalg.norm(pt - _pt)
            b = not (_pt[0] <= pt[0] and _pt[1] > pt[1] and dst_traveled + d < max_charge)
            unused_mask.append(np.array([b, b]))
        avail_pts = np.ma.MaskedArray(avail_pts, unused_mask) # all points in forward quadrant 
    
    elif quad == 3:
        for _pt in avail_pts:
            d = np.linalg.norm(pt - _pt)
            b = not (_pt[0] <= pt[0] and _pt[1] <= pt[1] and dst_traveled + d < max_charge)
            unused_mask.append(np.array([b, b]))
        avail_pts = np.ma.MaskedArray(avail_pts, unused_mask) # all points in forward quadrant 
    
    elif quad == 4: 
        for _pt in avail_pts:
            d = np.linalg.norm(pt - _pt)
            b = not (_pt[0] > pt[0] and _pt[1] <= pt[1] and dst_traveled + d < max_charge)
            unused_mask.append(np.array([b, b]))
        avail_pts = np.ma.MaskedArray(avail_pts, unused_mask) # all points in forward quadrant 

    return avail_pts

# Override pts to something we can work with
#pts = np.array([[.5, .5], [.6, .6], [.7, .7], [.8, .8], [.82, .9], [.6, .8], [0, 1], [0, 0], [1, 0]])
pts_mask = np.zeros(pts.shape, bool)
pts_mask[used_pts] = True
unused_pts = np.ma.MaskedArray(pts, pts_mask) # available points

def find_next_route(avail_pts, max_charge: float, dst_traveled: float=0):

    quad = None
    current_loc = home
    order = []

    while dst_traveled < max_charge:
    
        next_pts = find_points_from(current_loc, avail_pts, max_charge=3.0, dst_traveled=dst_traveled, quad=quad)
        unusable = [idx for idx, element in enumerate(np.ma.getmask(next_pts)) if np.array_equal(element, np.array([True, True]))]

        dhs = distance.cdist([current_loc], next_pts)[0]
        dhs_mask = np.zeros(dhs.shape, bool)
        dhs_mask[unusable] = True

        a_dpt = np.ma.MaskedArray(dhs, dhs_mask) # distances of all point to cp
        #print(a_dpt)
        idx = a_dpt.argmin() # index of closest point to cp
        pt = pts[idx] # test point (pt)
        dst = a_dpt[idx]
        #print(pt)
        #print(idx)

        (dst_to_home, waypoints) = find_best_return_route_from(pt, unused_pts, max_charge=max_charge, dst_traveled=dst_traveled)
        if dst_traveled + dst + dst_to_home == max_charge:
            # Update used and unused points
            order.append(idx)
            order.extend(waypoints)
            used_pts.append(idx)
            used_pts.extend(waypoints)
            current_loc = home
            dst_traveled += dst + dst_to_home
        
        elif dst_traveled + dst + dst_to_home < max_charge:

            if pt[0] > current_loc[0] and pt[1] > current_loc[1]:
                quad = 1
            elif pt[0] <= current_loc[0] and pt[1] > current_loc[1]:
                quad = 2
            elif pt[0] <= current_loc[0] and pt[1] <= current_loc[1]:
                quad = 3
            elif pt[0] > current_loc[0] and pt[1] <= current_loc[1]:
                quad = 4
            else:
                print('error this should never run')
                exit()

            # Update used and unused points
            order.append(idx)
            used_pts.append(idx)
            current_loc = pt
            dst_traveled += dst
            pts_mask[used_pts] = True
            avail_pts = np.ma.MaskedArray(pts, pts_mask) # available points

        else:
            (dst_to_home, waypoints) = find_best_return_route_from(current_loc, unused_pts, max_charge=max_charge, dst_traveled=dst_traveled)
            dst_traveled += dst_to_home
            order.extend(waypoints)
            break

    # TODO - Return the route
    return (dst_traveled, order)

while len(set(used_pts)) < len(pts):
    route = find_next_route(unused_pts, max_charge=3.0)
    order += route[1]
    print('Route traveled: (dist: %s, points: %s)' % route)

check_order(pts, order)

"""
p_temp = []
path_back = []

dp = distance.cdist([p], unsued_pts)[0] # distances to current point (cp)
a_dpt = np.ma.MaskedArray(dp, dhs_mask) # distances of all point to cp
idx = a_dpt.argmin() # index of closest point to cp
pt = unsued_pts[idx].compressed() # test point (pt) 
print(pt)

dpt = dp[idx] # distance of pt to cp
dpt_h = unused_dhs [idx] # distance of pt to home

# decide in what quadrant dir the pt is from home
# pt: test point 
# p: last point 
if pt[0] >= home[0] and pt[1] > home[1]:
    quad = 1 
elif  pt[0] < home[0] and pt[1] > home[1]:
    quad = 2 
elif pt[0] < home[0] and pt[1] <= home[1]:
    quad = 3 
elif pt[0] >= home[0] and pt[1] <= home[1]:
    quad = 4

f_pts = find_points_from(pt, unsued_pts, quad) # get mask for quadrant 
b_pts = find_return_points_from(pt, unsued_pts, quad) # get mask for quadrant 

f_dp = distance.cdist([pt], f_pts)[0] # distances of pts in the fwd dir 
b_dp = distance.cdist([pt], b_pts)[0] # distances of pts in the bwd dir 
f_idx = f_dp.argmin() # index of closes point in fd dir 
b_idx = b_dp.argmin() # index of closes point in bd dir 

f_dph = unused_dhs[f_idx]
b_dph = unused_dhs[b_idx]

f_pt = f_pts [f_idx]
b_pt = b_pts[b_idx]
"""

