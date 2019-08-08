import numpy as np
import os

np.set_printoptions(precision=3, linewidth=200, suppress=True)

ENABLE_DISPLAY = True# 'disp' in sys.argv
ENABLE_PLOT = False #'plot' in sys.argv

home_config = 'half_sitting'

#solver 
th_stop = 1e-9
maxiter = 200
reginit = .1


#
nv = 18


#task params
jumpHeight =  0.15
jumpLength = [0.3, 0.0, 0.]
timeStep = 1.0e-2
groundKnots = 45
flyingKnots = 20

mu = 0.5
contact_normal = np.array([0.0, 0.0, 1.0])
clearance = 0.15

#weights
weight_array_com = np.array([1., 0., 1.0])
weight_com = 1e4


weight_array_postural = np.array([0] * 3 + [0.] * 3 + [.01] * (nv - 6) + [0.1] * nv)
weight_postural = 1e-02
weight_joint_limits = 1e1
weight_control = 1e-04
weight_friction = 1e-1
weight_clearance = 1e1

weight_foot_pos_impact_xy = 1e4
weight_foot_pos_impact_z = 1e07
weight_array_postural_impact =  np.array([0.]*3 + [0.]*3 + [10.0] * (nv - 6) + [10.] * nv)
weight_postural_impact = 1e1
#impact pos
kp_contact = 0.
kd_contact = 1.0/timeStep


#terminal state
weight_array_postural_terminal_velocity = np.array([0] * 3 + [1.] * 3 + [0.0] * (nv - 6) + [1.] * nv)
weight_postural_terminal_velocity = 1e05

#controller 
dt  = 0.001


data_file = 'optim_traj.npz'
#TODO robot  urdf
#urdf = path + '/urdf/romeo.urdf'
#srdf = path + '/srdf/romeo_collision.srdf'

#visualization
cameraTF = [2., 2.68, 0.84, 0.2, 0.62, 0.72, 0.22]


SAVE_FIGURES = False