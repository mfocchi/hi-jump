import numpy as np
import os

np.set_printoptions(precision=3, linewidth=200, suppress=True)

ENABLE_DISPLAY = True# 'disp' in sys.argv
ENABLE_PLOT = 1 #'plot' in sys.argv


q0 = [0, 0, 0.5749, 0, 0, 0, 1, -0.2, 0.75, -1.5, -0.2, -0.75, 1.5, -0.2, 0.75, -1.5, -0.2, -0.75, 1.5]
urdfFileName = "hyq_last_torque_lim.urdf"
urdfSubPath = "/hyq_description/robots/"
 
#solver 
th_stop = 1e-9
maxiter = 30
reginit = .1

#
nv = 18

#task params
jumpHeight =  0.15

step = dict()
clearance_height = dict()
#test_name = 'FLAT'
test_name = 'PALLET'
#test_name = '2PALLET'

#orientReference = np.array([0.0, 0.0, 3.14])
orientReference = np.array([0.0, 0.0, 0.0])

if (test_name == 'FLAT'):
    step['FLAT']  = [0.2, 0.0, 0.0]

#1 pallet
if (test_name == 'PALLET'):
    pallet_size = [2.0, 2.0, 0.16]
    pallet_pos = [1.5, 0.0, 0.08]
    step['PALLET'] = [0.5, 0.0, pallet_size[2]]
#2 pallets
if (test_name == '2PALLET'):
    pallet_size = [2.0, 2.0, 0.1]
    pallet_pos = [1.5, 0.0, pallet_size[2]/2]
    pallet2_size = [0.1, pallet_size[1]/2,  pallet_size[2]/2]
    pallet2_pos = [0.55, 0.5, pallet_size[2]+pallet2_size[2]/2]


clearance_height['FLAT']  = 0.1
clearance_height['PALLET'] = 0.05
clearance_height['2PALLET'] = 0.05


params =[]

jumpLength = step[test_name]

lfFoot = 'lf_foot'
rfFoot = 'rf_foot'
lhFoot = 'lh_foot'
rhFoot = 'rh_foot'

timeStep = 1.0e-2
takeOffKnots = 45
rearingKnots = 0
flyingKnots = 20
landingKnots = 45

mu = 0.5
contact_normal = np.array([0.0, 0.0, 1.0])
clearance = clearance_height[test_name]

#weights
weight_array_com = np.array([1., 0., 1.0])
weight_com = 1e4

weight_array_orientation = np.array([0] * 3 + [1.] * 3 + [0.] * (nv - 6) + [0.0] * nv)
weight_orientation = 0e2  # 1e2

weight_array_postural = np.array([0] * 3 + [0.] * 3 + [.01] * (nv - 6) + [0.1] * nv)
#added famping on haas to avoid lateral motion of the legs
weight_array_postural[nv + 6] = 1
weight_array_postural[nv + 6 + 3] = 1
weight_array_postural[nv + 6 + 6] = 1
weight_array_postural[nv + 6 + 9] = 1
weight_postural = 1e-02
weight_joint_limits = 0*1e3

weight_torque_limits = 0* 1e1
weight_control = 1e-06
weight_friction = 1e-1
weight_clearance = 1e2

weight_foot_pos_impact_xy = 1e5
weight_foot_pos_impact_z = 1e07
weight_array_postural_impact =  np.array([0.]*3 + [0.]*3 + [10.0] * (nv - 6) + [10.] * nv)
weight_postural_impact = 1e1
#impact pos
kp_contact = 0.
kd_contact = 1.0/timeStep

#obstacle avoidance time parmaterized trapezoidal function
retractDuration = 5
extendDuration = 1
retractIndex = retractDuration
extendIndex = (flyingKnots*2+rearingKnots) - extendDuration


#terminal state
weight_array_postural_terminal_velocity = np.array([0] * 3 + [0.] * 3 + [0.0] * (nv - 6) + [1.] * nv)
weight_postural_terminal_velocity = 1e05

#controller 
dt  = 0.001


data_file = test_name+'.npz'
#TODO robot  urdf
#urdf = path + '/urdf/romeo.urdf'
#srdf = path + '/srdf/romeo_collision.srdf'

#visualization
cameraTF = None #[2., 2.68, 0.84, 0.2, 0.62, 0.72, 0.22]


height_map_resolution = np.array([0.01, 0.01]) 
height_map_xy0 = np.array([-0.5, 1.5])
height_map_size = 3.0
edge_position = pallet_pos[0]-0.5*pallet_size[0]-height_map_xy0[0]
# Box Blur kernel
kernel_size  = 3

SAVE_FIGURES = Falsearmature_inertia = 0

armature_inertia = 0
