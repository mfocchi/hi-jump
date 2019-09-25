import numpy as np
import os

np.set_printoptions(precision=3, linewidth=200, suppress=True)

ENABLE_DISPLAY = True# 'disp' in sys.argv
ENABLE_PLOT = 0 #'plot' in sys.argv


q0 = [0, 0, 0.230085, 0, 0, 0, 1,  -0.2, 0.75, -1.5, -0.2,0.75, -1.5,  -0.2, -0.75, 1.5,  -0.2, -0.75, 1.5]
urdfFileName = "solo12.urdf" #no torque limits   
urdfSubPath = "/solo_description/robots/"

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
test_name = 'FLAT'
#test_name = 'PALLET'
#test_name = '2PALLET'

#orientReference = np.array([0.0, 0.0, 3.14])
orientReference = np.array([0.0, 0.0, 0.0])

if (test_name == 'FLAT'):
    step['FLAT']  = [0.0, 0.0, 0.0]

#1 pallet
if (test_name == 'PALLET'):
    pallet_size = [2.0, 2.0, 0.05]
    pallet_pos = [1.25, 0.0, 0.025]
    step['PALLET'] = [0.5, 0.0, pallet_size[2]]
    edge_position = pallet_pos[0]-0.5*pallet_size[0]-height_map_xy0[0]
#2 pallets
if (test_name == '2PALLET'):
    pallet_size = [2.0, 2.0, 0.1]
    pallet_pos = [1.5, 0.0, pallet_size[2]/2]
    edge_position = pallet_pos[0]-0.5*pallet_size[0]-height_map_xy0[0]
    pallet2_size = [0.1, pallet_size[1]/2,  pallet_size[2]/2]
    pallet2_pos = [0.55, 0.5, pallet_size[2]+pallet2_size[2]/2]
    step['2PALLET'] = [1.0, 0.0, pallet_size[2]]


clearance_height['FLAT']  = 0.05
clearance_height['PALLET'] = 0.05
clearance_height['2PALLET'] = 0.05


params =[]

jumpLength = step[test_name]

lfFoot = 'FL_FOOT'
rfFoot = 'FR_FOOT'
lhFoot = 'HL_FOOT'
rhFoot = 'HR_FOOT'

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
weight_orientation =  0e2 #1e2

#added famping on haas to avoid lateral motion of the legs
weight_array_postural = np.array([0] * 3 + [0.] * 3 + [5.] * (nv - 6) + [1.] * nv)

weight_postural = 5e-01
weight_joint_limits = 4*1e3

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
data_file_q = test_name+'_q.txt'
data_file_qd = test_name+'_qd.txt'
data_file_tau = test_name+'_tau.txt'
#TODO robot  urdf
#urdf = path + '/urdf/romeo.urdf'
#srdf = path + '/srdf/romeo_collision.srdf'

#visualization
cameraTF = None #[2., 2.68, 0.84, 0.2, 0.62, 0.72, 0.22]


height_map_resolution = np.array([0.01, 0.01]) 
height_map_xy0 = np.array([-0.5, 1.5])
height_map_size = 3.0

# Box Blur kernel
kernel_size  = 3

SAVE_FIGURES = False

armature_inertia = 0.00000447
gear_ratio = 9 