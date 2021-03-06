import sys
import numpy as np
import pinocchio as se3
import crocoddyl
from crocoddyl import a2m, m2a
import hi_jump.jump_functions as quadruped
from hi_jump.jump_functions import interpolate_trajectory
import hi_jump.utils as utils
#import optim_params_solo as conf
import optim_params_hyq as conf
from tf.transformations import euler_from_quaternion
import time
import copy
from hi_jump.height_map_filter import createPalletMap,createFlatMap,  create2PalletMap, smoothHeightMap, plotHeightMap, computeDerivative
from crocoddyl.cost import HeightMap
import matplotlib.pyplot as plt

import rospy as ros
#from geometry_msgs.msg import Point
from ros_impedance_controller.srv import get_map
from ros_impedance_controller.srv import get_mapRequest
from ros_impedance_controller.srv import get_mapResponse

    
class CallbackJump:
    def __init__(self, level=0):
        self.level = level

    def __call__(self, solver):
        cost_total = {}
        N = {}
        for i, (m, d) in enumerate(zip(solver.problem.runningModels, solver.problem.runningDatas)):
            if "differential" in d.__dict__.keys():
                costs_data = d.differential.costs.costs
                costs_model = m.differential.costs.costs
            else:
                costs_data = d.costs.costs
                costs_model = m.costs.costs
                
            for key in costs_data.keys():
                if key not in cost_total:
                    cost_total[key] = [0.0, 0.0]
                    N[key] = 0
                cost_total[key][0] += costs_data[key].cost
                cost_total[key][1] += costs_model[key].weight * costs_data[key].cost
                N[key] += 1
        
        print "\n   %18s \t %10s \t %10s"%('TASK NAME', 'AVG COST', 'WEIGHTED COST')
        for key in np.sort(cost_total.keys()):
            print "   %18s \t %10.2f \t %10.2f"%(key, np.sqrt(2*cost_total[key][0]/N[key]), 
                                                 cost_total[key][1])
                   
# Loading the HyQ model
ROBOT =  utils.loadRobot(conf.urdfFileName, conf.urdfSubPath, conf.q0)

if conf.ENABLE_DISPLAY:
    utils.setWhiteBackground(ROBOT)
    if conf.test_name == 'PALLET' or conf.test_name == '2PALLETS' :
        ROBOT.viewer.gui.addBox("world/pallet", conf.pallet_size[0], conf.pallet_size[1], conf.pallet_size[2], (1.0,0.2,.2,.5))
        ROBOT.viewer.gui.setColor("world/pallet", (1.,0.,0.,1.))
        ROBOT.viewer.gui.applyConfiguration("world/pallet", (conf.pallet_pos[0], conf.pallet_pos[1], conf.pallet_pos[2], 0,0,0,1))
        ROBOT.viewer.gui.setLightingMode("world/pallet", "ON")
    if conf.test_name == '2PALLETS' :
        ROBOT.viewer.gui.addBox("world/pallettop", conf.pallet2_size[0], conf.pallet2_size[1], conf.pallet2_size[2], (1.0,0.2,.2,.5))
        ROBOT.viewer.gui.setColor("world/pallettop", (1.,0.,0.,1.))
        ROBOT.viewer.gui.applyConfiguration("world/pallettop", (conf.pallet2_pos[0], conf.pallet2_pos[1], conf.pallet2_pos[2], 0,0,0,1))

#    ROBOT.viewer.gui.addLight("world/michilight", 0, 5.0, (1.,1.,1.,1.))

rmodel = ROBOT.model
rdata = rmodel.createData()

# Defining the initial state of the robot
q0 = rmodel.referenceConfigurations['half_sitting'].copy()
v0 = se3.utils.zero(rmodel.nv)
x0 = crocoddyl.m2a(np.concatenate([q0, v0]))

# Setting up the 3d walking problem

# get custom height map
#1 pallet  
if conf.test_name == 'PALLET' :
    height_map = createPalletMap(conf.pallet_size[2], conf.edge_position, 
                                 conf.height_map_resolution[0], conf.height_map_size)
#2 pallet                             
if conf.test_name == '2PALLET' :
    height_map = create2PalletMap(conf.pallet_size[2], conf.edge_position, conf.pallet2_size[0], conf.pallet2_size[2], conf.height_map_resolution[0], conf.height_map_size)   

#print height_map
if conf.test_name == 'FLAT' :
    height_map = createFlatMap(0.0, conf.height_map_resolution[0], conf.height_map_size)


height_map_blur = smoothHeightMap(conf.kernel_size, height_map)
height_map_der_x = computeDerivative(height_map_blur, conf.height_map_resolution[0],'X')
height_map_der_y = computeDerivative(height_map_blur, conf.height_map_resolution[1],'Y')
#plotHeightMap(height_map_blur)    
heightMap = HeightMap(height_map_blur, conf.height_map_xy0, conf.height_map_resolution, height_map_der_x, height_map_der_y)

#for x in np.arange(0.0, 0.5, 0.01):
#    print "Height for x=%.2f: %.2f"%(x, heightMap.getHeight(np.array([x,0.0])))
    
#get real height map
#get_map_service = ros.ServiceProxy("/hyq/ros_impedance_controller/get_map", get_map)
## prepare request
#req_msg = get_mapRequest()
#req_msg.target.x = 0.5
#req_msg.target.y = 0.0
#req_msg.width = 3.0
#req_msg.length = 3.0 
#req_msg.resolution_x = 0.1 
#req_msg.resolution_y = 0.1 


#send request and get response
#res = get_mapResponse()
#res  =  get_map_service(req_msg)   
#
##unvectorize message
#col_length= (int)(len(res.height_array)/ res.row_length)
#heightMap = np.zeros((col_length, res.row_length))          
#for i in range(col_length):
#    for j in range(res.row_length):
#        heightMap[i, j] = res.height_array[i*res.row_length + j]
#
#plotHeightMap(heightMap)
#plt.show()

#plotHeightMap(height_map_blur)


#TODO NOT CLEAR
gait = quadruped.SimpleQuadrupedalGaitProblem(conf, rmodel, heightMap)
callbacks = [crocoddyl.CallbackDDPLogger(), crocoddyl.CallbackDDPVerbose(), CallbackJump()]
if conf.ENABLE_DISPLAY:
    callbacks += [crocoddyl.CallbackSolverDisplayParallel(ROBOT, -1, 1, conf.cameraTF)]
    
print('WARMING UP')

# Creating a jumping problem
conf_warm_start = conf
weight_clearance = conf_warm_start.weight_clearance
conf_warm_start.weight_clearance = 0.0
ddp = crocoddyl.SolverFDDP(gait.createJumpingProblem(x0, conf_warm_start))
# Added the callback functions
print('*** SOLVE  jumpin ***')
# Solving the problem with the DDP solver
ddp.th_stop = conf_warm_start.th_stop
ddp.callback = callbacks
ddp.solve(
    maxiter=3,
    regInit=conf_warm_start.reginit,
    init_xs=[rmodel.defaultState] * len(ddp.models()),
    init_us=[
        m.differential.quasiStatic(d.differential, rmodel.defaultState) if isinstance(
            m, crocoddyl.IntegratedActionModelEuler) else np.zeros(0)
        for m, d in zip(ddp.problem.runningModels, ddp.problem.runningDatas)
    ])
    
print "First optimization finished"
conf.weight_com = 0.0
conf.weight_clearance = weight_clearance
xs, us = ddp.xs, ddp.us
ddp = crocoddyl.SolverFDDP(gait.createJumpingProblem(x0, conf))
ddp.callback = callbacks
ddp.th_stop = conf.th_stop
time.sleep(2.0)
ddp.solve(maxiter=conf.maxiter, regInit=conf.reginit, init_xs=xs, init_us=us)

# Defining the final state as initial one for the next phase
x0 = ddp.xs[-1]

# Display the entire motion
if conf.ENABLE_DISPLAY:
    time.sleep(2.0)
    ts = [10*m.timeStep if isinstance(m, crocoddyl.IntegratedActionModelEuler) else 0. for m in ddp.models()]
    utils.displayPhaseMotion(ROBOT, ddp.xs, ts)

# Plotting the entire motion
if conf.ENABLE_PLOT:
    quadruped.plotSolution(rmodel, ddp.xs, ddp.us)

#extract q, v, tau
# find time steps corresponding to impacts (which have zero duration)
i_no_impact = [i for (i,u) in enumerate(ddp.us) if u.shape[0]>0]
xs = np.array([ddp.xs[i] for i in i_no_impact])
us = np.array([ddp.us[i] for i in i_no_impact])

q  = xs[:,7:rmodel.nq]
qd = xs[:,rmodel.nq+6:]
tau = us
x_base  = xs[:,:7]
v_base  = xs[:,rmodel.nq:rmodel.nq+6]
f = [ddp.problem.runningDatas[i].differential.f for i in i_no_impact]
for i in range(len(f)):
    if len(f[i])!=12:
        f[i] = np.concatenate((f[i], np.zeros(12-len(f[i]))))
f = np.array(f)
# alternative way
#f = [m.differential.f for m in ddp.problem.runningDatas if 'differential' in m.__dict__.keys()]
quat = xs[:,3:7]
euler = np.array([euler_from_quaternion(quat[i,:], axes='szyx') for i in range(len(quat))])

# rotate GRF to world frame
v_com = np.zeros((len(i_no_impact),3))
for i in range(f.shape[0]):
    m = ddp.problem.runningModels[i_no_impact[i]]
    d = ddp.problem.runningDatas[i_no_impact[i]]
    v_com[i,:] = d.differential.pinocchio.vcom[0].A1
#    if(len(d.differential.f)==0):        
        #print "Time step %d, CoM acc:"%(i), (v_com[i,:]-v_com[i-1,:])/conf.timeStep        
    for j, frame_name in enumerate(m.differential.contact.contacts.keys()):
        # get rotation matrix
        frame_index = m.differential.contact.contacts[frame_name].frame
        R = m2a(d.differential.pinocchio.oMf[frame_index].rotation)
        f[i,j*3:(j+1)*3] = R.dot(f[i,j*3:(j+1)*3]) 


#interpolate
q_cont = interpolate_trajectory(q, conf.timeStep, conf.dt)
qd_cont = interpolate_trajectory(qd, conf.timeStep, conf.dt)
tau_cont = interpolate_trajectory(tau, conf.timeStep, conf.dt)
x_base_cont = interpolate_trajectory(x_base, conf.timeStep, conf.dt) #to FIX
v_base_cont = interpolate_trajectory(v_base, conf.timeStep, conf.dt)
f_cont = interpolate_trajectory(f, conf.timeStep, conf.dt)
euler_cont = interpolate_trajectory(euler, conf.timeStep, conf.dt)


# generate gravity compensation torques
tau_gravity = ddp.problem.runningModels[0].differential.quasiStatic(ddp.problem.runningDatas[0].differential, rmodel.defaultState)
            
np.savez(conf.data_file, 
         q_des = q_cont, qd_des = qd_cont, tau_des = tau_cont,
         x_base_des=x_base_cont, v_base_des=v_base_cont, f_des=f_cont, euler_des=euler_cont,
         tau_gravity=tau_gravity)


#majid traj
q_nohaas = q_cont[:,(1,2, 4,5,7,8,10,11)]
qd_nohaas = qd_cont[:,(1,2, 4,5,7,8,10,11)]
tau_nohaas = tau_cont[:,(1,2, 4,5,7,8,10,11)]
time  = np.linspace(0.0, conf.dt*q_cont.shape[0], q_cont.shape[0],  endpoint=False)


np.savetxt(conf.data_file_q, np.hstack((time.reshape((q_nohaas.shape[0],1)), q_nohaas)), fmt="%f")
np.savetxt(conf.data_file_qd,   np.hstack((time.reshape((q_nohaas.shape[0],1)), qd_nohaas)), fmt="%f")         
np.savetxt(conf.data_file_tau,  np.hstack((time.reshape((q_nohaas.shape[0],1)), tau_nohaas)) , fmt="%f")         

