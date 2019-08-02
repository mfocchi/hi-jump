import sys
import numpy as np
import pinocchio as se3
from pinocchio.robot_wrapper import RobotWrapper
import crocoddyl
from crocoddyl import a2m, m2a
import hi_jump.jump_functions as quadruped
from hi_jump.jump_functions import interpolate_trajectory
import hi_jump.utils as utils
import optim_params as conf
from tf.transformations import euler_from_quaternion

def loadHyQ(modelPath='/opt/openrobots/share/example-robot-data'):
    URDF_FILENAME = "hyq_last.urdf"
    URDF_SUBPATH = "/hyq_description/robots/" + URDF_FILENAME
    robot = RobotWrapper.BuildFromURDF(modelPath + URDF_SUBPATH, [modelPath], se3.JointModelFreeFlyer())
    # TODO define default position inside srdf
    robot.q0.flat[7:] = [-0.2, 0.75, -1.5, -0.2, -0.75, 1.5, -0.2, 0.75, -1.5, -0.2, -0.75, 1.5]
    robot.q0[2] = 0.57750958
    robot.model.referenceConfigurations[conf.home_config] = robot.q0
    return robot
    
# Loading the HyQ model
ROBOT =  loadHyQ()

if conf.ENABLE_DISPLAY:
    utils.setWhiteBackground(ROBOT)

rmodel = ROBOT.model
rdata = rmodel.createData()

# Defining the initial state of the robot
q0 = rmodel.referenceConfigurations[conf.home_config].copy()
v0 = se3.utils.zero(rmodel.nv)
x0 = crocoddyl.m2a(np.concatenate([q0, v0]))

# Setting up the 3d walking problem
lfFoot = 'lf_foot'
rfFoot = 'rf_foot'
lhFoot = 'lh_foot'
rhFoot = 'rh_foot'

#TODO NOT CLEAR
gait = quadruped.SimpleQuadrupedalGaitProblem(rmodel, lfFoot, rfFoot, lhFoot, rhFoot)

print('Building the action models')


# Creating a jumping problem
ddp = crocoddyl.SolverFDDP(
    gait.createJumpingProblem(x0, conf.jumpHeight, conf.jumpLength, conf.timeStep, conf.groundKnots, conf.flyingKnots))


# Added the callback functions
print('*** SOLVE  jumpin ***')

ddp.callback = [crocoddyl.CallbackDDPLogger(), crocoddyl.CallbackDDPVerbose()]
if conf.ENABLE_DISPLAY:
    ddp.callback += [crocoddyl.CallbackSolverDisplay(ROBOT, 4, 1, conf.cameraTF)]

# Solving the problem with the DDP solver
ddp.th_stop = conf.th_stop
ddp.solve(
    maxiter=conf.maxiter,
    regInit=conf.reginit,
    init_xs=[rmodel.defaultState] * len(ddp.models()),
    init_us=[
        m.differential.quasiStatic(d.differential, rmodel.defaultState) if isinstance(
            m, crocoddyl.IntegratedActionModelEuler) else np.zeros(0)
        for m, d in zip(ddp.problem.runningModels, ddp.problem.runningDatas)
    ])

# Defining the final state as initial one for the next phase
x0 = ddp.xs[-1]

# Display the entire motion
if conf.ENABLE_DISPLAY:
    ts = [m.timeStep if isinstance(m, crocoddyl.IntegratedActionModelEuler) else 0. for m in ddp.models()]
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
    if len(f[i])==0:
        f[i] = np.zeros(12)
f = np.array(f)
# alternative way
#f = [m.differential.f for m in ddp.problem.runningDatas if 'differential' in m.__dict__.keys()]
quat = xs[:,3:7]
euler = np.array([euler_from_quaternion(quat[i,:], axes='szyx') for i in range(len(quat))])

# rotate GRF to world frame
for i in range(f.shape[0]):
    m = ddp.problem.runningModels[i_no_impact[i]]
    d = ddp.problem.runningDatas[i_no_impact[i]]
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
