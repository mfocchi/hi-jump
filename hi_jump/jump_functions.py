import numpy as np
import pinocchio
#import optim_params as conf
from crocoddyl import (ActivationModelWeightedQuad, ActivationModelInequality, ActuationModelFreeFloating, ActionModelImpact, CallbackDDPLogger,
                       CallbackDDPVerbose, CallbackSolverDisplay, ContactModel3D, ContactModelMultiple, CostModelCoM,
                       CostModelControl, CostModelFrameTranslation, CostModelForceLinearCone, CostModelState,
                       CostModelSum, DifferentialActionModelFloatingInContact, IntegratedActionModelEuler,
                       ImpulseModelMultiple, ImpulseModel3D, ShootingProblem, SolverDDP, SolverFDDP, StatePinocchio,
                       a2m, displayTrajectory, loadHyQ, m2a, CostDataForceLinearCone, ActionDataImpact,
                       CostModelNumDiff, CostDataNumDiff)


def plotSolution(rmodel, xs, us):
    import matplotlib.pyplot as plt
    # Getting the state and control trajectories
    nx, nq, nu = xs[0].shape[0], rmodel.nq, us[0].shape[0]
    X = [[0.]] * nx
    U = [[0.]] * nu
    for i in range(nx):
        X[i] = [x[i] for x in xs]
    for i in range(nu):
        U[i] = []
        for u in us:
            if u.shape != (0, ):
                U[i].append(u[i])

    # Plotting the joint positions, velocities and torques
    plt.figure(1)
    legJointNames = ['HAA', 'HFE', 'KFE']
    # LF foot
    plt.subplot(4, 3, 1)
    plt.title('joint position [rad]')
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(7, 10))]
    plt.ylabel('LF')
    plt.legend()
    plt.subplot(4, 3, 2)
    plt.title('joint velocity [rad/s]')
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(nq + 6, nq + 9))]
    plt.ylabel('LF')
    plt.legend()
    plt.subplot(4, 3, 3)
    plt.title('joint torque [Nm]')
    [plt.plot(U[k], label=legJointNames[i]) for i, k in enumerate(range(0, 3))]
    plt.ylabel('LF')
    plt.legend()

    # LH foot
    plt.subplot(4, 3, 4)
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(10, 13))]
    plt.ylabel('LH')
    plt.legend()
    plt.subplot(4, 3, 5)
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(nq + 9, nq + 12))]
    plt.ylabel('LH')
    plt.legend()
    plt.subplot(4, 3, 6)
    [plt.plot(U[k], label=legJointNames[i]) for i, k in enumerate(range(3, 6))]
    plt.ylabel('LH')
    plt.legend()

    # RF foot
    plt.subplot(4, 3, 7)
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(13, 16))]
    plt.ylabel('RF')
    plt.legend()
    plt.subplot(4, 3, 8)
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(nq + 12, nq + 15))]
    plt.ylabel('RF')
    plt.legend()
    plt.subplot(4, 3, 9)
    [plt.plot(U[k], label=legJointNames[i]) for i, k in enumerate(range(6, 9))]
    plt.ylabel('RF')
    plt.legend()

    # RH foot
    plt.subplot(4, 3, 10)
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(16, 19))]
    plt.ylabel('RH')
    plt.xlabel('knots')
    plt.legend()
    plt.subplot(4, 3, 11)
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(nq + 15, nq + 18))]
    plt.ylabel('RH')
    plt.xlabel('knots')
    plt.legend()
    plt.subplot(4, 3, 12)
    [plt.plot(U[k], label=legJointNames[i]) for i, k in enumerate(range(9, 12))]
    plt.ylabel('RH')
    plt.legend()
    plt.xlabel('knots')
    plt.show()

    plt.figure(2)
    rdata = rmodel.createData()
    Cx = []
    Cy = []
    for x in xs:
        q = a2m(x[:rmodel.nq])
        c = pinocchio.centerOfMass(rmodel, rdata, q)
        Cx.append(np.asscalar(c[0]))
        Cy.append(np.asscalar(c[1]))
    plt.plot(Cx, Cy)
    plt.title('CoM position')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.grid(True)
    plt.show()
    
    plt.figure()
    plt.plot(Cx)
    plt.title('CoM position x')
    plt.grid(True)
    plt.show()

    plt.figure()
    rdata = rmodel.createData()
    Cz = []
    for x in xs:
        q = a2m(x[:rmodel.nq])
        c = pinocchio.centerOfMass(rmodel, rdata, q)
        Cz.append(np.asscalar(c[2]))

    plt.plot(Cz)
    plt.title('CoM position Z')

    plt.grid(True)
    plt.show()



class TaskSE3:
    def __init__(self, oXf, frameId):
        self.oXf = oXf
        self.frameId = frameId


class SimpleQuadrupedalGaitProblem:
    def __init__(self, conf, rmodel, lfFoot, rfFoot, lhFoot, rhFoot):
        self.rmodel = rmodel
        self.rdata = rmodel.createData()
        self.state = StatePinocchio(self.rmodel)
        # Getting the frame id for all the legs
        self.lfFootId = self.rmodel.getFrameId(lfFoot)
        self.rfFootId = self.rmodel.getFrameId(rfFoot)
        self.lhFootId = self.rmodel.getFrameId(lhFoot)
        self.rhFootId = self.rmodel.getFrameId(rhFoot)
        # Defining default state
        q0 = self.rmodel.referenceConfigurations[conf.home_config]
        self.rmodel.defaultState = np.concatenate([m2a(q0), np.zeros(self.rmodel.nv)])
        self.firstStep = True  
        
        A = np.zeros((4,3))
        Tx = np.cross(np.array([1.0, 0.0, 0.0]), conf.contact_normal)
        Tx /= np.linalg.norm(Tx)
        Ty = np.cross(conf.contact_normal, Tx)
        A[0,:] =  Tx - conf.mu*conf.contact_normal
        A[1,:] = -Tx - conf.mu*conf.contact_normal
        A[2,:] =  Ty - conf.mu*conf.contact_normal
        A[3,:] = -Ty - conf.mu*conf.contact_normal
        print "A\n", A
        self.A_friction_cones = A

    def createJumpingProblem(self, x0, conf):
        q0 = a2m(x0[:self.rmodel.nq])
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        rfFootPos0 = self.rdata.oMf[self.rfFootId].translation
        rhFootPos0 = self.rdata.oMf[self.rhFootId].translation
        lfFootPos0 = self.rdata.oMf[self.lfFootId].translation
        lhFootPos0 = self.rdata.oMf[self.lhFootId].translation
        df = conf.jumpLength[2] - rfFootPos0[2]
        rfFootPos0[2] = 0.
        rhFootPos0[2] = 0.
        lfFootPos0[2] = 0.
        lhFootPos0[2] = 0.
        comRef = m2a(rfFootPos0 + rhFootPos0 + lfFootPos0 + lhFootPos0) / 4
        comRef[2] = m2a(pinocchio.centerOfMass(self.rmodel, self.rdata, q0))[2]

        four_foot_support = [self.lfFootId, self.lhFootId, self.rfFootId, self.rhFootId]
        loco3dModel = []
        
        
        #TAKE OFF PHASE        
        takeOff = [
            self.createSwingFootModel(
                conf,
                four_foot_support,
            ) for k in range(conf.takeOffKnots)
        ]


        #create foot task for obstacle avoidance       
        footPosRelativeZ = np.zeros(2*conf.flyingKnots+conf.rearingKnots)    
        clearanceTasks = []
        for i in range(len(footPosRelativeZ)):
            if (i < conf.retractIndex):                
                footPosRelativeZ[i] =  conf.clearance * i / (conf.retractDuration)
            elif (i >= conf.retractIndex) and (i < conf.extendIndex):
                footPosRelativeZ[i] = conf.clearance 
            elif (i >= conf.extendIndex) and (i < len(footPosRelativeZ)):
                footPosRelativeZ[i] = conf.clearance - conf.clearance*(i - conf.extendIndex)/(conf.extendDuration) 
            else:
                print "out of bounds"
            
            #add the tasks in the list
            clearanceDic_lf = dict()     
            clearanceDic_lf['id'] = self.lfFootId
            clearanceDic_lf['pos'] = m2a(lfFootPos0) + np.array([0.0,0.0, footPosRelativeZ[i] ])
        
            clearanceDic_rf = dict()     
            clearanceDic_rf['id'] = self.rfFootId
            clearanceDic_rf['pos'] = m2a(rfFootPos0) + np.array([0.0,0.0, footPosRelativeZ[i] ])
            
            clearanceDic_lh = dict()     
            clearanceDic_lh['id'] = self.lhFootId
            clearanceDic_lh['pos'] = m2a(lhFootPos0) + np.array([0.0,0.0, footPosRelativeZ[i] ])

            clearanceDic_rh = dict()     
            clearanceDic_rh['id'] = self.rhFootId
            clearanceDic_rh['pos'] = m2a(rhFootPos0) + np.array([0.0,0.0, footPosRelativeZ[i] ]) 
            
            if i<conf.rearingKnots:
                clearanceTasks += [[clearanceDic_lf, clearanceDic_rf]]
            else:
                clearanceTasks += [[clearanceDic_lf, clearanceDic_rf, clearanceDic_lh, clearanceDic_rh]]
                
        #REARING PHASE 
        two_foot_support = [ self.lhFootId,  self.rhFootId]
        rearing = []
        for k in range(conf.rearingKnots):
            rearing += [self.createSwingFootModel(conf, two_foot_support, clearanceTask = clearanceTasks[k])]

        flyingUpPhase = []      
        for k in range(conf.flyingKnots):
            flyingUpPhase += [self.createSwingFootModel(conf, [],
                                      np.array([0.5*conf.jumpLength[0], 0.5*conf.jumpLength[1],  conf.jumpHeight]) * (k + 1) / conf.flyingKnots + comRef, 
                                      clearanceTask = clearanceTasks[conf.rearingKnots+k])]

        flyingDownPhase = []
        for k in range(conf.flyingKnots):                             
            flyingDownPhase += [self.createSwingFootModel(conf, [], clearanceTask = clearanceTasks[conf.rearingKnots+conf.flyingKnots+k])]


        f0 = np.matrix(conf.jumpLength).T
        footTask = [
            TaskSE3(pinocchio.SE3(np.eye(3), lfFootPos0 + f0), self.lfFootId),
            TaskSE3(pinocchio.SE3(np.eye(3), rfFootPos0 + f0), self.rfFootId),
            TaskSE3(pinocchio.SE3(np.eye(3), lhFootPos0 + f0), self.lhFootId),
            TaskSE3(pinocchio.SE3(np.eye(3), rhFootPos0 + f0), self.rhFootId)
        ]
        landingPhase = [
            self.createImpactModel(conf, four_foot_support, footTask)
        ]
        f0[2] = df
        landed = [
            self.createSwingFootModel(conf, four_foot_support, comTask=comRef + m2a(f0))
            for k in range(conf.landingKnots)
        ]
        
        #terminal state        
        terminalVelocity = CostModelState(self.rmodel, self.state, self.rmodel.defaultState, landed[-1].differential.actuation.nu,
                                  ActivationModelWeightedQuad(conf.weight_array_postural_terminal_velocity**2))
        landed[-1].differential.costs.addCost("terminalVelocity", terminalVelocity, conf.weight_postural_terminal_velocity)
        
        
        
        loco3dModel += takeOff
        loco3dModel += rearing
        loco3dModel += flyingUpPhase
        loco3dModel += flyingDownPhase
        loco3dModel += landingPhase
        loco3dModel += landed
        
        problem = ShootingProblem(x0, loco3dModel, loco3dModel[-1])
        
#        # QUICK FIX: Set contactData on CostDataForceLinearCone 
        for d in problem.runningDatas + [problem.terminalData]:
            # skip the impact phase
            if isinstance(d, ActionDataImpact): continue
                
            contacts = d.differential.contact.contacts
            costs = d.differential.costs.costs
            
            # skip the flying phase
            if len(contacts)==0: continue
                
            for foot_id in four_foot_support:
                contact_key = [key for key in contacts.keys() if str(foot_id) in key]
                if len(contact_key)==1:
                    friction_key = [key for key in costs.keys() if str(foot_id) in key]
                    assert(len(friction_key)==1)
                    cost_data = costs[friction_key[0]]
                    if isinstance(cost_data, CostDataForceLinearCone):
                        cost_data.contact = contacts[contact_key[0]]
                    elif isinstance(cost_data, CostDataNumDiff):
                        cost_data.data0.contact = contacts[contact_key[0]]
                        for datax in cost_data.datax:
                            datax.contact = contacts[contact_key[0]]
                        for datau in cost_data.datau:
                            datau.contact = contacts[contact_key[0]]
                    else:
                        assert(False)
                    
        
        return problem

 
    def createSwingFootModel(self, conf, supportFootIds, comTask=None, clearanceTask=None):
        """ Action model for a swing foot phase.

        :param timeStep: step duration of the action model
        :param supportFootIds: Ids of the constrained feet
        :param comTask: CoM task
        :param swingFootTask: swinging foot task
        :return action model for a swing foot phase
        """
        # Creating the action model for floating-base systems. A walker system
        # is by default a floating-base system
        actModel = ActuationModelFreeFloating(self.rmodel)

        # Creating the cost model for a contact phase
        costModel = CostModelSum(self.rmodel, actModel.nu)

        # Creating a 3D multi-contact model, and then including the supporting
        # foot
        contactModel = ContactModelMultiple(self.rmodel)
        
        for i in supportFootIds:
            supportContactModel = ContactModel3D(self.rmodel, i, ref=[0., 0., 0.], gains=[conf.kp_contact, conf.kd_contact])
            contactModel.addContact('contact_' + str(i), supportContactModel)
            
            costFriction = CostModelForceLinearCone(self.rmodel, supportContactModel, self.A_friction_cones, nu=actModel.nu)
            #numeric friction 
#            state = StatePinocchio(self.rmodel)
#            nq = self.rmodel.nq
#            reevals=[lambda m, d, x, u: pinocchio.forwardKinematics(m, d, a2m(x[:nq]), a2m(x[nq:])), 
#                     lambda m, d, x, u: pinocchio.computeJointJacobians(m, d, a2m(x[:nq])), 
#                     lambda m, d, x, u: pinocchio.updateFramePlacements(m, d)]
#            costFrictionFinDiff = CostModelNumDiff(costFriction, state, withGaussApprox=True,
#                                                   reevals=reevals)
            #costModel.addCost("frictionCone_"+str(i), costFrictionFinDiff, conf.weight_friction)
            costModel.addCost("frictionCone_"+str(i), costFriction, conf.weight_friction)
        
        if isinstance(comTask, np.ndarray):
            comTrack = CostModelCoM(self.rmodel, comTask, actModel.nu, ActivationModelWeightedQuad(conf.weight_array_com**2))
            costModel.addCost("comTrack", comTrack, conf.weight_com)
        #clearance task
        if clearanceTask is not None and conf.weight_clearance > 0:
            for foot in clearanceTask:
                #set clearance only on z component
                activation = ActivationModelInequality(np.array([-1e08, -1e08, 0.0 ]), 
                                                             np.array([1e08, 1e08, 1e08]))    
                clearanceTrack = \
                        CostModelFrameTranslation(self.rmodel,
                                                  foot['id'],
                                                  foot['pos'],
                                                  nu=0,
                                                  activation = activation)           
                costModel.addCost("clearanceTrack_" + str(foot['id']), clearanceTrack, conf.weight_clearance)
                
        
        if conf.weight_joint_limits>0.0:                
            qMin = np.concatenate((-1e10*np.ones(6), m2a(self.rmodel.lowerPositionLimit)[7:]))
            qMax = np.concatenate((+1e10*np.ones(6), m2a(self.rmodel.upperPositionLimit)[7:]))
#            qMax[8] = -0.5
#            qMax[14] = -0.5
            vMin = np.array(conf.nv*[-1e10])
            vMax = np.array(conf.nv*[+1e10])
            activation =  ActivationModelInequality(np.concatenate((qMin, vMin)), np.concatenate((qMax, vMax)))
            jointLimits = CostModelState(self.rmodel, self.state, np.zeros(self.state.nx), actModel.nu,
                                         activation=activation)
            costModel.addCost("jointLim", jointLimits, conf.weight_joint_limits)
        
        if conf.weight_torque_limits>0.0:
            #print self.rmodel.effortLimit.T
            activation =  ActivationModelInequality(-m2a(self.rmodel.effortLimit)[6:], m2a(self.rmodel.effortLimit)[6:])
            torqueLimits = CostModelControl(self.rmodel, actModel.nu, activation=activation)
            costModel.addCost("torqueLim", torqueLimits, conf.weight_torque_limits)
            
        stateReg = CostModelState(self.rmodel, self.state, self.rmodel.defaultState, actModel.nu,
                                  ActivationModelWeightedQuad(conf.weight_array_postural**2))
        costModel.addCost("stateReg", stateReg, conf.weight_postural)                 
                  
        ctrlReg = CostModelControl(self.rmodel, actModel.nu)
        costModel.addCost("ctrlReg", ctrlReg, conf.weight_control)

        # Creating the action model for the KKT dynamics with simpletic Euler
        # integration scheme
        dmodel = DifferentialActionModelFloatingInContact(self.rmodel, actModel, contactModel, costModel)
        model = IntegratedActionModelEuler(dmodel)
        model.timeStep = conf.timeStep
        return model


    def createImpactModel(self, conf, supportFootIds, swingFootTask):
        """ Action model for impact models.

        An impact model consists of describing the impulse dynamics against a set of contacts.
        :param supportFootIds: Ids of the constrained feet
        :param swingFootTask: swinging foot task
        :return impact action model
        """
        # Creating a 3D multi-contact model, and then including the supporting foot
        #DYNAMICS
        impulseModel = ImpulseModelMultiple(self.rmodel)
        for i in supportFootIds:
            supportContactModel = ImpulseModel3D(self.rmodel, i)
            impulseModel.addImpulse("impulse_" + str(i), supportContactModel)

        #1 - COST
        # Creating the cost model for a contact phase
        costModel = CostModelSum(self.rmodel, nu=0)

        #POSTURAL ||qdes-q||
        stateReg = CostModelState(self.rmodel, self.state, self.rmodel.defaultState, 0,
                                  ActivationModelWeightedQuad(conf.weight_array_postural_impact**2))
                                                                                                 
        #2 - COST to have the feet in a certain position                                 
        costModel.addCost("stateReg", stateReg, conf.weight_postural_impact)
        if swingFootTask is not None:
            for i in swingFootTask:
                #
                weights = np.array([conf.weight_foot_pos_impact_xy, conf.weight_foot_pos_impact_xy, conf.weight_foot_pos_impact_z])
                activation = ActivationModelWeightedQuad(weights)
                footTrack = \
                    CostModelFrameTranslation(self.rmodel,
                                              i.frameId,
                                              m2a(i.oXf.translation),
                                              nu=0,
                                              activation = activation)
                costModel.addCost("footTrack_" + str(i.frameId), footTrack, 1.0)
                
                # impactFootVelCost = CostModelFrameVelocity(self.rmodel, i.frameId, nu=0)
                # costModel.addCost("impactVel_"+str(i),
                #                   impactFootVelCost, 1e6)

        # Creating the action model for the KKT dynamics with simpletic Euler
        # integration scheme
        model = ActionModelImpact(self.rmodel, impulseModel, costModel)
        return model

def interpolate_trajectory(x_in, dt_in, dt_out):
    # INTERPOLATE WITH DIFFERENT TIME STEP
    assert(dt_out<=dt_in)
    
    m = x_in.shape[1]      # size of x vector
    N_in  = x_in.shape[0] # number of time steps of x
    N_out = int(((N_in-1)*dt_in)/dt_out) + 1   # number of time steps in output traj
    x_out  = np.empty((N_out,m))*np.nan
    
    N_inner = int(dt_in/dt_out)
    for i in range(N_in-1):                    
        for j in range(N_inner):
            ii = i*N_inner + j
            alpha = j/float(N_inner)
            x_out[ii,:] = (1-alpha)*x_in[i,:] + alpha*x_in[i+1,:]
    x_out[-1,:] = x_in[-1,:]

    return x_out
