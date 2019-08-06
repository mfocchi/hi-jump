import numpy as np
import pinocchio
import optim_params as conf
from crocoddyl import (ActivationModelWeightedQuad, ActivationModelInequality, ActuationModelFreeFloating, ActionModelImpact, CallbackDDPLogger,
                       CallbackDDPVerbose, CallbackSolverDisplay, ContactModel3D, ContactModelMultiple, CostModelCoM,
                       CostModelControl, CostModelFrameTranslation, CostModelForceLinearCone, CostModelState,
                       CostModelSum, DifferentialActionModelFloatingInContact, IntegratedActionModelEuler,
                       ImpulseModelMultiple, ImpulseModel3D, ShootingProblem, SolverDDP, SolverFDDP, StatePinocchio,
                       a2m, displayTrajectory, loadHyQ, m2a, CostDataForceLinearCone, ActionDataImpact)


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
    def __init__(self, rmodel, lfFoot, rfFoot, lhFoot, rhFoot):
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

    def createJumpingProblem(self, x0, jumpHeight, jumpLength, timeStep, groundKnots, flyingKnots):
        q0 = a2m(x0[:self.rmodel.nq])
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        rfFootPos0 = self.rdata.oMf[self.rfFootId].translation
        rhFootPos0 = self.rdata.oMf[self.rhFootId].translation
        lfFootPos0 = self.rdata.oMf[self.lfFootId].translation
        lhFootPos0 = self.rdata.oMf[self.lhFootId].translation
        df = jumpLength[2] - rfFootPos0[2]
        rfFootPos0[2] = 0.
        rhFootPos0[2] = 0.
        lfFootPos0[2] = 0.
        lhFootPos0[2] = 0.
        comRef = m2a(rfFootPos0 + rhFootPos0 + lfFootPos0 + lhFootPos0) / 4
        comRef[2] = m2a(pinocchio.centerOfMass(self.rmodel, self.rdata, q0))[2]

        four_foot_support = [self.lfFootId, self.lhFootId, self.rfFootId, self.rhFootId]
        loco3dModel = []
        takeOff = [
            self.createSwingFootModel(
                timeStep,
                four_foot_support,
            ) for k in range(groundKnots)
        ]
        flyingUpPhase = [
            self.createSwingFootModel(timeStep, [],
                                      np.array([0.5*jumpLength[0], 0.5*jumpLength[1],  jumpHeight]) * (k + 1) / flyingKnots + comRef)
            for k in range(flyingKnots)
        ]
#        flyingUpPhase += [self.createSwingFootModel(timeStep, [], 
#                                                    np.array([0.5*jumpLength[0], 0.5*jumpLength[1],  jumpHeight]) + comRef)]

        flyingDownPhase = []
        for k in range(flyingKnots):
            flyingDownPhase += [self.createSwingFootModel(timeStep, [])]

        f0 = np.matrix(jumpLength).T
        footTask = [
            TaskSE3(pinocchio.SE3(np.eye(3), lfFootPos0 + f0), self.lfFootId),
            TaskSE3(pinocchio.SE3(np.eye(3), rfFootPos0 + f0), self.rfFootId),
            TaskSE3(pinocchio.SE3(np.eye(3), lhFootPos0 + f0), self.lhFootId),
            TaskSE3(pinocchio.SE3(np.eye(3), rhFootPos0 + f0), self.rhFootId)
        ]
        landingPhase = [
            self.createImpactModel(four_foot_support, footTask)
        ]
        f0[2] = df
        landed = [
            self.createSwingFootModel(timeStep, four_foot_support, comTask=comRef + m2a(f0))
            for k in range(groundKnots)
        ]
        loco3dModel += takeOff
        loco3dModel += flyingUpPhase
        loco3dModel += flyingDownPhase
        loco3dModel += landingPhase
        loco3dModel += landed
        
        problem = ShootingProblem(x0, loco3dModel, loco3dModel[-1])
        
        # QUICK FIX: Set contactData on CostDataForceLinearCone 
        for m in problem.runningDatas + [problem.terminalData]:
            # skip the impact phase
            if isinstance(m, ActionDataImpact): continue
                
            contacts = m.differential.contact.contacts
            costs = m.differential.costs.costs
            
            # skip the flying phase
            if len(contacts)==0: continue
                
            for foot_id in four_foot_support:
                contact_key = [key for key in contacts.keys() if str(foot_id) in key]
                friction_key = [key for key in costs.keys() if str(foot_id) in key]
                assert(len(contact_key)==1)
                assert(len(friction_key)==1)
                assert(isinstance(costs[friction_key[0]], CostDataForceLinearCone))
                costs[friction_key[0]].contact = contacts[contact_key[0]]
        
        return problem

 
    def createSwingFootModel(self, timeStep, supportFootIds, comTask=None):
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
            costModel.addCost("frictionCone_"+str(i), costFriction, conf.weight_friction)
        
        if isinstance(comTask, np.ndarray):
            comTrack = CostModelCoM(self.rmodel, comTask, actModel.nu)
            costModel.addCost("comTrack", comTrack, conf.weight_com)


        stateReg = CostModelState(self.rmodel, self.state, self.rmodel.defaultState, actModel.nu,
                                  ActivationModelWeightedQuad(conf.weight_array_postural**2))
                         
                  
        ctrlReg = CostModelControl(self.rmodel, actModel.nu)
        costModel.addCost("stateReg", stateReg, conf.weight_postural)
        costModel.addCost("ctrlReg", ctrlReg, conf.weight_control)

        # Creating the action model for the KKT dynamics with simpletic Euler
        # integration scheme
        dmodel = DifferentialActionModelFloatingInContact(self.rmodel, actModel, contactModel, costModel)
        model = IntegratedActionModelEuler(dmodel)
        model.timeStep = timeStep
        return model


    def createImpactModel(self, supportFootIds, swingFootTask):
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
                                              nu=0)
                costModel.addCost("footTrack_" + str(i.frameId), footTrack, conf.weight_foot_pos_impact)
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
