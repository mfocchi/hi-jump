# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 16:52:08 2018

@author: rorsolino
"""

#!/usr/bin/env python

import copy
import numpy as np
from scipy.linalg import block_diag
import os

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


import rospy as ros
import sys
import time
import threading



from std_srvs.srv    import Empty, EmptyRequest

from sensor_msgs.msg import JointState
from gazebo_msgs.msg import ContactsState
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState

from tf.transformations import euler_from_quaternion
from std_srvs.srv import Empty
from termcolor import colored
from jet_leg.hyq_kinematics import HyQKinematics

from utils import Utils
from jet_leg.math_tools import Math

from ros_impedance_controller.srv import set_pids
from ros_impedance_controller.srv import set_pidsRequest
from ros_impedance_controller.msg import pid

import hi_jump.utils as  hijump_utils

#important
np.set_printoptions(precision = 3, linewidth = 200, suppress = True)

#prevent creating pyc files
sys.dont_write_bytecode = True


FIGURE_PATH = '../figs/'
FILE_EXTENSIONS = ['png']; #'pdf']; #['png']; #,'eps'];
FIGURES_DPI = 150;

#dont trukate printing of matrices!
np.set_printoptions(threshold=np.inf)

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
#import utils
sys.stderr = stderr


class ControlThread(threading.Thread):
    def __init__(self, conf):  
        
        threading.Thread.__init__(self)
        
        self.conf = conf
        self.grForcesW = np.zeros(12)
        self.basePoseW = np.zeros(6)
        self.quaternion = np.zeros(4)
        self.baseTwistW = np.zeros(6)
        self.q = np.zeros(12)
        self.qd = np.zeros(12)
        self.q_des = np.array([-0.2, 0.7, -1.4, -0.2, 0.7, -1.4, -0.2, -0.7, 1.4, -0.2, -0.7, 1.4]);
        self.qd_des = np.zeros(12)
        self.tau_ffwd =np.zeros(12)
        self.tau =np.zeros(12)
        
        self.log_data = False
        self.f_log = []
                
        self.contact_counter = 0
        self.joint_counter = 0
        self.sim_time  = 0.0
        self.numberOfReceivedMessages = 0
        self.numberOfPublishedMessages = 0

        self.robotMass = 0
        self.desVelocity = np.array([0.0, 0.0,0.0])
        self.desAngVelocity = np.array([0.0, 0.0, 0.0])
        self.u = Utils()
        timer = 0.0
        
    def start_log(self):
        self.log_data = True
    
    def stop_log(self):
        self.log_data = False
        
    def run(self):
        
        self.robot_name = ros.get_param('/robot_name')
        # subscribers
        # contact
        self.sub_contact = ros.Subscriber("/"+self.robot_name+"/contacts_state", ContactsState, callback=self._receive_contact, queue_size=100)
        # base pose
        self.sub_pose = ros.Subscriber("/"+self.robot_name+"/ground_truth", Odometry, callback=self._receive_pose, queue_size=100)
        # joint states
        self.sub_pose = ros.Subscriber("/"+self.robot_name+"/joint_states", JointState, callback=self._receive_jstate, queue_size=100)
        # publishers
        # impedance controller
        self.pub_des_jstate = ros.Publisher("/"+self.robot_name+"/ros_impedance_controller/command", JointState, queue_size=1)
        ros.wait_for_service("/"+self.robot_name+"/freeze_base")

        # services
        # freezeBase
        self.freeze_base = ros.ServiceProxy("/"+self.robot_name+"/freeze_base",Empty)
        self.pause_physics_client = ros.ServiceProxy('/gazebo/pause_physics', Empty)
        self.unpause_physics_client = ros.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.set_pd_service = ros.ServiceProxy("/"+self.robot_name+"/ros_impedance_controller/set_pids", set_pids)
   

    def loadConfig(self):
        return

    def setPDs(self, kp, kd, ki):
        
        #create the message
        req_msg = set_pidsRequest()
        req_msg.data = []
     
        #fill in the message with des values for kp kd     
        for i in range(len(self.joint_names)):            
            joint_pid = pid()
            joint_pid.joint_name = self.joint_names[i]
            joint_pid.p_value = kp 
            joint_pid.d_value = kd
            joint_pid.i_value = ki
            req_msg.data += [joint_pid]
            
        #send request and get response (in this case none)
        self.set_pd_service(req_msg)

    


    def _receive_contact(self, msg):
        self.contact_counter += 1
#        print self.contact_counter, "Contact received", msg.header.stamp
        
        self.grForcesW[0] = msg.states[0].wrenches[0].force.x
        self.grForcesW[1] =  msg.states[0].wrenches[0].force.y
        self.grForcesW[2] =  msg.states[0].wrenches[0].force.z
        self.grForcesW[3] = msg.states[1].wrenches[0].force.x
        self.grForcesW[4] =  msg.states[1].wrenches[0].force.y
        self.grForcesW[5] =  msg.states[1].wrenches[0].force.z
        self.grForcesW[6] = msg.states[2].wrenches[0].force.x
        self.grForcesW[7] =  msg.states[2].wrenches[0].force.y
        self.grForcesW[8] =  msg.states[2].wrenches[0].force.z
        self.grForcesW[9] = msg.states[3].wrenches[0].force.x
        self.grForcesW[10] =  msg.states[3].wrenches[0].force.y
        self.grForcesW[11] =  msg.states[3].wrenches[0].force.z
        
        if(self.log_data):
            self.f_log += [self.grForcesW.copy()]
            
    def _receive_pose(self, msg):
        # These are base pose and twist that is different than COM due to offset
        self.quaternion = (
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w)
        euler = euler_from_quaternion(self.quaternion, axes='szyx')
        

        self.basePoseW[0] = msg.pose.pose.position.x
        self.basePoseW[1] = msg.pose.pose.position.y
        self.basePoseW[2] = msg.pose.pose.position.z

        self.basePoseW[3] = euler[0]
        self.basePoseW[4] = euler[1]
        self.basePoseW[5] = euler[2]


        self.baseTwistW[0] = msg.twist.twist.linear.x
        self.baseTwistW[1] = msg.twist.twist.linear.y
        self.baseTwistW[2] = msg.twist.twist.linear.z
        self.baseTwistW[3] = msg.twist.twist.angular.x
        self.baseTwistW[4] = msg.twist.twist.angular.y
        self.baseTwistW[5] = msg.twist.twist.angular.z

   
    def _receive_jstate(self, msg):
        self.joint_counter += 1
#        print self.joint_counter, "Joint state received", msg.header.stamp
        #need to map to robcogen only the arrays coming from gazebo
        self.q = self.u.mapFromRos(msg.position)
        self.qd = self.u.mapFromRos(msg.velocity)
        self.tau = self.u.mapFromRos(msg.effort)
        self.joint_names = msg.name
        self.numberOfReceivedMessages+=1
        
    def send_des_jstate(self, q_des, qd_des, tau_ffwd):
         msg = JointState()
         msg.position = q_des
         msg.velocity = qd_des
         msg.effort = tau_ffwd                
         self.pub_des_jstate.publish(msg)     
         self.numberOfPublishedMessages+=1

    def register_node(self):
        ros.init_node('sub_pub_node_python', anonymous=False)

    def deregister_node(self):
        ros.signal_shutdown("manual kill")
        
    def get_sim_time(self):
        return self.sim_time
        
    def get_contact(self):
        return self.contactsW
    def get_pose(self):
        return self.basePoseW
    def get_jstate(self):
        return self.q

    def initKinematics(self,kin):
        kin.init_homogeneous()
        kin.init_jacobians()

    def updateKinematics(self,kin):
        # q is continuously updated
        kin.update_homogeneous(self.q)
        kin.update_jacobians(self.q)
        self.actual_feetB = kin.forward_kin(self.q)

    def plotJoints(self, q, q_des, label="q"):
        plt.figure()
        plt.title("Joints")
        
        for i in range(12) :
            plt.subplot(4, 3, i+1)
            plt.plot(q[:,i] , label="Actual")
            plt.plot(q_des[:,i], label="Desired", linestyle='--')
            plt.ylabel(label+" "+str(i), fontsize=10)
            plt.grid()
            if i==0: plt.legend(loc="best")
            scale =  self.conf.timeStep / self.conf.dt
            flyingUpStart =self.conf.groundKnots
            flyingDownStart = flyingUpStart + self.conf.flyingKnots
            touchDownStart  =  flyingDownStart + self.conf.flyingKnots  
            
            plt.axvspan(0, scale*self.conf.groundKnots, alpha=0.3, color='red') 
            plt.axvspan(scale*flyingUpStart, scale*(flyingUpStart + self.conf.flyingKnots), alpha=0.2, color='green') 
            plt.axvspan(scale*flyingDownStart, scale*(flyingDownStart + self.conf.flyingKnots),  alpha=0.2, color='blue') 
            plt.axvspan(scale*touchDownStart, scale*(touchDownStart + self.conf.groundKnots),  alpha=0.2, color='gray') 

        self.saveFigure(label)
                
    def plotGRF(self, f, f_des):
        plt.figure()
        plt.title("Forces")        
        N_f = f.shape[0]
        N_f_des = f_des.shape[0]
        time_f = np.arange(0.0, N_f_des*self.conf.dt, self.conf.dt*N_f_des/N_f)
        time_f_des = np.arange(0.0, N_f_des*self.conf.dt, self.conf.dt)        
        for i in range(12) :
            plt.subplot(4, 3, i+1)
            plt.plot(time_f, f[:,i], label="Actual")
            plt.plot(time_f_des, f_des[:,i], label="Desired", linestyle='--')
            plt.ylabel("f "+str(i), fontsize=10)
            plt.grid()
            if i==0: plt.legend(loc="best")
                    
            flyingUpStart =self.conf.groundKnots
            flyingDownStart = flyingUpStart + self.conf.flyingKnots
            touchDownStart  =  flyingDownStart + self.conf.flyingKnots         
            
            plt.axvspan(0, self.conf.timeStep*self.conf.groundKnots, alpha=0.3, color='red') 
            plt.axvspan(self.conf.timeStep*flyingUpStart, self.conf.timeStep*(flyingUpStart + self.conf.flyingKnots), alpha=0.2, color='green') 
            plt.axvspan(self.conf.timeStep*flyingDownStart, self.conf.timeStep*(flyingDownStart + self.conf.flyingKnots),  alpha=0.2, color='blue') 
            plt.axvspan(self.conf.timeStep*touchDownStart, self.conf.timeStep*(touchDownStart + self.conf.groundKnots),  alpha=0.2, color='gray') 

        self.saveFigure('grf')
        
    def plotCones(self, f, f_des):
        #this is a cross check for the friction cone constraints
        N_f = f.shape[0]
        N_f_des = f_des.shape[0]
        mu_f = np.zeros(N_f)
        mu_f_des = np.zeros(N_f_des)
        time_f = np.arange(0.0, N_f_des*self.conf.dt, self.conf.dt*N_f_des/N_f)
        time_f_des = np.arange(0.0, N_f_des*self.conf.dt, self.conf.dt)        
  
        plt.figure()        
        for j in range(4):
            plt.subplot(2, 2, j+1)
            for i in range(f_des.shape[0]):            
                if f_des[i,3*j+2] != 0:  mu_f_des[i] = np.linalg.norm(f_des[i,3*j:3*j+2])/f_des[i,3*j+2]
            for i in range(f.shape[0]):
                if f[i,3*j+2] != 0:  mu_f[i] = np.linalg.norm(f[i,3*j:3*j+2])/f[i,3*j+2]
            

            plt.plot(time_f, mu_f, label="actual mu")
            plt.plot(time_f_des, mu_f_des, label="desired mu", linestyle='--')
            plt.legend()
            plt.ylabel("mu", fontsize=10)
        self.saveFigure('mu')   

    def saveFigure(self,title):
        if(self.conf.SAVE_FIGURES):
            for ext in FILE_EXTENSIONS:
                plt.gcf().savefig(FIGURE_PATH+title.replace(' ', '_')+'.'+ext, format=ext, dpi=FIGURES_DPI, bbox_inches='tight');


    
def talker(p):

    p.start()
    p.register_node()

    #create the objects
    kin = HyQKinematics()
    math = Math()



    #load configs
    p.loadConfig()
    p.unpause_physics_client(EmptyRequest())

    name = "Python Controller"


    data = np.load(p.conf.data_file)
    q_des_array =  data['q_des']
    qd_des_array = data['qd_des']
    tau_des_array = data['tau_des']
    x_base_des_array = data["x_base_des"]
    v_base_des_array = data["v_base_des"]
    f_des_array = data["f_des"]
    euler_des_array = data["euler_des"]

    # GOZERO Keep the fixed configuration for the joints at the start of simulation 
    p.q_des = p.u.mapFromRos(q_des_array[0,:])
    
    p.setPDs(500.0, 16.0, 0.0)
    p.qd_des = np.zeros(12)
    p.tau_ffwd = np.zeros(12)
    gravity_comp = p.u.mapFromRos(data['tau_gravity'])
    resp = p.freeze_base()
    
    start_t = time.time()
    print("reset posture...")

    while time.time()-start_t < 1.5: 
        p.send_des_jstate(p.q_des, p.qd_des, p.tau_ffwd)
        time.sleep(p.conf.dt)
    print "q err prima freeze base", (p.q-p.q_des)
    resp = p.freeze_base()    
    time.sleep(2.0)


    print "q err pre grav comp", (p.q-p.q_des)
    print("compensating gravity...")
    start_t = time.time()
    while time.time()-start_t < 2.5: 
        p.send_des_jstate(p.q_des, p.qd_des, gravity_comp)    
        time.sleep(p.conf.dt)
    print "q err post grav comp", (p.q-p.q_des)
    
    
#    tau_int = np.zeros_like(p.tau)
#    count = 0
#    while True:
#        time.sleep(0.001)
#        tau_int += (p.q_des - p.q)*0.001*100
#        p.send_des_jstate(p.q_des, p.qd_des,  gravity_comp + tau_int)
#        count +=1        
#        if np.max(np.abs(p.q-p.q_des))<0.01: 
#            break
#        else:
#            if ((count % 300) == 0): print "Wait for integral to converge", np.max(np.abs(p.q-p.q_des))
#    

    
    #init the kinematics (homogeneous and jacobians for feet position I guess)
    p.initKinematics(kin)
    #update the kinematics
    p.updateKinematics(kin)
    # p.basePoseW[p.u.sp_crd["LX"]: p.u.sp_crd["LX"]+3] = np.array([1.5, 0, 0.6])
 
    # for loop 
    N = q_des_array.shape[0] 
    joint_counter_array = np.zeros(N)
    q_array = np.zeros_like(q_des_array)
    q_ros_array = np.zeros_like(q_des_array)
    qd_array = np.zeros_like(q_des_array)
    tau_array = np.zeros_like(q_des_array)
    f_array = np.zeros((N,12))
    f_counter_array = np.zeros(N)
    x_base_array = np.zeros((N,7))
    v_base_array = np.zeros((N,6))
    euler_array = np.zeros((N,3))
    
    time_start = np.zeros(N)
    time_spent = np.zeros(N)
    p.start_log()
    for i in range(N):
        time_start[i] =  time.time()       
#        if (i%15 ==0 ):        print p.u.mapFromRos(qd_des_array[i,:])
#        if (i%15 ==0 ):        print p.u.mapFromRos(tau_des_array[i,:])
        q_des_array[i,:] = p.u.mapFromRos(q_des_array[i,:])
        qd_des_array[i,:] = p.u.mapFromRos(qd_des_array[i,:])
        tau_des_array[i,:] = p.u.mapFromRos(tau_des_array[i,:])
        f_des_array[i,:] = p.u.mapFromRos(f_des_array[i,:])
        
        p.send_des_jstate(q_des_array[i,:], qd_des_array[i,:], tau_des_array[i,:]) 
        
        q_array[i,:] = p.q
        q_ros_array[i,:] = p.u.mapFromRos(p.q)
        qd_array[i,:] = p.qd
        tau_array[i,:] = p.tau
        f_array[i,:] = p.grForcesW
        f_counter_array[i] = p.contact_counter
        joint_counter_array[i] = p.joint_counter
        x_base_array[i,:] = np.concatenate((p.basePoseW[:3], p.quaternion))
        
        v_base_array[i,:] = p.baseTwistW
        euler_array[i,:] = p.basePoseW[3:6]
        
        time_spent[i] = time.time() - time_start[i]
        if(time_spent[i] < p.conf.dt): time.sleep(0.9*(p.conf.dt-time_spent[i]))
        
    p.stop_log()
    
    print 'de registering...'
    p.deregister_node()    
    
#    p.plotJoints(q_array, q_des_array, label="q")
#    p.plotJoints(qd_array, qd_des_array, label="qd")
#    p.plotJoints(tau_array, tau_des_array, label="tau")
    p.plotGRF(np.array(p.f_log), f_des_array)
#    p.plotCones(np.array(p.f_log), f_des_array)
#    
#    plt.figure()
#    for i in range(3):
#        plt.plot(x_base_array[:,i], label='x base '+str(i))
#        plt.plot(x_base_des_array[:,i], '--', label='x base des '+str(i))
#    plt.legend()  
#    p.saveFigure('x_base')
#    plt.figure()
#    for i in range(3):
#        plt.plot(v_base_array[:,i], label='v base '+str(i))
#        plt.plot(v_base_des_array[:,i], '--', label='v base des '+str(i))
#    plt.legend()
#    plt.show()  
#    p.saveFigure('v_base')

#    ROBOT =  hijump_utils.loadHyQ()
#    hijump_utils.setWhiteBackground(ROBOT)
#    ts = np.ones(x_base_array.shape[0])*p.conf.dt
#    hijump_utils.displayPhaseMotion(ROBOT, np.hstack((x_base_array, q_ros_array)), 10*ts)
#    
#    plt.figure()
#    for i in range(3):
#        plt.plot(euler_array[:,i], label='euler '+str(i))
#        plt.plot(euler_des_array[:,i], '--', label='euler des '+str(i))
#    plt.legend()
#    plt.show()
#

    

#    plt.figure()    
#    plt.plot(time_start[1:]-time_start[:-1], ' *')
#    plt.title("Time start")
#    plt.figure()
#    plt.plot(time_spent, ' *')
#    plt.title("Time spent")
#    plt.show()
    
    
    
    # computation_time = (time.time() - start_t_IP)
    # print("Total time: --- %s seconds ---" % computation_time)
    # print 'number of published messages ', p.numberOfPublishedMessages
    # avgTime = computation_time/p.numberOfPublishedMessages
    # print 'average publishing time [ms]', avgTime
    # print 'average publishing frequency [Hz]', 1.0/avgTime
    #
    # print 'number of received messages ', p.numberOfReceivedMessages
    # avgTime = computation_time/p.numberOfReceivedMessages
    # print 'average subscription time [ms]', avgTime
    # print 'average subscription frequency [Hz]', 1.0/avgTime
    

    
        