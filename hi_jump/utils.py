import numpy as np
from pinocchio.robot_wrapper import RobotWrapper
import pinocchio as se3


def setWhiteBackground(robot):
    if not hasattr(robot, 'viewer'):
        # Spawn robot model
        robot.initDisplay(loadModel=True)
        # Set white background and floor
        window_id = robot.viewer.gui.getWindowID('python-pinocchio')
        robot.viewer.gui.setBackgroundColor1(window_id, [1., 1., 1., 1.])
        robot.viewer.gui.setBackgroundColor2(window_id, [1., 1., 1., 1.])
        robot.viewer.gui.addFloor('hpp-gui/floor')
        robot.viewer.gui.setScale('hpp-gui/floor', [0.5, 0.5, 0.5])
        robot.viewer.gui.setColor('hpp-gui/floor', [0.7, 0.7, 0.7, 1.])
        robot.viewer.gui.setLightingMode('hpp-gui/floor', 'OFF')


def displayPhaseMotion(robot, qs, ts):
    import time
    if len(qs) == len(ts) + 1:
        for k, q in enumerate(qs[1:]):
            dt = ts[k]
            robot.display(np.matrix(q).T)
            time.sleep(dt)
    else:
        for k, q in enumerate(qs):
            dt = ts[k]
            robot.display(np.matrix(q).T)
            time.sleep(dt)
            
def loadRobot(urdfFileName, urdfSubPath, q0, modelPath='/opt/openrobots/share/example-robot-data'):
    #URDF_FILENAME = "hyq_last.urdf" #no torque limits
#    URDF_FILENAME = "hyq_last_torque_lim.urdf"
#    URDF_SUBPATH = "/hyq_description/robots/" + URDF_FILENAME

#    URDF_FILENAME = "solo.urdf" #no torque limits   
#    URDF_SUBPATH = "/robot_properties_solo/urdf/" + urdfFileName 
    robot = RobotWrapper.BuildFromURDF(modelPath + urdfSubPath + urdfFileName, [modelPath], se3.JointModelFreeFlyer())
    # TODO define default position inside srdf
    robot.q0.flat[:] = q0
    robot.model.referenceConfigurations['half_sitting'] = robot.q0
    return robot