import numpy as np

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