from hi_jump.communication.communication import ControlThread, talker
import rospy as ros
import hi_jump.scripts.optim_params as conf

p = ControlThread(conf)
try:
    talker(p)
except ros.ROSInterruptException:
    pass