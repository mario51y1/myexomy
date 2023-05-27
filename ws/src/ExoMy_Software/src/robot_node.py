#!/usr/bin/env python
import time
from exomy.msg import RoverCommand, MotorCommands, Screen
import rospy
from rover import Rover
import message_filters
from exomy_detect.msg import Detection
from std_msgs import Bool

global exomy
exomy = Rover()

global auto_mode
auto_mode = False
global annotations
annotations = None

def joy_callback(message):
    cmds = MotorCommands()

    if message.motors_enabled is True:
        if auto_mode:
            cmds.motor_angles = exomy.autoSteeringAngle(annotations)
            cmds.motor_speeds = exomy.autoVelocity(annotations)
        else:
            exomy.setLocomotionMode(message.locomotion_mode)

            cmds.motor_angles = exomy.joystickToSteeringAngle(
                message.vel, message.steering)
            cmds.motor_speeds = exomy.joystickToVelocity(
                message.vel, message.steering)
    else:
        cmds.motor_angles = exomy.joystickToSteeringAngle(0, 0)
        cmds.motor_speeds = exomy.joystickToVelocity(0, 0)

    robot_pub.publish(cmds)

def set_auto_callback(message):
    auto_mode = True if message.data else False
    print('Received change: auto = ' + str(auto_mode))

def set_anotation_callback(message):
    annotations=message

if __name__ == '__main__':
    rospy.init_node('robot_node')
    rospy.loginfo("Starting the robot node")
    global robot_pub
    joy_sub = rospy.Subscriber(
        "/rover_command", RoverCommand, joy_callback, queue_size=1)

    annotation_sub = rospy.Subscriber('annotated_image/data', Detection, set_anotation_callback)
    annotation_sub = rospy.Subscriber('set_auto', Bool, set_auto_callback)

    rate = rospy.Rate(10)

    robot_pub = rospy.Publisher("/motor_commands", MotorCommands, queue_size=1)

    rospy.spin()
