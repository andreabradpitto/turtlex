#!/usr/bin/env python3

import rospy
import smach
import smach_ros
import time
import random

## Acquire maximum x-axis parameter from launch file
#map_x_max = rospy.get_param('map/x_max')
## Acquire maximum y-axis parameter from launch file
#map_y_max = rospy.get_param('map/y_max')
## Acquire minimum x-axis parameter from launch file
#map_x_min = rospy.get_param('map/x_min')
## Acquire minimum y-axis parameter from launch file
#map_y_min = rospy.get_param('map/y_min')
## Acquire x-axis home position parameter from launch file
#home_x = rospy.get_param('home/x')
## Acquire y-axis home position parameter from launch file
#home_y = rospy.get_param('home/y')


class Clean(smach.State):
    # state initialization: set the outcomes
    def __init__(self):
        # initialisation function, it should not wait
        smach.State.__init__(self, outcomes=['obs_detected'])

    # Clean state execution
    def execute(self, userdata):
        # function called when exiting from the node, it can be blocking

        # define loop rate for the state
        self.rate = rospy.Rate(200)

        time.sleep(1)

        return 'obs_detected'


class Parse(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['small_obs', 'wall_obs'])

    def execute(self, userdata):

        time.sleep(1)
 
        if random.randint(0, 2) < 1:
            return 'wall_obs'

        else:
            return 'small_obs'


class Grab(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['grab_done'])

        # subscribed topic, used to receive commands from the human.py node                    
        #rospy.Subscriber('play_topic', String, self.room_color_callback)

    def execute(self, userdata):

        return 'grab_done'

    ## Grab state callback
    #def room_color_callback(self, data):
        #rospy.loginfo('dog')


class Dispose(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['obs_disposed'])

    def execute(self, userdata):
        return 'obs_disposed'


def main():
    rospy.init_node('dog_fsm_node', anonymous = True)

    # Create a SMACH state machine
    sm = smach.StateMachine(outcomes=['container_interface'])
    # Open the container
    with sm:
        # Add states to the container
        smach.StateMachine.add('CLEAN', Clean(), 
                               transitions={'obs_detected':'PARSE'})

        smach.StateMachine.add('PARSE', Parse(), 
                               transitions={'small_obs':'GRAB', 
                                            'wall_obs':'CLEAN'})

        smach.StateMachine.add('GRAB', Grab(), 
                               transitions={'grab_done':'DISPOSE'})

        smach.StateMachine.add('DISPOSE', Dispose(), 
                               transitions={'obs_disposed':'CLEAN'})

    # Create and start the introspection server for visualization
    sis = smach_ros.IntrospectionServer('server_name', sm, '/SM_ROOT')
    sis.start()

    # outcome = sm.execute() # an output variable is to be used if
                             # this finite state machine is nested
                             # inside another one

    # Execute the state machine
    sm.execute()

    # Wait for ctrl+c to stop the application
    rospy.spin()
    sis.stop()

if __name__ == '__main__':
    main()
