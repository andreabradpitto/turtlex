#!/usr/bin/env python3

import rospy
import sys

from std_msgs.msg import String

from task_behavior_engine import branch
from task_behavior_engine import tree

from task_behavior_ros import introspection
from task_behavior_ros import time
from task_behavior_ros import topic

import SetWaypoint, NoToyDetected, WaypointNavigation, PushCurrentLocation, \
       ApproachToy, GrabToy, DropboxNavigation, DropToy


# sequencer
# selector, SetWaypoint
# sequencer, sequencer | 
# NoToyDetected, WaypointNavigation | PushCurrentLocation, ApproachToy, GrabToy, DropboxNavigation, DropToy

# In pratica quando ho due nodi se quello "a sinistra" finisce e parte quello "a destra", allora lo devo
# in generale fare blocking, cioè non fa return active. Altrimenti riparte subito un nuovo giro di quello
# prioritario; l'unica eccezione è con NoToyDetected e WaypointNavigation, ove il primo deve essere immediato,
# mentre il secondo è active. Questo perché cosi la navigazione viene fermata ogni tot per vedere se la telecamera
# ha visto qualcosa. NoToyDetected ritorna FAIL, così parte l'arm_sequencer. DropToy darà fail così si passa a SetWaypoint.
# Forse WaypointNavigation non deve accedere a SetWaypoint? Ho visto che dopo successo di NoToyDetected e WaypointNavigation
# parte SetWaypoint


def main_sequencer(name):
    ''' Main behavior of the tree
        @param name [str] The name of the behavior
    '''

    main_seq = branch.Sequencer(name)

    nav_sel = branch.Selector("nav_selector")
    set_waypoint = SetWaypoint(name="SetWaypoint", counter=7)

    arm_seq = branch.Sequencer("arm_sequencer")
    nav_seq = branch.Sequencer("nav_sequencer")

    no_toy_detected = NoToyDetected(name="NoToyDetected", counter=5)
    waypoint_navigation = WaypointNavigation(name="WaypointNavigation", counter=5)

    push_current_location = PushCurrentLocation(name="PushCurrentLocation", counter=6)
    approach_toy = ApproachToy(name="ApproachToy", counter=5)
    grab_toy = GrabToy(name="GrabToy", counter=5)
    dropbox_navigation = DropboxNavigation(name="DropboxNavigation", counter=5)
    drop_toy = DropToy(name="DropToy", counter=5)

    main_seq.add_child(nav_sel)
    main_seq.add_child(set_waypoint)

    nav_sel.add_child(nav_seq)
    nav_sel.add_child(arm_seq)

    nav_seq.add_child(no_toy_detected)
    nav_seq.add_child(waypoint_navigation)

    arm_seq.add_child(push_current_location)
    arm_seq.add_child(approach_toy)
    arm_seq.add_child(grab_toy)
    arm_seq.add_child(dropbox_navigation)
    arm_seq.add_child(drop_toy)

    return main_seq


if __name__ == '__main__':

    rospy.init_node('turtlex_bt')
    rospy.logdebug("Behavior Tree started")

    behavior = main_sequencer(name="turtlex_sequencer")  # Create the main behavior

    rospy.on_shutdown(behavior._cancel)  # Run any cancel hooks on shutdown (ctrl+c)

    ros_status = introspection.Introspection(behavior)  # Set up the inspection server to publish current status

    r = rospy.Rate(5)  # Set up repeating rate to tick the behavior

    try:
        while not rospy.is_shutdown():

            result = behavior.tick()  # Tick the behavior
            ros_status.publish_status()  # Publish the status to the introspection server

            rospy.loginfo("behavior result: " + str(result))

            if not result == tree.NodeStatus.ACTIVE:
                rospy.logwarn("finised with status: " + str(result))
            r.sleep()
    except:
        e = sys.exc_info()[0]
        rospy.logerr(e)
