#!/usr/bin/env python3

import rospy
import sys

from std_msgs.msg import String

from task_behavior_engine import branch
from task_behavior_engine import tree

from task_behavior_ros import introspection
from task_behavior_ros import time
from task_behavior_ros import topic


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

class MyCounter(tree.Node):

    """ A counter wait returns SUCCEED after time has lasped.
        @param name [str] The name of this node
        @param counter [int] Amount of counts

        configurable nodedata:
            counter [int] Amount of counts
    """

    def __init__(self, name, counter, *args, **kwargs):
        super(MyCounter, self).__init__(
            name=name, configure_cb=self.configure, run_cb=self.run, *args, **kwargs)
        rospy.loginfo("setting up MyCounter for " + self._name)
        self.counter = counter

    def configure(self, nodedata):
        rospy.logdebug("MyCounter.configure()" + self._name)
        self.counter = nodedata.get_data('counter', self.counter)

    def run(self, nodedata):
        rospy.logdebug("MyCounter.run()" + self._name)
        if self.counter > 0:
            self.counter -= 1
        else:
            return tree.NodeStatus(tree.NodeStatus.SUCCESS, "Counting finished")
        rospy.loginfo(self._name + " counting remaining: " + str(self.counter))

        return tree.NodeStatus(tree.NodeStatus.ACTIVE)


def main_sequencer(name):
    ''' Main behavior of the tree
        @param name [str] The name of the behavior
    '''

    WaypointNavigation = MyCounter(name="WaypointNavigation", counter=5)

    NoToyDetected = MyCounter(name="NoToyDetected", counter=5)
    SetWaypoint = MyCounter(name="SetWaypoint", counter=7)

    PushCurrentLocation = MyCounter(name="PushCurrentLocation", counter=6)
    ApproachToy = MyCounter(name="ApproachToy", counter=5)
    GrabToy = MyCounter(name="GrabToy", counter=5)
    DropboxNavigation = MyCounter(name="DropboxNavigation", counter=5)
    DropToy = MyCounter(name="DropToy", counter=5)

    nav_sel = branch.Selector("nav_selector")

    arm_seq = branch.Sequencer("arm_sequencer")
    nav_seq = branch.Sequencer("nav_sequencer")

    main_seq = branch.Sequencer(name)
    main_seq.add_child(nav_sel)
    main_seq.add_child(SetWaypoint)

    nav_sel.add_child(nav_seq)
    nav_sel.add_child(arm_seq)

    nav_seq.add_child(NoToyDetected)
    nav_seq.add_child(WaypointNavigation)

    arm_seq.add_child(PushCurrentLocation)
    arm_seq.add_child(ApproachToy)
    arm_seq.add_child(GrabToy)
    arm_seq.add_child(DropboxNavigation)
    arm_seq.add_child(DropToy)

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
