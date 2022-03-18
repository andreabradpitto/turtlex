#!/usr/bin/env python3

import rospy
import sys

from std_msgs.msg import String

from task_behavior_engine import branch
from task_behavior_engine import tree

from task_behavior_ros import introspection
from task_behavior_ros import time
from task_behavior_ros import topic


def timed_publish(name, msg, timeout):
    ''' Publish a std_msg/String message after timeout
        @param name [str] The name of the behavior
        @param msg [str] The message to publish
        @param timeout [float] The time (in seconds) to wait to publish
    '''
    timer = time.TimedWait(name="pub wait", timeout=timeout)
    message = String()
    message.data = msg
    publish = topic.TopicPublisher(
        name="publish", topic_name="test", topic_type=String, msg=message)

    publish_timer = branch.Sequencer(name)
    publish_timer.add_child(timer)
    publish_timer.add_child(publish)

    return publish_timer

if __name__ == '__main__':
    rospy.init_node('example')
    rospy.loginfo("started")

    # Create the timed_publish behavior.
    behavior = timed_publish(name="example", msg="hello world", timeout=1.0)

    # Run any cancel hooks on shutdown (ctrl+c).
    rospy.on_shutdown(behavior._cancel)
    # Set up the inspection server to publish current status.
    ros_status = introspection.Introspection(behavior)

    # Set up repeating rate to tick the behavior.
    r = rospy.Rate(10)

    try:
        while not rospy.is_shutdown():
            # Tick the behavior.
            result = behavior.tick()
            # Publish the status to the introspection server.
            ros_status.publish_status()
            rospy.loginfo("behavior result: " + str(result))

            if not result == tree.NodeStatus.ACTIVE:
                rospy.logwarn("finised with status: " + str(result))
            r.sleep()
    except:
        e = sys.exc_info()[0]
        rospy.logerr(e)
