import rospy
from task_behavior_engine import tree

class DropboxNavigation(tree.Node):

    """ A counter wait returns SUCCEED after time has lasped.
        @param name [str] The name of this node
        @param counter [int] Amount of counts

        configurable nodedata:
            counter [int] Amount of counts
    """

    def __init__(self, name, counter, *args, **kwargs):
        super(DropboxNavigation, self).__init__(
            name=name, configure_cb=self.configure, run_cb=self.run, *args, **kwargs)
        rospy.loginfo("setting up DropboxNavigation for " + self._name)
        self.counter = counter

    def configure(self, nodedata):
        rospy.logdebug("DropboxNavigation.configure()" + self._name)
        self.counter = nodedata.get_data('counter', self.counter)

    def run(self, nodedata):
        rospy.logdebug("DropboxNavigation.run()" + self._name)
        if self.counter > 0:
            self.counter -= 1
        else:
            return tree.NodeStatus(tree.NodeStatus.SUCCESS, "Counting finished")
        rospy.loginfo(self._name + " counting remaining: " + str(self.counter))

        return tree.NodeStatus(tree.NodeStatus.ACTIVE)