from __future__ import absolute_import
from .tree import Node
from .tree import NodeStatus


class Success(Node):

    """ A Success node always returns NodeStatus.SUCCESS. """

    def __init__(self, name, *args, **kwargs):
        super(Success, self).__init__(name, run_cb=self.run, *args, **kwargs)

    def run(self, nodedata):
        return NodeStatus(NodeStatus.SUCCESS)


class Fail(Node):

    """ A Fail node always returns NodeStatus.FAIL. """

    def __init__(self, name, *args, **kwargs):
        super(Fail, self).__init__(name, run_cb=self.run, *args, **kwargs)

    def run(self, nodedata):
        return NodeStatus(NodeStatus.FAIL)


class Continue(Node):

    """ A Continue node always returns NodeStatus.ACTIVE. """

    def __init__(self, name, *args, **kwargs):
        super(Continue, self).__init__(name, run_cb=self.run, *args, **kwargs)

    def run(self, nodedata):
        return NodeStatus(NodeStatus.ACTIVE)
