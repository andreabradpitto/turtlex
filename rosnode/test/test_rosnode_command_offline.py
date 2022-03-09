#!/usr/bin/env python3

import os
import sys 
import unittest
import time
        
from subprocess import Popen, PIPE, check_call, call

class TestRosnodeOffline(unittest.TestCase):

    def setUp(self):
        pass

    ## test that the rosmsg command works
    def test_cmd_help(self):
        cmd = 'rosnode'
        sub = ['ping', 'machine', 'list', 'info', 'kill']
        
        output = Popen([cmd], stdout=PIPE).communicate()[0]
        self.assert_('Commands' in output)
        output = Popen([cmd, '-h'], stdout=PIPE).communicate()[0]
        self.assert_('Commands' in output)
        for c in sub:
            # make sure command is in usage statement
            self.assert_("%s %s"%(cmd, c) in output)

        for c in sub:
            output = Popen([cmd, c, '-h'], stdout=PIPE, stderr=PIPE).communicate()
            self.assert_("Usage:" in output[0], "[%s]: %s"%(c, output))
            self.assert_("%s %s"%(cmd, c) in output[0], "%s: %s"%(c, output[0]))
            
        # test no args on commands that require args
        for c in ['ping', 'info']:
            output = Popen([cmd, c], stdout=PIPE, stderr=PIPE).communicate()
            self.assert_("Usage:" in output[0] or "Usage:" in output[1], "[%s]: %s"%(c, output))
            
    def test_offline(self):
        cmd = 'rosnode'

        # point at a different 'master'
        env = os.environ.copy()
        env['ROS_MASTER_URI'] = 'http://localhost:11312'
        kwds = { 'env': env, 'stdout': PIPE, 'stderr': PIPE}

        msg = "ERROR: Unable to communicate with master!\n"

        output = Popen([cmd, 'list',], **kwds).communicate()
        self.assert_(msg in output[1])
        output = Popen([cmd, 'ping', 'talker'], **kwds).communicate()
        self.assertEquals(msg, output[1])
        output = Popen([cmd, 'info', 'talker'], **kwds).communicate()
        self.assert_(msg in output[1])

        output = Popen([cmd, 'kill', 'talker'], **kwds).communicate()
        self.assert_(msg in output[1])
        
