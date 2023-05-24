#!/usr/bin/env python3

import yaml
import docker
import os
import git
import glob

class WorldAPIClient:
    def __init__(self, robot):
        super(WorldAPIClient, self).__init__(robot=node_name)
        self.robot = robot #upon calling this class from lib, specify input robot
        self.robot_type = "unknown"
        self.active_config = None
        self.config_path = None
