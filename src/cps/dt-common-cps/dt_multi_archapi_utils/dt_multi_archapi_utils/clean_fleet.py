#!/usr/bin/env python3
#This script is part of the DT Architecture Library for dt-commons

import yaml
import os
import git
import glob

from git import Repo
from dt_archapi_utils.arch_message import ApiMessage

'''
    THIS SCRIPT TAKES IN A FLEET .YAML FILE AS SPECIFIED IN THE DESIGN DOCUMENT.
    IT RETURNS A MORE CLEAN AND STRIPPED VERSION OF THE FLEET FILE, WITH ONLY
    THE HOSTNAMES OF THE DEVICES LISTED IN THE FLEET.
'''

class CleanFleet:
    def __init__(self):
        self.status = ApiMessage()
        self.fleet = None
        self.dt_version = "daffy"
        self.fleet_path = "/data/assets/dt-architecture-data/lists/" #change to data/config/fleets/...


    def clean_list(self, fleet=None):
        self.fleet = fleet
        fleet_list = {}

        #Warning
        if fleet is None:
            print("No fleet specified, please specify to avoid errors... using test file")
            self.fleet = "test"

        #For testing & development only
        try:
            with open(self.fleet_path + self.fleet + ".yaml", 'r') as f:
                file = yaml.load(f, Loader=yaml.FullLoader)
                if "devices" in file:
                    fleet_list = file["devices"]
                    print(fleet_list)
                    return fleet_list

            return fleet_list

        except FileNotFoundError: #error msg
            print("Error: did not find file")
            fleet_list = {"rom"}
            return fleet_list
            #return self.status.error(status="error", msg="data cannot be JSON decoded")

"""
        #Use fleet file written to disc through Dashboard
        if os.path.isdir("/data/config/fleets"):
            #For testing & development only
            fleet_path = "/data/config/fleets"
            path_to_list = fleet_path + "/" + str(fleet) + ".yaml"
            try:
                with open(path_to_list, 'r') as f:
                    file = yaml.load(f, Loader=yaml.FullLoader)
                    fleet_list = file["devices"]
                return fleet_list

            except FileNotFoundError: #error msg
                return self.status.error(status="error", msg="Fleet file could not be found in " + path_to_list)
        else:
            return self.status.error(status="error", msg="No such directory /data/config/fleets - create or reflash using ente")
"""
