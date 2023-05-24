#!/usr/bin/env python3
#This script is part of the DT Architecture Library for dt-commons

import docker
import os
import time
import requests

from multiprocessing import Process, Manager
from dt_archapi_utils.arch_message import ApiMessage, JobLog

'''
    THIS SCRIPT TAKES CARE OF SENDING AND RECEIVING HTTP REQUESTS USING THE
    REQUESTS LIB FROM PYTHON. THE RECEIVED (RAW) MESSAGES ARE STACKED AND SENT
    BACK TO THE MultiArchAPIClient LIB FOR FURTHER PROCESSING.
'''

class MultiApiWorker:
    def __init__(self, fleet=None, port="8083"):
        self.fleet = fleet
        self.port = port

        #Initialize imported classes
        self.status = ApiMessage()

        #Initialize imported classes - default from single worker
        self.manager = Manager()
        self.log = self.manager.dict()
        self.process = None


    def http_get_request(self, device=None, endpoint=None):
        #Can be generalized by specifying a fleet + for loop, instead of device
        #Now chosen as such to avoid piling up any msg here before sending to multi_arch_client
        #Not tested on performance/delay for large fleets

        #Create request url and request object
        url = 'http://' + str(device) + '.local:' + str(self.port) + '/device' + endpoint
        r = requests.get(url)
        if int(r.status_code) != int(200):
            self.status.msg["status"] = "error"
            self.status.msg["message"] = "Bad request for " + str(device) + "with error code " + str(r.status_code)
            self.status.msg["data"] = {}
            #error = "Bad request for " + str(device)
            return self.status.msg

        try:
            #Save reponse
            response = r.json()
            return response
        except ValueError: #error msg
            self.status.msg["status"] = "error"
            self.status.msg["message"] = "Data cannot be JSON decoded for " + str(device)
            self.status.msg["data"] = {}
            #error = "Data cannot be JSON decoded for " + str(device)
            return self.status.msg


    def http_post_request(self, endpoint=None):
        return None






"""
    def set_config(self, mod_config):
        if self.process is None or not self.process.is_alive():
            #Initialize process of stopping and launching correct modules
            self.process = Process(target=self.set_config_proc, args=(mod_config, self.log,))
            #Start self.process on this object
            self.process.start()
            #Pass process id as soon as started for monitoring
            status_id = {"job id": self.process.pid}
            return status_id
        else:
            #Still in another process
            #Perhaps an idea to overrule the current process? Overcomplicating?
            return "busy"


    def pull_image(self, image_url):
        if self.process is None or not self.process.is_alive():
            self.process = Process(target=self.pull_image_proc, args=(image_url, self.log,))
            self.process.start()
            return {"job id": self.process.pid}
        else:
            #Still in another process
            return "busy"


    def stop_containers(self):
        if self.process is None or not self.process.is_alive():
            self.process = Process(target=self.stop_containers_proc, args=(self.log,))
            self.process.start()
            status_id = {"jobid": self.process.pid}
            return status_id
        else:
            #Still in another process
            return "busy"


    def clear_log(self):
        self.log.clear()
        c = ConfigMessage()
        return c.msg


######################################################################################
#SUBPROCESSES: used as input for Process() objects, have no real output

    def set_config_proc(self, mod_config, log):
        #Get process id from OS & set logging service
        pid = os.getpid()
        progress = JobLog(pid)
        progress.record("stopping all running containers")
        log[pid] = progress.log

        #Stop all running containers with dt-architecture-api prefix
        if self.stop_containers_proc(log) == True:
            progress.record("Succeeded in stopping all containers")
            #Read out module info from self.configuration_info
            if "modules" in mod_config:
                modules = mod_config["modules"]
                total_number = len(modules)
                counter = 0
                #Go through each required module
                for m in modules:
                    progress.record("Starting " + m)
                    #Read out appended mod_config info in self.configuration_info
                    if "configuration" in modules[m]:
                        try:
                            config = modules[m]["configuration"]
                            #Remove listed container if running before launching again, irrespective of image name
                            try:
                                c = self.client.containers.get(config["name"])
                                progress.record("Found existing container: " + config["name"] + ":" + c.status)
                                log[pid] = progress.log
                                clog = c.remove()
                            except docker.errors.NotFound as oops:
                                pass
                            #Launch container (config["name"] equals to container_name in module file)
                            progress.record("Starting new container " + config["name"])
                            log[pid] = progress.log
                            container = self.client.containers.run(detach=True, **config)
                            progress.record("Started " + m)
                            progress.record("status: " + str(container.status) + " ==>" + str(container.logs()))
                            log[pid] = progress.log
                        except docker.errors.APIError as error:
                             progress.record("Docker error when starting " + m + " : " + str(error) + "\n" + str(config))
                             log[pid] = progress.log
                    else:
                        progress.record("No configuration for module " + m + " : " + " skipping")
                    #Update for-loop
                    counter = counter + 1
                    progress.update_progress((counter/total_number ) * 100)
                    log[pid] = progress.log
                #All modules checked
                progress.complete()
                log[pid] = progress.log

            else:
                progress.error("No modules provided in configuration provided")
                log[pid] = progress.log
        else:
            progress.error("Failed to stop all running containers")
            log[pid] = progress.log


    def stop_containers_proc(self, log):
        #Get process id from OS, set up logging service
        pid = os.getpid()
        progress = JobLog(pid)
        log[pid] = progress.log
        #Define imagename from dt-architecture-api build
        img_prefix = "duckietown/dt-architecture-api"
        #Get list of running containers from docker_client
        clist = self.client.containers.list()

        for c in clist:
            #Get image attribute of Container class
            image = c.attrs['Config']['Image']
            #If there is a container running not started by arch-api
            if not image.startswith(img_prefix):
                #Stop container
                try:
                    progress.record("stopping " + str(image))
                    c.stop()
                    progress.record("stopped " + str(image))
                    log[pid] = progress.log
                    time.sleep(30)
                except docker.errors.APIError as error:
                    #Log APIError
                    progress.error(str(error))
                    log[pid] = progress.log
                    return False
        #Log completed process id
        progress.complete()
        log[pid] = progress.log
        return True


    def pull_image_proc(self, url, log):
        #Get process id from OS, set up logging service
        pid = os.getpid()
        progress = JobLog(pid)
        log[pid] = progress.log
        #Read input
        if ":" in url:
            image_url, image_tag = url.split(":",1)
        else:
            image_url = url
            image_tag = "latest"
        try:
            progress.record("pulling " + image_url + ":" + image_tag)
            self.client.images.pull(image_url, tag=image_tag )
            progress.complete()
            log[pid] = progress.log
        except docker.errors.APIError as error:
            #Log APIError
            progress.error(str(error))
            log[pid] = progress.log

#########################################################################################
"""
