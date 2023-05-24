import json
import time
import zeroconf
import socket
import netifaces
from threading import Thread, Semaphore

from dt_class_utils import DTProcess

DT_SERVICE_TYPE = '_duckietown._tcp.local.'
DT_SERVICE_NAME = lambda name: 'DT::{name}::{hostname}.{type}'.format(
    name=name, hostname=socket.gethostname(), type=DT_SERVICE_TYPE
)
HEARTBEAT_HZ = 0.5
PASSIVELY_REPUBLISH_EVERY_SECS = 60.0
CHECK_FOR_INTERFACES_EVERY_SECS = 10.0


class DTService:

    def __init__(self, name, port=0, payload=None, paused=False):
        if DTProcess.get_instance() is None:
            print('ERROR: You are trying to create an object of type DTService before '
                  'an object of type DTProcess. DTService objects inherit the lifecycle '
                  'of the singleton instance of DTProcess. Create a DTProcess first.')
            exit(1)
        self._app = DTProcess.get_instance()
        self._zc = zeroconf.Zeroconf()
        self._name = name
        self._port = port
        self._payload = json.dumps((payload if payload is not None else dict())).encode()
        self._do_work = True
        self._last_worked = 0
        self._last_checked_ifaces = 0
        self._worker = Thread(target=self._work)
        self._active = not paused
        self._is_shutdown = False
        self._published_once = False
        self._last_published_IPs = []
        self._network_semaphore = Semaphore(1)
        DTProcess.get_instance().register_shutdown_callback(self.shutdown)
        # start worker
        self._worker.start()

    def _work(self):
        while True:
            if self._do_work:
                with self._network_semaphore:
                    if self._is_shutdown:
                        return
                    # ---
                    srv = self._service_info()
                    if self._active:
                        # register or update
                        try:
                            self._zc.update_service(srv)
                            self._published_once = True
                            self._last_published_IPs = srv.addresses
                        except KeyError:
                            # updating failed because of KeyError, try registering first
                            try:
                                self._zc.register_service(srv)
                                self._published_once = True
                                self._last_published_IPs = srv.addresses
                            except (zeroconf.NonUniqueNameException, BaseException):
                                pass
                        except BaseException:
                            pass
                    else:
                        # unregister
                        try:
                            if self._published_once:
                                self._published_once = False
                                self._zc.unregister_service(srv)
                        except (KeyError, BaseException):
                            pass
                self._do_work = False
                self._last_worked = time.time()
            # passive update
            if time.time() - self._last_worked > PASSIVELY_REPUBLISH_EVERY_SECS:
                self._do_work = True
            # new ifaces
            if time.time() - self._last_checked_ifaces > CHECK_FOR_INTERFACES_EVERY_SECS:
                if self._has_new_ipv4_addresses():
                    self._app.logger.debug(
                        'Service[{}]: Network configuration changed'.format(self._name)
                    )
                    self._do_work = True
                self._last_checked_ifaces = time.time()
            # sleep
            time.sleep(1.0 / HEARTBEAT_HZ)

    def update(self, payload=None):
        # update payload if given
        if payload is not None:
            self._payload = json.dumps(payload).encode()
        # work
        self._do_work = True

    def republish_now(self):
        self._do_work = True

    def resume(self):
        if not self._active:
            self._app.logger.debug('Service[{}]: RESUMED!'.format(self._name))
        # ---
        self._active = True
        self.republish_now()

    def pause(self):
        if self._active:
            self._app.logger.debug('Service[{}]: PAUSED!'.format(self._name))
        # ---
        self._active = False
        self.republish_now()

    def yes(self):
        return self.resume()

    def no(self):
        return self.pause()

    def shutdown(self):
        self.pause()
        self._is_shutdown = True
        # unregister everything
        with self._network_semaphore:
            srv = self._service_info()
            try:
                if self._published_once:
                    self._zc.unregister_service(srv)
            except (KeyError, BaseException):
                pass

    def _service_info(self):
        return zeroconf.ServiceInfo(
            type_=DT_SERVICE_TYPE,
            name=DT_SERVICE_NAME(self._name),
            addresses=self._get_all_ipv4_addresses(),
            port=self._port,
            properties=b' ' + self._payload
        )

    def _has_new_ipv4_addresses(self):
        ipv4s_now = set(DTService._get_all_ipv4_addresses())
        ipv4s_last = set(self._last_published_IPs)
        return \
            len(ipv4s_now) != len(ipv4s_last) or \
            len(ipv4s_now.intersection(ipv4s_last)) != len(ipv4s_now)

    @staticmethod
    def _get_all_ipv4_addresses():
        ip_list = []
        for iface in netifaces.interfaces():
            addresses = netifaces.ifaddresses(iface)
            if netifaces.AF_INET in addresses:
                for link in addresses[netifaces.AF_INET]:
                    ip_list.append(DTService._encode_ipv4(link['addr']))
        return ip_list

    @staticmethod
    def _encode_ipv4(ip4):
        return socket.inet_pton(socket.AF_INET, ip4)
