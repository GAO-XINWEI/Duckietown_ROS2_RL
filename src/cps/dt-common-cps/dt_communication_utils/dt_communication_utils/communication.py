import io
import json
from abc import abstractmethod

import copy
import time
import socket
import inspect
import logging
import threading
from hashlib import sha256
from ipaddress import IPv4Address
from typing import Callable, Union, Optional, Any
from dataclasses import dataclass

import lcm
from genpy import Message as GenericROSMessage

from .dt_communication_msg_t import dt_communication_msg_t

logging.basicConfig()

HOSTNAME = socket.gethostname()
ANYBODY = "*"
ANYBODY_BUT_ME = f"~{HOSTNAME}"


@dataclass
class DTCommunicationMessageHeader(object):
    """
    Models the header of a Communication Group Message.

    Parameters

    - timestamp:      (:obj:`int`): the timestamp the message was generated in microseconds
    - origin:         (:obj:`str`): the hostname of the origin machine
    - destination:    (:obj:`str`): the hostname of the destination machine
    - txt:            (:obj:`str`): extra data encoded as JSON attached to the message
    """
    timestamp: Optional[int]
    origin: str
    destination: Optional[str]
    txt: Optional[str]

    def i_sent_this(self):
        return self.origin == HOSTNAME


class _TypedCommunicationGroup(object):

    def __init__(self, msg_type: GenericROSMessage):
        # check msg_type
        if not inspect.isclass(msg_type):
            raise ValueError(f"Field `msg_type` expected to be of type `class`, "
                             f"got {msg_type.__class__.__name__} instead.")
        # ---
        self.MsgClass = msg_type

    @property
    @abstractmethod
    def logger(self) -> logging.Logger:
        """
        Internal logger.

        :return: Internal logger.
        :rtype:  logging.Logger

        :meta private:
        """
        pass

    def encode(self, msg: Any) -> Optional[bytes]:
        """
        Encodes a ROS message before it is encapsulated into the LCM message.

        :param msg:     (:obj:`Any`):  Message to encode.
        :return:        Encoded message.
        :rtype:         :obj:`bytes`

        :meta private:
        """
        # make sure the message is in the right type
        if not isinstance(msg, self.MsgClass):
            self.logger.warning(f"Expected message of type `{self.MsgClass.__name__}`, "
                                 f"got `{msg.__class__.__name__}` instead.")
            return None
        # ---
        buff = io.BytesIO()
        msg.serialize(buff)
        return buff.getvalue()

    def decode(self, data: bytes, metadata: dict) -> Optional[GenericROSMessage]:
        """
        Decodes a payload right after it comes out of the LCM message.

        :param data:        (:obj:`bytes`):  Message to payload.
        :param metadata:    (:obj:`dict`):  Message metadata.
        :return:            Decoded message.
        :rtype:             :obj:`GenericROSMessage`

        :meta private:
        """
        # make sure the content type matches
        if metadata['msg_type'] != self.MsgClass.__name__:
            self.logger.warning(f"Expected message of type `{self.MsgClass.__name__}`, "
                                 f"got `{metadata['msg_type']}` instead.")
            return None
        # decode
        msg = self.MsgClass()
        msg.deserialize(data)
        return msg


class DTRawCommunicationGroup(object):
    """
    Raw Communication Group allowing processes to exchange `bytes` over the network.

    Base class for :py:class:`DTCommunicationGroup`, it can be used when it is necessary
    to exchange raw bytes rather than ROS messages.

    It is highly suggested to try and use :py:class:`DTCommunicationGroup` whenever possible,
    exchanging raw bytes is usually a recipe for disaster.

    Args:
        name        (:obj:`str`): the name of the group
        ttl         (:obj:`int`): (Time to live) the number of hops the message can do throughout
                    the network before it is discarded.
                    See `Time to live (Wikipedia) <https://en.wikipedia.org/wiki/Time_to_live>`_.
        loglevel    (:obj:`int`): Logger's level of verbosity

    """

    IP_NETWORK = "239.255.0.0/20"
    DEFAULT_PORT = 7667
    DEFAULT_CHANNEL = "/__default__"
    LCM_HEARTBEAT_HZ = 1

    def __init__(self, name: str, ttl: int = 1, loglevel: int = logging.WARNING):
        self._name = name
        self._ttl = ttl
        self._id = self._get_group_id()
        self._url = self._get_url(self.DEFAULT_PORT)
        self._is_shutdown = False
        self._logger = logging.getLogger(f'CommGroup[#{self._id}]')
        self._logger.setLevel(loglevel)
        self._publishers = set()
        self._subscribers = set()
        self._metadata = {}
        # create LCM handler
        self._logger.info(f'Creating LCM handler on URL: `{self._url}`')
        self._lcm = lcm.LCM(self._url)
        self._mailman = threading.Thread(target=self._spin)
        self._mailman.start()

    @property
    def id(self) -> int:
        """
        Unique group's ID, automatically computed from the group's name.

        :return: Unique group's ID.
        :rtype:  int

        :meta private:
        """
        return self._id

    @property
    def url(self) -> str:
        """
        The underlying UDPm LCM url used by this group.

        :return: Group's LCM url.
        :rtype:  str
        """
        return self._url

    @property
    def name(self) -> str:
        """
        Unique group's name.

        :return: Unique group's name.
        :rtype:  str
        """
        return self._name

    @property
    def ttl(self) -> int:
        """
        Time to live.

        :return: Time to live.
        :rtype:  int
        """
        return self._ttl

    @property
    def handler(self) -> lcm.LCM:
        """
        Underlying LCM handler object.

        :return: Underlying LCM handler object.
        :rtype:  lcm.LCM

        :meta private:
        """
        return self._lcm

    @property
    def is_shutdown(self) -> bool:
        """
        Wether the group is shutdown.

        :return: Wether the group is shutdown.
        :rtype:  bool
        """
        return self._is_shutdown

    @property
    def logger(self) -> logging.Logger:
        """
        Internal logger.

        :return: Internal logger.
        :rtype:  logging.Logger

        :meta private:
        """
        return self._logger

    @property
    def metadata(self) -> dict:
        """
        Metadata to append to each message leaving this group handler.

        :return: Metadata.
        :rtype:  dict

        :meta private:
        """
        return self._metadata

    def Subgroup(self, name: str, loglevel: int = None) -> '_DTRawCommunicationSubGroup':
        """
        Creates a Communication Subgroup from this group.

        :param name:        (:obj:`str`): Name of the subgroup (unique within this group).
        :param loglevel:    (:obj:`int`): Logger's level of verbosity.
        :return: :obj:`_DTRawCommunicationSubGroup`
        """
        if loglevel is None:
            loglevel = self._logger.level
        return _DTRawCommunicationSubGroup(self, name, loglevel)

    def Publisher(self) -> 'DTCommunicationPublisher':
        """
        Creates a Publisher object on this group.

        :return: A new Publisher.
        :rtype:  :obj:`DTCommunicationPublisher`
        """
        pub = DTCommunicationPublisher(self, self.DEFAULT_CHANNEL)
        self.add_publisher(pub)
        return pub

    def Subscriber(self, callback: Callable) -> 'DTCommunicationSubscriber':
        """
        Creates a Subscriber object on this group.

        :param callback:    (:obj:`Callable`):  Callback function used to process incoming
                                                messages. Such function should expect two
                                                arguments, `payload` (:obj:`bytes`) and
                                                `header` (:obj:`DTCommunicationMessageHeader`:),
                                                in this order.
        :return: A new Subscriber.
        :rtype:  :obj:`DTCommunicationPublisher`
        """
        sub = DTCommunicationSubscriber(self, self.DEFAULT_CHANNEL, callback)
        self.add_subscriber(sub)
        return sub

    def add_publisher(self, publisher: 'DTCommunicationPublisher'):
        """
        Adds a publisher to the list of publishers attached to this group.

        :param publisher:    (:obj:`DTCommunicationPublisher`):  Publisher to add.

        :meta private:
        """
        self._publishers.add(publisher)

    def add_subscriber(self, subscriber: 'DTCommunicationSubscriber'):
        """
        Adds a subscriber to the list of subscribers attached to this group.

        :param subscriber:    (:obj:`DTCommunicationSubscriber`):  Subscriber to add.

        :meta private:
        """
        self._subscribers.add(subscriber)

    def remove_publisher(self, publisher: 'DTCommunicationPublisher'):
        """
        Removes a publisher from the list of publishers attached to this group.

        :param publisher:    (:obj:`DTCommunicationPublisher`):  Publisher to remove.

        :meta private:
        """
        self._publishers.remove(publisher)

    def remove_subscriber(self, subscriber: 'DTCommunicationSubscriber'):
        """
        Removes a subscriber from the list of subscribers attached to this group.

        :param subscriber:    (:obj:`DTCommunicationSubscriber`):  Subscriber to remove.

        :meta private:
        """
        self._subscribers.remove(subscriber)

    @staticmethod
    def encode(msg: bytes) -> bytes:
        """
        Encodes a message before it is encapsulated into the LCM message.

        :param msg:    (:obj:`bytes`):  Message to encode.
        :return:    Encoded message.
        :rtype:     :obj:`bytes`

        :meta private:
        """
        return msg

    @staticmethod
    def decode(data: bytes, _: dict) -> bytes:
        """
        Decodes a payload right after it comes out of the LCM message.

        :param data:    (:obj:`bytes`):  Message to payload.
        :param _:       (:obj:`dict`):  Message metadata.
        :return:        Decoded message.
        :rtype:         :obj:`bytes`

        :meta private:
        """
        return data

    def shutdown(self):
        """
        Shuts down the group.
        """
        # mark it as shutdown
        self._is_shutdown = True
        # wait for the mailman to return
        self._mailman.join()
        # shutdown all publishers
        for pub in copy.copy(self._publishers):
            pub.shutdown()
        # shutdown all subscribers
        for sub in copy.copy(self._subscribers):
            sub.shutdown()

    def _get_url(self, port: int) -> str:
        """
        Returns the UDPm URL for this group.

        :param port:    (:obj:`int`)    Port number.
        :return:        UDPm URL.
        :rtype:         str
        """
        return f"udpm://{self._get_group_ip()}:{port}?ttl={self.ttl}"

    def _get_group_id(self) -> int:
        """
        Returns the group's ID computed from the group's name.

        :return:        Group's ID.
        :rtype:         int
        """
        _, masklen = self.IP_NETWORK.split('/')
        range_len = 2 ** (32 - int(masklen))
        return int(sha256(self.name.encode('utf-8')).hexdigest(), 16) % range_len

    def _get_group_ip(self) -> IPv4Address:
        """
        Get IPv4 address corresponding to the Group's ID.

        :return:    Group's address.
        :rtype:     IPv4Address
        """
        base_ip, _ = self.IP_NETWORK.split('/')
        base_ip = IPv4Address(base_ip)
        ip_id = self._get_group_id()
        group_ip = base_ip + ip_id
        return group_ip

    def _spin(self):
        """
        Keeps the LCM handler spinning.
        """
        period = (1.0 / self.LCM_HEARTBEAT_HZ) * 1000
        try:
            while not self.is_shutdown:
                self._lcm.handle_timeout(period)
        except KeyboardInterrupt:
            pass


class DTCommunicationGroup(_TypedCommunicationGroup, DTRawCommunicationGroup):
    """
    Communication Group allowing processes to exchange ROS messages over the network.

    Args:
        name        (:obj:`str`): the name of the group
        msg_type    (:obj:`GenericROSMessage`): type of message exchanged in this group.
        ttl         (:obj:`int`): (Time to live) the number of hops the message can do throughout
                    the network before it is discarded.
                    See `Time to live (Wikipedia) <https://en.wikipedia.org/wiki/Time_to_live>`_.
        loglevel    (:obj:`int`): Logger's level of verbosity

    """

    def __init__(self, name: str, msg_type: GenericROSMessage, ttl: int = 1,
                 loglevel: int = logging.WARNING):
        # call super constructors
        _TypedCommunicationGroup.__init__(self, msg_type)
        DTRawCommunicationGroup.__init__(self, name, ttl, loglevel)
        self._metadata = {
            "msg_type": msg_type.__name__
        }

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    def Subgroup(self, name: str, msg_type: GenericROSMessage, loglevel: int = None) \
            -> '_DTCommunicationSubGroup':
        """
        Creates a Communication Subgroup from this group.

        :param name:        (:obj:`str`): Name of the subgroup (unique within this group).
        :param msg_type:    (:obj:`GenericROSMessage`): type of message exchanged in this subgroup.
        :param loglevel:    (:obj:`int`): Logger's level of verbosity.
        :return: :obj:`_DTCommunicationSubGroup`
        """
        if loglevel is None:
            loglevel = self._logger.level
        return _DTCommunicationSubGroup(self, name, msg_type, loglevel)

    def Subscriber(self, callback: Callable) -> 'DTCommunicationSubscriber':
        """
        Creates a Subscriber object on this group.

        :param callback:    (:obj:`Callable`):  Callback function used to process incoming
                                                messages. Such function should expect two
                                                arguments, `message` (:obj:`GenericROSMessage`) and
                                                `header` (:obj:`DTCommunicationMessageHeader`),
                                                in this order.
        :return: A new Subscriber.
        :rtype:  :obj:`DTCommunicationPublisher`
        """
        return super(DTCommunicationGroup, self).Subscriber(callback)


class _DTRawCommunicationSubGroup(object):
    """
    Raw Communication SubGroup.

    Args:
        group       (:obj:`DTRawCommunicationGroup`): underlying communication group.
        name        (:obj:`str`): the name of the group
                    See `Time to live (Wikipedia) <https://en.wikipedia.org/wiki/Time_to_live>`_.
        loglevel    (:obj:`int`): Logger's level of verbosity

    """

    def __init__(self, group: DTRawCommunicationGroup, name: str, loglevel: int = logging.WARNING):
        self._group = group
        self._name = name.strip()
        self._topic = '/' + self._name.strip('/')
        self._is_shutdown = False
        self._logger = logging.getLogger(f'CommSubGroup[#{self._group.id}{self._topic}]')
        self._logger.setLevel(loglevel)
        self._publishers = set()
        self._subscribers = set()

    @property
    def name(self) -> str:
        """
        Unique subgroup's name (within the group).

        :return: Unique subgroup's name (within the group).
        :rtype:  str
        """
        return self._name

    @property
    def handler(self) -> lcm.LCM:
        """
        Underlying LCM handler object.

        :return: Underlying LCM handler object.
        :rtype:  lcm.LCM

        :meta private:
        """
        return self._group.handler

    @property
    def is_shutdown(self) -> bool:
        """
        Wether the subgroup is shutdown.

        :return: Wether the subgroup is shutdown.
        :rtype:  bool
        """
        return self._is_shutdown

    @property
    def logger(self) -> logging.Logger:
        """
        Internal logger.

        :return: Internal logger.
        :rtype:  logging.Logger

        :meta private:
        """
        return self._logger

    @property
    def metadata(self) -> dict:
        """
        Metadata to append to each message leaving this subgroup.

        :return: Metadata.
        :rtype:  dict

        :meta private:
        """
        return self._group.metadata

    def Publisher(self) -> 'DTCommunicationPublisher':
        """
        Creates a Publisher object on this subgroup.

        :return: A new Publisher.
        :rtype:  :obj:`DTCommunicationPublisher`
        """
        pub = DTCommunicationPublisher(self, self._topic)
        self.add_publisher(pub)
        return pub

    def Subscriber(self, callback: Callable) -> 'DTCommunicationSubscriber':
        """
        Creates a Subscriber object on this group.

        :param callback:    (:obj:`Callable`):  Callback function used to process incoming
                                                messages. Such function should expect two
                                                arguments `payload` (:obj:`bytes`) and
                                                `header` (:obj:`DTCommunicationMessageHeader`),
                                                in this order.
        :return: A new Subscriber.
        :rtype:  :obj:`DTCommunicationPublisher`
        """
        sub = DTCommunicationSubscriber(self, self._topic, callback)
        self.add_subscriber(sub)
        return sub

    def add_publisher(self, publisher: 'DTCommunicationPublisher'):
        """
        Adds a publisher to the list of publishers attached to this subgroup.

        :param publisher:    (:obj:`DTCommunicationPublisher`):  Publisher to add.

        :meta private:
        """
        self._publishers.add(publisher)
        self._group.add_publisher(publisher)

    def add_subscriber(self, subscriber: 'DTCommunicationSubscriber'):
        """
        Adds a subscriber to the list of subscribers attached to this subgroup.

        :param subscriber:    (:obj:`DTCommunicationSubscriber`):  Subscriber to add.

        :meta private:
        """
        self._subscribers.add(subscriber)
        self._group.add_subscriber(subscriber)

    def remove_publisher(self, publisher: 'DTCommunicationPublisher'):
        """
        Removes a publisher from the list of publishers attached to this subgroup.

        :param publisher:    (:obj:`DTCommunicationPublisher`):  Publisher to remove.

        :meta private:
        """
        self._publishers.remove(publisher)
        self._group.remove_publisher(publisher)

    def remove_subscriber(self, subscriber: 'DTCommunicationSubscriber'):
        """
        Removes a subscriber from the list of subscribers attached to this subgroup.

        :param subscriber:    (:obj:`DTCommunicationSubscriber`):  Subscriber to remove.

        :meta private:
        """
        self._subscribers.remove(subscriber)
        self._group.remove_subscriber(subscriber)

    def encode(self, msg: bytes) -> bytes:
        """
        Encodes a message before it is encapsulated into the LCM message.

        :param msg:    (:obj:`bytes`):  Message to encode.
        :return:    Encoded message.
        :rtype:     :obj:`bytes`

        :meta private:
        """
        return self._group.encode(msg)

    def decode(self, data: bytes, metadata: dict) -> bytes:
        """
        Decodes a payload right after it comes out of the LCM message.

        :param data:            (:obj:`bytes`):  Message to payload.
        :param metadata:        (:obj:`dict`):  Message metadata.
        :return:                Decoded message.
        :rtype:                 :obj:`bytes`

        :meta private:
        """
        return self._group.decode(data, metadata)

    def shutdown(self):
        """
        Shuts down the subgroup.
        """
        # mark it as shutdown
        self._is_shutdown = True
        # TODO: this has the potential of SegFault-ing the process. It goes like this:
        #       .
        #       There is an LCM handler that is used by two threads in Python.
        #       One is called the "mailman", which is basically taking care of processing the
        #       messages coming in and then the main application thread, which consumes messages
        #       that the mailman puts in a queue.
        #       .
        #       The app thread is the one deciding when the whole process should terminate.
        #       So, the app thread initializes a `shutdown()` sequence and starts detaching
        #       publishers/subscribers from the handler, with the idea being that once the
        #       mailman returns, it never goes out again to deliver messages.
        #       .
        #       On the C side of the handler, though, when the mailman returns with a message,
        #       it tries to look up the corresponding subscriber and its callbacks,
        #       but those were already destroyed by the app thread, hence the SegFault.
        #       .
        #       While this can be solved in a `CommunicationGroup` by waiting for the mailman to
        #       return, it is not as easy in a `SubGroup`, where the mailman is shared across
        #       subgroups.
        #
        # shutdown all publishers
        for pub in copy.copy(self._publishers):
            pub.shutdown()
        # shutdown all subscribers
        for sub in copy.copy(self._subscribers):
            sub.shutdown()


class DTCommunicationPublisher(object):

    def __init__(self, group: Union[DTRawCommunicationGroup, _DTRawCommunicationSubGroup],
                 topic: str):
        """
        (For internal use only)
        Creates a new Publisher for a Group or Subgroup.

        Args:
            group:  (:obj:`Union[DTRawCommunicationGroup, _DTRawCommunicationSubGroup]`):
                    Underlying group/subgroup.
            topic:  (:obj:`str`):   Topic name.

        :meta private:
        """
        self._group = group
        self._topic = topic

    def publish(self, data: Any, destination: str = ANYBODY, txt: str = None):
        """
        Publishes a new message.

        :param data:            (:obj:`Any`):   Message to publish.
        :param destination:     (:obj:`str`):   (Optional) Destination of this message. Used
                                                to send private messages within a group.
        :param txt:             (:obj:`str`)    (Optional) JSON-encoded string of user metadata.

        :raises ValueError:     A given argument is of the wrong type.
        """
        # let the group encode the data first
        data = self._group.encode(data)
        if data is None:
            return
        # check input (data)
        if not isinstance(data, bytes):
            raise ValueError(f'Field `data` must be of type `bytes`, '
                             f'given `{str(type(data))}` instead.')
        # check input (destination)
        if destination is not None and not isinstance(destination, str):
            raise ValueError(f'Field `destination` must be of type `str`, '
                             f'given `{str(type(destination))}` instead.')
        # check input (txt)
        if txt is not None and not isinstance(txt, str):
            raise ValueError(f'Field `txt` must be of type `str`, '
                             f'given `{str(type(txt))}` instead.')
        # create empty message
        msg = dt_communication_msg_t()
        # populate message
        msg.timestamp = time.time_ns() // 1000
        msg.group = self._group.name
        msg.origin = HOSTNAME
        msg.destination = (destination or ANYBODY).strip()
        msg.metadata = json.dumps(self._group.metadata)
        msg.txt = txt or ""
        msg.length = len(data)
        msg.payload = data
        # publish message
        msg = msg.encode()
        self._group.handler.publish(self._topic, msg)

    def shutdown(self):
        """
        Shuts down the publisher.
        """
        self._group.remove_publisher(self)


class DTCommunicationSubscriber(object):

    def __init__(self, group: Union[DTRawCommunicationGroup, _DTRawCommunicationSubGroup],
                 topic: str, callback: Callable):
        """
        (For internal use only)
        Creates a new Subscriber for a Group or Subgroup.

        Args:
            group:  (:obj:`Union[DTRawCommunicationGroup, _DTRawCommunicationSubGroup]`):
                    Underlying group/subgroup.
            topic:  (:obj:`str`):   Topic name.
            callback:  (:obj:`Callable`):   Callback.

        :meta private:
        """
        self._group = group
        self._topic = topic
        self._callback = callback
        self._subscription_handler = \
            self._group.handler.subscribe(self._topic, self.__inner_callback__)
        self._collisions = set()

    def shutdown(self):
        """
        Shuts down the subscriber.
        """
        self._group.handler.unsubscribe(self._subscription_handler)
        self._group.remove_subscriber(self)

    def __inner_callback__(self, _, data):
        msg = None
        try:
            msg = dt_communication_msg_t.decode(data)
        except ValueError:
            pass
        # check if the message was decoded successfully
        if msg is None:
            self._group.logger.warning("Received invalid message. Ignoring it.")
            return
        # make sure there is no group collision here
        if msg.group != self._group.name:
            if msg.group not in self._collisions:
                self._group.logger.warning(
                    f"Collision detected between the groups `{msg.group}` "
                    f"and `{self._group.name}`. If you are the administrator, "
                    f"we suggest you increase the IP address pool dedicate to "
                    f"UDP Multicast.")
                self._collisions.add(msg.group)
            return
        # make sure we are not supposed to receive this message
        if msg.destination == ANYBODY_BUT_ME:
            return
        # make sure we are the intended destination of this message
        if msg.destination != ANYBODY \
                and not msg.destination.startswith('~') \
                and msg.destination != HOSTNAME:
            return
        # parse metadata
        metadata = json.loads(msg.metadata)
        # expose message metadata as a DTCommunicationMessageHeader object
        header = DTCommunicationMessageHeader(
            timestamp=msg.timestamp,
            origin=msg.origin,
            destination=msg.destination,
            txt=msg.txt or None
        )
        # let the group decode the data first
        payload = self._group.decode(msg.payload, metadata)
        if payload is None:
            return
        # call user callback
        self._callback(payload, header)


class _DTCommunicationSubGroup(_TypedCommunicationGroup, _DTRawCommunicationSubGroup):

    def __init__(self, group: DTCommunicationGroup, name: str, msg_type: GenericROSMessage,
                 loglevel: int = logging.WARNING):
        # call super constructors
        _TypedCommunicationGroup.__init__(self, msg_type)
        _DTRawCommunicationSubGroup.__init__(self, group, name, loglevel)
        self._metadata = {
            "msg_type": msg_type.__name__
        }

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    @property
    def metadata(self):
        return self._metadata
