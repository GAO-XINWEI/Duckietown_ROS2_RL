#!/usr/bin/env python3

from builtin_interfaces.msg import Time


class BuiltinTimeWrapper(Time):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def from_ros_time(cls, time: Time):
        return cls(sec=time.sec, nanosec=time.nanosec)

    def __add__(self, other: 'BuiltinTimeWrapper'):
        time = BuiltinTimeWrapper()
        time.sec = self.sec + other.sec
        time.nanosec = self.nanosec + other.nanosec
        return time

    def __sub__(self, other: 'BuiltinTimeWrapper'):
        time = BuiltinTimeWrapper()
        time.sec = self.sec - other.sec
        time.nanosec = self.nanosec - other.nanosec
        return time

    def __truediv__(self, factor):
        time = BuiltinTimeWrapper()
        time.sec = int(self.sec / factor)
        time.nanosec = int(self.nanosec / factor)
        return time

    def __gt__(self, other: 'BuiltinTimeWrapper'):
        return (self.sec > other.sec) or \
            (self.sec == other.sec and self.nanosec > other.nanosec)
    
    def __ge__(self, other: 'BuiltinTimeWrapper'):
        return self.__gt__(other) or self.__eq__(other)

    def __lt__(self, other: 'BuiltinTimeWrapper'):
        return (self.sec < other.sec) or \
             (self.sec == other.sec and self.nanosec < other.nanosec)

    def __le__(self, other: 'BuiltinTimeWrapper'):
        return self.__lt__(other) or self.__eq__(other)


def test():
    BTW = BuiltinTimeWrapper
    # Equality
    assert(BTW(sec=123, nanosec=581293) == BTW(sec=123, nanosec=581293))
    assert(not BTW(sec=121, nanosec=581293) == BTW(sec=123, nanosec=581293))
    assert(not BTW(sec=123, nanosec=581293) == BTW(sec=123, nanosec=581292))

    # Addition
    assert(BTW(sec=1, nanosec=2) + BTW(sec=3, nanosec=4)
        == BTW(sec=4, nanosec=6))

    # Subtraction
    assert(BTW(sec=10, nanosec=101) - BTW(sec=3, nanosec=4)
        == BTW(sec=7, nanosec=97))

    # Division
    assert(BTW(sec=10000, nanosec=36000) / 2 == BTW(sec=5000, nanosec=18000))
    assert(BTW(sec=10000, nanosec=50000) / 5 == BTW(sec=2000, nanosec=10000))

    # Greater
    assert(BTW(sec=124, nanosec=581293) > BTW(sec=123, nanosec=581293))
    assert(BTW(sec=123, nanosec=581294) > BTW(sec=123, nanosec=581293))
    assert(not BTW(sec=123, nanosec=581293) > BTW(sec=123, nanosec=581293))
    assert(not BTW(sec=122, nanosec=581293) > BTW(sec=123, nanosec=581293))
    assert(not BTW(sec=123, nanosec=581292) > BTW(sec=123, nanosec=581293))

    # Greater or equal
    assert(BTW(sec=123, nanosec=581293) >= BTW(sec=123, nanosec=581293))
    assert(BTW(sec=124, nanosec=581293) >= BTW(sec=123, nanosec=581293))
    assert(BTW(sec=123, nanosec=581294) >= BTW(sec=123, nanosec=581293))
    assert(not BTW(sec=122, nanosec=581293) >= BTW(sec=123, nanosec=581293))
    assert(not BTW(sec=123, nanosec=581292) >= BTW(sec=123, nanosec=581293))

    # Less than
    assert(BTW(sec=122, nanosec=581293) < BTW(sec=123, nanosec=581293))
    assert(BTW(sec=123, nanosec=581292) < BTW(sec=123, nanosec=581293))
    assert(not BTW(sec=123, nanosec=581293) < BTW(sec=123, nanosec=581293))
    assert(not BTW(sec=123, nanosec=581293) < BTW(sec=123, nanosec=581293))
    assert(not BTW(sec=123, nanosec=581294) < BTW(sec=123, nanosec=581293))

    # Less than or equal
    assert(BTW(sec=123, nanosec=581293) <= BTW(sec=123, nanosec=581293))
    assert(BTW(sec=122, nanosec=581293) <= BTW(sec=123, nanosec=581293))
    assert(BTW(sec=123, nanosec=581292) <= BTW(sec=123, nanosec=581293))
    assert(not BTW(sec=124, nanosec=581293) <= BTW(sec=123, nanosec=581293))
    assert(not BTW(sec=123, nanosec=581294) <= BTW(sec=123, nanosec=581293))


if __name__ == "__main__":
    test()
