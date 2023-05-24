#!/usr/bin/env python3

import time

class BatteryHistory:

    def __init__(self):
        self._history = []

    def add(self, point):
        if len(self._history) == 0 or point['percentage'] != self._history[-1][1]['percentage']:
            self._history.append((time.time(), point))

    def get(self):
        last_t = None
        cumulative = 0
        history = []
        now = time.time()
        for t, d in self._history:
            cumulative += 0 if last_t is None else (t - last_t)
            history.append({
                'time': {
                    'absolute': int(t),
                    'elapsed': int(now - t),
                    'cumulative': int(cumulative)
                },
                'data': d
            })
            last_t = t
        return
