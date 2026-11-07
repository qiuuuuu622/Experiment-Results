"""Bounded, instance-local cache for image geometry, never image features.

Prepared metadata is read-only. Pixel tensors must always come from the current
call. Order, dimensions, device and dtype are part of the key. No cross-process
or implicit global state is used.
"""

from collections import OrderedDict


class GeometryCache:
    def __init__(self, capacity=16):
        if capacity < 1:
            raise ValueError('capacity must be positive')
        self.capacity = capacity
        self.entries = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, *, grids, device, dtype, rope_dim, rope_theta,
            downsample_ratio, build):
        grids = tuple((int(h), int(w)) for h, w in grids)
        if any(h <= 0 or w <= 0 for h, w in grids):
            raise ValueError('grid dimensions must be positive')
        key = (grids, str(device), str(dtype), int(rope_dim),
               float(rope_theta), int(downsample_ratio))
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            return self.entries[key]
        self.misses += 1
        value = build()
        self.entries[key] = value
        if len(self.entries) > self.capacity:
            self.entries.popitem(last=False)
        return value

    def clear(self):
        self.entries.clear()
