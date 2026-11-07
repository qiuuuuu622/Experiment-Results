import pytest

from vllm_dsv41_vision.metadata_cache import GeometryCache


def test_geometry_reuse_is_order_device_and_dtype_specific():
    cache = GeometryCache(2)
    args = dict(grids=[(4, 6), (3, 9)], device='cuda:0', dtype='bf16',
                rope_dim=32, rope_theta=10000, downsample_ratio=3,
                build=object)
    first = cache.get(**args)
    assert cache.get(**args) is first
    reversed_grids = cache.get(**{**args, 'grids': [(3, 9), (4, 6)]})
    assert reversed_grids is not first
    on_other_device = cache.get(**{**args, 'device': 'cuda:1'})
    assert on_other_device is not first
    assert len(cache.entries) == 2
    # LRU eviction must rebuild the original metadata, not return stale device data.
    new_first = cache.get(**args)
    assert new_first is not first
    assert cache.get(**{**args, 'dtype': 'fp32'}) is not new_first
    cache.clear()
    assert not cache.entries


def test_builder_failure_never_caches_partial_metadata():
    cache = GeometryCache()
    args = dict(grids=[(4, 6)], device='cuda:0', dtype='bf16',
                rope_dim=32, rope_theta=10000, downsample_ratio=3)
    def fail():
        raise RuntimeError('preparation failed')
    with pytest.raises(RuntimeError):
        cache.get(**args, build=fail)
    assert len(cache.entries) == 0
    with pytest.raises(ValueError):
        cache.get(**{**args, 'grids': [(0, 6)]}, build=object)
