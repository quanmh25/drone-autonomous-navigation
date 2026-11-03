"""Verify saved maps preserve occupancy and frozen coordinates for A*."""

import math

from drone_vision_control.astar_navigation import astar, inflated_grid
from drone_vision_control.astar_navigation import load_snapshot
from drone_vision_control.module_rgbd.map_snapshot import save_snapshot
from nav_msgs.msg import OccupancyGrid
import numpy as np
import pytest
import yaml


def message():
    msg = OccupancyGrid()
    msg.header.frame_id = 'map'
    msg.info.width, msg.info.height = 4, 3
    msg.info.resolution = .1
    msg.info.origin.position.x, msg.info.origin.position.y = 1., 2.
    msg.info.origin.orientation.w = 1.
    msg.data = [-1, 0, 30, 100, 0, 0, 0, 0, 100, 0, 0, 0]
    return msg


def test_snapshot_round_trip_preserves_exact_occupancy(tmp_path):
    msg = message()
    path = save_snapshot(msg, tmp_path / 'arena.npz')
    restored = load_snapshot(path)
    assert list(restored.data) == list(msg.data)
    assert restored.header.frame_id == 'odom'
    assert restored.info.width == 4 and restored.info.height == 3
    assert restored.info.resolution == pytest.approx(.1)
    assert (tmp_path / 'arena.pgm').read_bytes().startswith(b'P5\n4 3\n255\n')
    metadata = yaml.safe_load((tmp_path / 'arena.yaml').read_text())
    assert metadata['frame_id'] == 'odom'
    assert metadata['image'] == 'arena.pgm'


def test_freeze_composes_translation_and_rotation(tmp_path):
    path = save_snapshot(message(), tmp_path / 'arena.npz', (3., 4., math.pi / 2))
    grid = load_snapshot(path)
    assert grid.info.origin.position.x == pytest.approx(1.)
    assert grid.info.origin.position.y == pytest.approx(5.)
    assert grid.info.origin.orientation.z == pytest.approx(math.sqrt(.5))


def test_unknown_map_does_not_commit_snapshot(tmp_path):
    msg = message()
    msg.data = [-1] * 12
    with pytest.raises(ValueError):
        save_snapshot(msg, tmp_path / 'arena.npz')
    assert not (tmp_path / 'arena.npz').exists()


def test_saved_grid_produces_same_astar_route(tmp_path):
    msg = message()
    original = np.asarray(msg.data).reshape(3, 4)
    saved = load_snapshot(save_snapshot(msg, tmp_path / 'arena.npz'))
    restored = np.asarray(saved.data).reshape(3, 4)
    before = astar(inflated_grid(original, .1, 0.), (0, 1), (3, 2))
    after = astar(inflated_grid(restored, .1, 0.), (0, 1), (3, 2))
    assert before == after and before


def test_tilted_map_projects_and_preserves_occupied_cells(tmp_path):
    from drone_vision_control.module_rgbd.map_snapshot import project_tilted_grid
    from geometry_msgs.msg import Transform

    transform = Transform()
    transform.rotation.y = math.sin(.2)
    transform.rotation.w = math.cos(.2)
    transform.translation.x = 3.
    projected = project_tilted_grid(message(), transform)
    assert projected.header.frame_id == 'odom'
    assert any(cell == 100 for cell in projected.data)
    assert any(cell == -1 for cell in projected.data)
    saved = load_snapshot(save_snapshot(projected, tmp_path / 'tilted.npz'))
    assert list(saved.data) == list(projected.data)


def test_vertical_grid_rejected_instead_of_corrupting_map():
    from drone_vision_control.module_rgbd.map_snapshot import project_tilted_grid
    from geometry_msgs.msg import Transform

    transform = Transform()
    transform.rotation.y = transform.rotation.w = math.sqrt(.5)
    with pytest.raises(ValueError, match='vertical'):
        project_tilted_grid(message(), transform)


@pytest.mark.parametrize('controller', ['astar_navigation', 'module_rgbd.map_snapshot'])
def test_source_config_found_with_copied_install(tmp_path, monkeypatch, controller):
    from importlib import import_module

    default_map_directory = import_module(
        f'drone_vision_control.{controller}').default_map_directory

    package = tmp_path / 'src' / 'drone_vision_control'
    (package / 'config').mkdir(parents=True)
    (package / 'package.xml').write_text('<package/>')
    share = tmp_path / 'install' / 'drone_vision_control' / 'share' / 'drone_vision_control'
    (share / 'config').mkdir(parents=True)
    (share / 'config' / 'README.md').write_text('regular copied file')
    monkeypatch.chdir(tmp_path / 'install')
    assert default_map_directory(share) == str(package / 'config')
