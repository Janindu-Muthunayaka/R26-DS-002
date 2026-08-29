"""EXIF orientation must be applied on arrival.

CameraX writes rotation into EXIF and does not rotate pixels, so something has
to apply the tag. Which library does that is version-dependent — PIL does not
on Image.open, OpenCV does in imread and (on current versions) in imdecode.
Depending on a default is how this bug returns on someone else's machine.

So these tests do not assert how OpenCV behaves. They assert the OUTCOME: that
our loaders return the upright image, matching an independently rotated
reference array, on any OpenCV. The reference is built with np.rot90 rather
than with PIL, so the test cannot pass merely by agreeing with the code under
test.
"""
import io

import cv2
import numpy as np
import pytest
from PIL import Image

from core.imaging import imread_upright, imdecode_upright

W, H = 300, 150          # deliberately non-square, so a rotation is visible
ORIENT_ROTATE_90 = 6     # EXIF value meaning "rotate 90 CW to display"


def _tagged_jpeg(tmp_path, orientation=None):
    """A landscape image with an asymmetric mark, optionally EXIF-tagged."""
    arr = np.full((H, W, 3), 240, np.uint8)
    arr[0:20, 0:120] = 20          # dark bar along the top-left edge
    im = Image.fromarray(arr)
    p = tmp_path / f'o{orientation}.jpg'
    if orientation is None:
        im.save(p, quality=95)
    else:
        exif = Image.Exif()
        exif[274] = orientation    # 274 = Orientation
        im.save(p, exif=exif, quality=95)
    return p


def _reference_upright():
    """The expected result, built independently of PIL and of the code under
    test. Orientation 6 means 'rotate 90 clockwise to display', which is
    np.rot90 with k=-1."""
    arr = np.full((H, W, 3), 240, np.uint8)
    arr[0:20, 0:120] = 20
    return np.rot90(arr, k=-1)


def test_matches_an_independent_reference(tmp_path):
    """Catches wrong direction, mirroring, and double rotation at once."""
    p = _tagged_jpeg(tmp_path, ORIENT_ROTATE_90)
    got = imread_upright(p)
    ref = _reference_upright()
    assert got.shape == ref.shape, (
        f'{got.shape} != {ref.shape} — wrong rotation, or applied twice')
    # JPEG is lossy, so compare coarsely rather than exactly
    diff = np.abs(got.astype(int) - ref[:, :, ::-1].astype(int)).mean()
    assert diff < 12, f'mean abs pixel difference {diff:.1f} — content moved'


def test_opencv_behaviour_is_recorded_not_relied_on(tmp_path):
    """Not an assertion about OpenCV — a note that our result must not depend
    on it. Both paths must agree with each other whatever cv2 does."""
    p = _tagged_jpeg(tmp_path, ORIENT_ROTATE_90)
    assert imread_upright(p).shape == imdecode_upright(p.read_bytes()).shape


def test_imread_upright_applies_exif(tmp_path):
    p = _tagged_jpeg(tmp_path, ORIENT_ROTATE_90)
    up = imread_upright(p)
    assert up is not None
    assert up.shape[:2] == (W, H), (
        f'expected the image rotated to {(W, H)}, got {up.shape[:2]} — '
        'EXIF orientation was not applied')


def test_imdecode_upright_applies_exif(tmp_path):
    p = _tagged_jpeg(tmp_path, ORIENT_ROTATE_90)
    data = p.read_bytes()
    up = imdecode_upright(data)
    assert up is not None
    assert up.shape[:2] == (W, H)
    # and the same result as the path-based loader
    assert np.array_equal(up, imread_upright(p))


def test_untagged_image_is_unchanged(tmp_path):
    """No orientation tag must mean no rotation — not a default guess."""
    p = _tagged_jpeg(tmp_path, None)
    assert imread_upright(p).shape[:2] == (H, W)
    assert imdecode_upright(p.read_bytes()).shape[:2] == (H, W)


def test_the_dark_bar_moves_where_it_should(tmp_path):
    """Shape alone could be satisfied by a transpose that mirrors the image.
    Orientation 6 sends the top-left bar down the right-hand edge."""
    p = _tagged_jpeg(tmp_path, ORIENT_ROTATE_90)
    up = cv2.cvtColor(imread_upright(p), cv2.COLOR_BGR2GRAY)
    h, w = up.shape
    right_strip = up[:, w - 20:]
    left_strip = up[:, :20]
    assert right_strip.mean() < left_strip.mean(), (
        'the mark did not land on the right edge — the rotation direction '
        'is wrong, or the image was mirrored')


def test_garbage_bytes_do_not_raise(tmp_path):
    assert imdecode_upright(b'not an image at all') is None
    assert imread_upright(tmp_path / 'does_not_exist.jpg') is None
