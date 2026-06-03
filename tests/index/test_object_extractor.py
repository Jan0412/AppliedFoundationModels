"""Tests for src.index.preprocess.ObjectExtractor."""
from __future__ import annotations

import warnings

import numpy as np
import pytest
from PIL import Image

from src.index.preprocess import ObjectExtractor

from .conftest import IMG_H, IMG_W, _centered_mask, make_sam_vit_mock


# ---------------------------------------------------------------------------
# Construction & validation
# ---------------------------------------------------------------------------


def test_init_stores_background(mock_sam_vit_model):
    ext = ObjectExtractor(sam=mock_sam_vit_model, background="white")
    assert ext.background == "white"


def test_init_stores_margin(mock_sam_vit_model):
    ext = ObjectExtractor(sam=mock_sam_vit_model, margin=0.2)
    assert ext.margin == pytest.approx(0.2)


def test_init_stores_output_size(mock_sam_vit_model):
    ext = ObjectExtractor(sam=mock_sam_vit_model, output_size=128)
    assert ext.output_size == 128


def test_init_stores_top_n(mock_sam_vit_model):
    ext = ObjectExtractor(sam=mock_sam_vit_model, top_n=3)
    assert ext.top_n == 3


def test_init_top_n_default_is_1(mock_sam_vit_model):
    ext = ObjectExtractor(sam=mock_sam_vit_model)
    assert ext.top_n == 1


def test_init_stores_noise_params(mock_sam_vit_model):
    ext = ObjectExtractor(sam=mock_sam_vit_model, background="noise", noise_mean=64.0, noise_std=10.0)
    assert ext.noise_mean == pytest.approx(64.0)
    assert ext.noise_std == pytest.approx(10.0)


def test_init_invalid_background_raises(mock_sam_vit_model):
    with pytest.raises(ValueError, match="background"):
        ObjectExtractor(sam=mock_sam_vit_model, background="transparent")


# ---------------------------------------------------------------------------
# from_config
# ---------------------------------------------------------------------------


def test_from_config_reads_background(tmp_preprocess_config):
    ext = ObjectExtractor.from_config(tmp_preprocess_config)
    assert ext.background == "black"


def test_from_config_reads_margin(tmp_preprocess_config):
    ext = ObjectExtractor.from_config(tmp_preprocess_config)
    assert ext.margin == pytest.approx(0.05)


def test_from_config_reads_output_size(tmp_preprocess_config):
    ext = ObjectExtractor.from_config(tmp_preprocess_config)
    assert ext.output_size == 64


def test_from_config_reads_top_n(tmp_preprocess_config):
    ext = ObjectExtractor.from_config(tmp_preprocess_config)
    assert ext.top_n == 2


def test_from_config_reads_noise_mean(tmp_preprocess_config):
    ext = ObjectExtractor.from_config(tmp_preprocess_config)
    assert ext.noise_mean == pytest.approx(100.0)


def test_from_config_reads_noise_std(tmp_preprocess_config):
    ext = ObjectExtractor.from_config(tmp_preprocess_config)
    assert ext.noise_std == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# extract — single image
# ---------------------------------------------------------------------------


def test_extract_single_returns_list(extractor, fake_pil_image):
    result = extractor.extract(fake_pil_image)
    assert isinstance(result, list)


def test_extract_single_list_length_equals_top_n(extractor, fake_pil_image):
    """With top_n=1 and one valid mask, returns a one-element list."""
    result = extractor.extract(fake_pil_image)
    assert len(result) == 1


def test_extract_single_element_is_pil_image(extractor, fake_pil_image):
    result = extractor.extract(fake_pil_image)
    assert isinstance(result[0], Image.Image)


def test_extract_output_is_correct_size(extractor, fake_pil_image):
    result = extractor.extract(fake_pil_image)
    assert result[0].size == (224, 224)


def test_extract_output_is_rgb(extractor, fake_pil_image):
    result = extractor.extract(fake_pil_image)
    assert result[0].mode == "RGB"


def test_extract_respects_output_size_setting(mock_sam_vit_model, fake_pil_image):
    ext = ObjectExtractor(sam=mock_sam_vit_model, background="gray", output_size=64)
    result = ext.extract(fake_pil_image)
    assert result[0].size == (64, 64)


def test_extract_calls_sam_with_rgb_image(extractor, mock_sam_vit_model):
    """extract converts to RGB before passing to SAM."""
    img = Image.new("RGBA", (32, 32), color=(100, 150, 200, 128))
    extractor.extract(img)
    called_with = mock_sam_vit_model.invoke.call_args.args[0]
    assert called_with.mode == "RGB"


# ---------------------------------------------------------------------------
# extract — top_n > 1
# ---------------------------------------------------------------------------


def test_extract_top_n_2_returns_2_crops(fake_pil_image):
    """With top_n=2 and 3 valid masks, extract returns 2 crops."""
    m1 = _centered_mask()
    m2 = np.zeros((IMG_H, IMG_W), dtype=bool)
    m2[:4, :4] = True
    m3 = np.zeros((IMG_H, IMG_W), dtype=bool)
    m3[-4:, -4:] = True
    sam = make_sam_vit_mock(
        masks=[m1, m2, m3],
        scores=[0.9, 0.7, 0.6],
        boxes=[
            [IMG_W // 4, IMG_H // 4, 3 * IMG_W // 4, 3 * IMG_H // 4],
            [0, 0, 4, 4],
            [IMG_W - 4, IMG_H - 4, IMG_W, IMG_H],
        ],
    )
    ext = ObjectExtractor(sam=sam, background="gray", top_n=2, output_size=64)
    crops = ext.extract(fake_pil_image)
    assert len(crops) == 2
    for crop in crops:
        assert isinstance(crop, Image.Image)
        assert crop.size == (64, 64)


def test_extract_top_n_clamped_to_available_masks(fake_pil_image):
    """top_n=5 with only 1 mask returns a 1-element list, not 5."""
    sam = make_sam_vit_mock()  # single mask
    ext = ObjectExtractor(sam=sam, background="gray", top_n=5, output_size=64)
    crops = ext.extract(fake_pil_image)
    assert len(crops) == 1


def test_extract_top_n_crops_all_have_correct_size(fake_pil_image):
    m1 = _centered_mask()
    m2 = np.zeros((IMG_H, IMG_W), dtype=bool)
    m2[:4, :4] = True
    sam = make_sam_vit_mock(
        masks=[m1, m2],
        scores=[0.9, 0.7],
        boxes=[[IMG_W // 4, IMG_H // 4, 3 * IMG_W // 4, 3 * IMG_H // 4], [0, 0, 4, 4]],
    )
    ext = ObjectExtractor(sam=sam, background="gray", top_n=2, output_size=64)
    for crop in ext.extract(fake_pil_image):
        assert crop.size == (64, 64)


# ---------------------------------------------------------------------------
# extract — list input
# ---------------------------------------------------------------------------


def test_extract_list_returns_list(extractor, fake_pil_image):
    result = extractor.extract([fake_pil_image, fake_pil_image])
    assert isinstance(result, list)
    assert len(result) == 2


def test_extract_list_each_element_is_list(extractor, fake_pil_image):
    results = extractor.extract([fake_pil_image, fake_pil_image])
    for crops in results:
        assert isinstance(crops, list)


def test_extract_list_each_inner_list_has_pil_images(extractor, fake_pil_image):
    results = extractor.extract([fake_pil_image, fake_pil_image])
    for crops in results:
        for crop in crops:
            assert isinstance(crop, Image.Image)


def test_extract_list_each_crop_has_correct_size(extractor, fake_pil_image):
    results = extractor.extract([fake_pil_image, fake_pil_image])
    for crops in results:
        for crop in crops:
            assert crop.size == (224, 224)


def test_extract_calls_sam_once_per_image(extractor, mock_sam_vit_model, fake_pil_image):
    extractor.extract([fake_pil_image, fake_pil_image, fake_pil_image])
    assert mock_sam_vit_model.invoke.call_count == 3


# ---------------------------------------------------------------------------
# extract — type error
# ---------------------------------------------------------------------------


def test_extract_wrong_type_raises_typeerror(extractor):
    with pytest.raises(TypeError, match="PIL.Image.Image"):
        extractor.extract("path/to/image.png")


def test_extract_dict_raises_typeerror(extractor):
    with pytest.raises(TypeError):
        extractor.extract({"image": Image.new("RGB", (32, 32))})


# ---------------------------------------------------------------------------
# extract — no-mask fallback
# ---------------------------------------------------------------------------


def test_extract_no_masks_returns_resized_original(fake_pil_image):
    """When SAM finds no masks, extract returns the resized original image."""
    sam = make_sam_vit_mock(masks=[], scores=[], boxes=[])
    ext = ObjectExtractor(sam=sam, background="gray", output_size=64)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = ext.extract(fake_pil_image)
    assert len(result) == 1
    assert result[0].size == (64, 64)
    assert any("no masks" in str(warning.message).lower() for warning in w)


def test_extract_empty_mask_falls_back(fake_pil_image):
    """When the selected mask is all-False, extract returns the resized original."""
    empty = np.zeros((IMG_H, IMG_W), dtype=bool)
    sam = make_sam_vit_mock(masks=[empty], scores=[0.9], boxes=[[0, 0, IMG_W, IMG_H]])
    ext = ObjectExtractor(sam=sam, background="gray", output_size=64)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        result = ext.extract(fake_pil_image)
    assert len(result) == 1
    assert result[0].size == (64, 64)


# ---------------------------------------------------------------------------
# _apply_background — all four modes
# ---------------------------------------------------------------------------


def _make_simple_case():
    """Return (arr, mask) where foreground is top-left 2×2 of a 4×4 image."""
    arr = np.full((4, 4, 3), 200, dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=bool)
    mask[:2, :2] = True  # top-left foreground
    return arr, mask


def test_apply_background_gray_fills_background(mock_sam_vit_model):
    ext = ObjectExtractor(sam=mock_sam_vit_model, background="gray")
    arr, mask = _make_simple_case()
    result = ext._apply_background(arr, mask)
    bg = ~mask
    assert (result[bg] == np.array([128, 128, 128])).all()


def test_apply_background_gray_preserves_foreground(mock_sam_vit_model):
    ext = ObjectExtractor(sam=mock_sam_vit_model, background="gray")
    arr, mask = _make_simple_case()
    result = ext._apply_background(arr, mask)
    assert (result[mask] == np.array([200, 200, 200])).all()


def test_apply_background_black_fills_background(mock_sam_vit_model):
    ext = ObjectExtractor(sam=mock_sam_vit_model, background="black")
    arr, mask = _make_simple_case()
    result = ext._apply_background(arr, mask)
    assert (result[~mask] == np.array([0, 0, 0])).all()


def test_apply_background_black_preserves_foreground(mock_sam_vit_model):
    ext = ObjectExtractor(sam=mock_sam_vit_model, background="black")
    arr, mask = _make_simple_case()
    result = ext._apply_background(arr, mask)
    assert (result[mask] == np.array([200, 200, 200])).all()


def test_apply_background_white_fills_background(mock_sam_vit_model):
    ext = ObjectExtractor(sam=mock_sam_vit_model, background="white")
    arr, mask = _make_simple_case()
    result = ext._apply_background(arr, mask)
    assert (result[~mask] == np.array([255, 255, 255])).all()


def test_apply_background_white_preserves_foreground(mock_sam_vit_model):
    ext = ObjectExtractor(sam=mock_sam_vit_model, background="white")
    arr, mask = _make_simple_case()
    result = ext._apply_background(arr, mask)
    assert (result[mask] == np.array([200, 200, 200])).all()


def test_apply_background_noise_preserves_foreground(mock_sam_vit_model):
    ext = ObjectExtractor(sam=mock_sam_vit_model, background="noise", noise_mean=128.0, noise_std=5.0)
    arr, mask = _make_simple_case()
    result = ext._apply_background(arr, mask)
    assert (result[mask] == np.array([200, 200, 200])).all()


def test_apply_background_noise_stays_in_valid_range(mock_sam_vit_model):
    ext = ObjectExtractor(sam=mock_sam_vit_model, background="noise", noise_mean=128.0, noise_std=10.0)
    arr, mask = _make_simple_case()
    result = ext._apply_background(arr, mask)
    bg_pixels = result[~mask]
    assert bg_pixels.min() >= 0
    assert bg_pixels.max() <= 255


def test_apply_background_noise_differs_from_solid_fill(mock_sam_vit_model):
    """Noise background should not produce a uniform fill."""
    ext = ObjectExtractor(
        sam=mock_sam_vit_model, background="noise",
        noise_mean=128.0, noise_std=30.0,
    )
    arr = np.full((64, 64, 3), 200, dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=bool)
    mask[16:48, 16:48] = True
    result = ext._apply_background(arr, mask)
    bg_pixels = result[~mask].astype(float)
    assert bg_pixels.std() > 0.0


# ---------------------------------------------------------------------------
# _select_top_n_masks — selection logic
# ---------------------------------------------------------------------------


def test_select_top_n_masks_picks_single_mask(mock_sam_vit_model, fake_pil_image):
    """With only one mask, that mask is always selected."""
    ext = ObjectExtractor(sam=mock_sam_vit_model, background="gray")
    mask = _centered_mask()
    indices = ext._select_top_n_masks([mask], fake_pil_image)
    assert indices == [0]


def test_select_top_n_masks_prefers_central_over_peripheral(fake_pil_image):
    """Selection follows area × centrality, not raw area alone."""
    sam = make_sam_vit_mock()
    ext = ObjectExtractor(sam=sam, background="gray")

    peripheral = np.zeros((IMG_H, IMG_W), dtype=bool)
    peripheral[:, :4] = True  # left strip — large area, far from centre

    central = np.zeros((IMG_H, IMG_W), dtype=bool)
    cx, cy = IMG_W // 2, IMG_H // 2
    central[cy - 2:cy + 2, cx - 2:cx + 2] = True  # tiny centre patch

    h, w = IMG_H, IMG_W
    total = float(h * w)
    cx_f, cy_f = w / 2.0, h / 2.0
    max_dist = np.sqrt(cx_f ** 2 + cy_f ** 2)

    def score(m):
        ys, xs = np.where(m)
        area = m.sum() / total
        mcx, mcy = xs.mean(), ys.mean()
        dist = np.sqrt((mcx - cx_f) ** 2 + (mcy - cy_f) ** 2)
        centrality = 1.0 - dist / max_dist
        return area * centrality

    expected_best = 0 if score(peripheral) > score(central) else 1
    indices = ext._select_top_n_masks([peripheral, central], fake_pil_image)
    assert indices[0] == expected_best


def test_select_top_n_masks_large_central_wins(fake_pil_image):
    """A large, central mask wins over a small, peripheral one."""
    sam = make_sam_vit_mock()
    ext = ObjectExtractor(sam=sam, background="gray")

    small_edge = np.zeros((IMG_H, IMG_W), dtype=bool)
    small_edge[0:2, 0:2] = True

    large_centre = _centered_mask()

    indices = ext._select_top_n_masks([small_edge, large_centre], fake_pil_image)
    assert indices[0] == 1


def test_select_top_n_masks_skips_empty_masks(fake_pil_image):
    """All-False masks are ignored in scoring without raising."""
    sam = make_sam_vit_mock()
    ext = ObjectExtractor(sam=sam, background="gray")

    empty = np.zeros((IMG_H, IMG_W), dtype=bool)
    good = _centered_mask()

    indices = ext._select_top_n_masks([empty, good], fake_pil_image)
    assert indices[0] == 1


def test_select_top_n_masks_all_empty_returns_zero(fake_pil_image):
    """When all masks are empty, index 0 is returned without error."""
    sam = make_sam_vit_mock()
    ext = ObjectExtractor(sam=sam, background="gray")
    empty = np.zeros((IMG_H, IMG_W), dtype=bool)
    indices = ext._select_top_n_masks([empty, empty], fake_pil_image)
    assert indices[0] == 0


def test_select_top_n_masks_returns_n_indices(fake_pil_image):
    """With top_n=2 and 2 non-empty masks, returns 2 indices."""
    sam = make_sam_vit_mock()
    ext = ObjectExtractor(sam=sam, background="gray", top_n=2)

    m1 = _centered_mask()
    m2 = np.zeros((IMG_H, IMG_W), dtype=bool)
    m2[:4, :4] = True

    indices = ext._select_top_n_masks([m1, m2], fake_pil_image)
    assert len(indices) == 2
    assert len(set(indices)) == 2  # no duplicates


def test_select_top_n_masks_sorted_best_first(fake_pil_image):
    """Returned indices are in descending score order (best first)."""
    sam = make_sam_vit_mock()
    ext = ObjectExtractor(sam=sam, background="gray", top_n=2)

    large_centre = _centered_mask()        # high score
    small_edge = np.zeros((IMG_H, IMG_W), dtype=bool)
    small_edge[0:2, 0:2] = True            # low score

    indices = ext._select_top_n_masks([small_edge, large_centre], fake_pil_image)
    assert indices[0] == 1   # large_centre is best
    assert indices[1] == 0


def test_select_top_n_masks_clamped_to_available(fake_pil_image):
    """top_n=5 with only 1 non-empty mask returns a 1-element list."""
    sam = make_sam_vit_mock()
    ext = ObjectExtractor(sam=sam, background="gray", top_n=5)
    indices = ext._select_top_n_masks([_centered_mask()], fake_pil_image)
    assert len(indices) == 1


# ---------------------------------------------------------------------------
# _make_canvas
# ---------------------------------------------------------------------------


def test_make_canvas_gray_value(mock_sam_vit_model):
    ext = ObjectExtractor(sam=mock_sam_vit_model, background="gray")
    canvas = ext._make_canvas(8, 8)
    assert (canvas == 128).all()


def test_make_canvas_black_value(mock_sam_vit_model):
    ext = ObjectExtractor(sam=mock_sam_vit_model, background="black")
    canvas = ext._make_canvas(8, 8)
    assert (canvas == 0).all()


def test_make_canvas_white_value(mock_sam_vit_model):
    ext = ObjectExtractor(sam=mock_sam_vit_model, background="white")
    canvas = ext._make_canvas(8, 8)
    assert (canvas == 255).all()


def test_make_canvas_noise_shape(mock_sam_vit_model):
    ext = ObjectExtractor(sam=mock_sam_vit_model, background="noise")
    canvas = ext._make_canvas(16, 24)
    assert canvas.shape == (16, 24, 3)
    assert canvas.dtype == np.uint8


def test_make_canvas_noise_in_range(mock_sam_vit_model):
    ext = ObjectExtractor(sam=mock_sam_vit_model, background="noise")
    canvas = ext._make_canvas(32, 32)
    assert canvas.min() >= 0
    assert canvas.max() <= 255


# ---------------------------------------------------------------------------
# Square canvas (aspect-ratio centering)
# ---------------------------------------------------------------------------


def test_extract_rect_image_output_is_square(fake_pil_image_rect):
    """A non-square input image is still returned as a square output."""
    mask = np.zeros((32, 48), dtype=bool)
    mask[8:24, 12:36] = True
    sam = make_sam_vit_mock(masks=[mask], scores=[0.9], boxes=[[12, 8, 36, 24]], img_h=32, img_w=48)
    ext = ObjectExtractor(sam=sam, background="gray", output_size=64)
    result = ext.extract(fake_pil_image_rect)
    assert result[0].size == (64, 64)


# ---------------------------------------------------------------------------
# Standalone (no Runnable inheritance)
# ---------------------------------------------------------------------------


def test_is_not_langchain_runnable(mock_sam_vit_model):
    """ObjectExtractor is a plain class, not a LangChain Runnable."""
    from langchain_core.runnables import Runnable

    ext = ObjectExtractor(sam=mock_sam_vit_model, background="gray")
    assert not isinstance(ext, Runnable)
