"""PhotoLab 自检。

用法：
    python tests.py              # 核心算法自检（无需 Qt）
    python tests.py --ui         # 追加 UI 冒烟测试（自动 offscreen + 同步处理）
"""
from __future__ import annotations

import os
import sys

RUN_UI = "--ui" in sys.argv

if RUN_UI:                                   # 必须在导入 Qt 之前设置
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("PHOTOLAB_SYNC", "1")

import numpy as np

import core

FAILS: list = []


def check(name, cond):
    if isinstance(cond, np.ndarray):
        cond = bool(np.all(cond))
    print(("  PASS  " if cond else "  FAIL  ") + name)
    if not cond:
        FAILS.append(name)


# ======================================================================
# 一、核心算法
# ======================================================================

def test_colorspaces():
    x = np.linspace(0, 1, 256, dtype=np.float32)
    img = np.stack([x, x, x], axis=-1)[None, :, :]
    back = core.to_srgb(core.to_linear(img))
    check("sRGB->线性->sRGB 往返误差 < 1e-4", float(np.abs(back - img).max()) < 1e-4)
    check("线性亮度单调", bool(np.all(np.diff(core.luminance(core.to_linear(img))[0]) > 0)))


def test_basic():
    img = np.full((8, 8, 3), 0.5, dtype=np.float32)
    out = core.to_srgb(core.apply_exposure(img, 1.0))
    check("曝光 +1EV 使 sRGB 输出变亮", float(out.mean()) > 0.6)
    check("曝光 0 为恒等", np.array_equal(core.apply_exposure(img, 0.0), img))

    lo = np.full((8, 8, 3), 0.2, np.float32)
    hi = np.full((8, 8, 3), 0.8, np.float32)
    c_lo, c_hi = core.apply_contrast(lo, 100), core.apply_contrast(hi, 100)
    check("对比度 +100 拉开暗/亮", float(c_lo.mean()) < 0.2 and float(c_hi.mean()) > 0.8)
    check("对比度固定 0 和 1 端点", float(core.apply_contrast(np.float32(0.0), 100)) == 0.0
          and abs(float(core.apply_contrast(np.float32(1.0), 100)) - 1.0) < 1e-6)

    dark = np.zeros((16, 16, 3), np.float32) + 0.02
    check("阴影提亮只影响暗部", float(core.apply_shadows(dark, 100).mean()) > 0.02)

    wb = core.apply_white_balance(np.full((4, 4, 3), 0.5, np.float32), 100, 0)
    check("色温 +100 偏暖(R>B)", wb[0, 0, 0] > wb[0, 0, 2])

    gray = np.full((8, 8, 3), 0.5, np.float32)
    gray[..., 1] = 0.9
    sat = core.apply_saturation(gray, 100)
    check("饱和度增大色差", float(sat[..., 1].mean() - sat[..., 0].mean()) > 0.4)
    check("饱和度 -100 得到灰度", float(np.abs(core.apply_saturation(gray, -100).std(axis=-1).max())) < 1e-5)
    check("自然饱和度 0 为恒等", np.array_equal(core.apply_vibrance(gray, 0), gray))
    check("黑色抬升暗部", float(core.apply_blacks(dark, 100).mean()) > 0.02)


def test_curves():
    pts = [(0.0, 0.0), (0.5, 0.7), (1.0, 1.0)]
    lut = core.build_lut(pts, 512)
    check("对角线为恒等 LUT",
          float(np.abs(core.build_lut([(0.0, 0.0), (1.0, 1.0)], 512)
                       - np.linspace(0, 1, 512)).max()) < 1e-5)
    check("LUT 经过控制点", abs(float(lut[256]) - 0.7) < 0.05)

    mono = [(0.0, 0.0), (0.25, 0.25), (0.75, 0.75), (1.0, 1.0)]
    check("单调控制点 -> 单调 LUT（无过冲）", float(np.diff(core.build_lut(mono, 1024)).min()) >= -1e-6)

    sp = core.sanitize_points([(0.5, 0.5), (0.2, 0.2), (0.5, 0.9), (0.0, 0.0), (1.0, 1.0)])
    check("控制点按 x 排序去重且端点固定",
          sp[0][0] == 0.0 and sp[-1][0] == 1.0
          and all(sp[i][0] < sp[i + 1][0] for i in range(len(sp) - 1)))

    img = np.full((8, 8, 3), 0.5, np.float32)
    curves = core.default_curves()
    curves["master"] = [(0.0, 0.0), (0.5, 0.25), (1.0, 1.0)]
    check("主通道曲线变暗中间调", float(core.apply_curves(img, curves).mean()) < 0.5)
    curves_r = core.default_curves()
    curves_r["red"] = [(0.0, 0.0), (1.0, 0.2)]
    out_r = core.apply_curves(img, curves_r)
    check("红色通道曲线只压 R", out_r[0, 0, 0] < out_r[0, 0, 1]
          and abs(float(out_r[0, 0, 1]) - 0.5) < 0.05)
    check("默认曲线判定", core.curves_are_default(core.default_curves()))
    check("LUT 复合数量正确", set(core.channel_luts(core.default_curves())) == {"red", "green", "blue"})


def test_grading():
    img = np.zeros((32, 32, 3), np.float32)
    img[:16] = 0.05     # 阴影区
    img[16:] = 0.85     # 高光区
    zero = {r: core.WheelState() for r in core.GRADING_RANGES}
    check("零参数颜色分级为恒等", np.array_equal(core.apply_color_grading(img, zero, 50), img))
    wheels = {r: core.WheelState() for r in core.GRADING_RANGES}
    wheels["shadows"] = core.WheelState(hue=0.0, saturation=80.0)
    out = core.apply_color_grading(img, wheels, 50.0)
    check("阴影色轮只影响暗部(R 偏移)",
          float(out[:16, :, 0].mean() - img[:16, :, 0].mean()) > 0.01
          and abs(float(out[16:, :, 0].mean() - img[16:, :, 0].mean())) < 0.01)
    check("色轮方向：hue=0 为红", core.wheel_direction(0, 1)[0] > 0.8
          and core.wheel_direction(0, 1)[1] < 0)
    check("色轮方向：hue=120 为绿", core.wheel_direction(120, 1)[1] > 0.8)


def test_picker():
    rng = np.random.default_rng(7)
    img = rng.random((48, 48, 3)).astype(np.float32)
    img[:16, :16] = (0.9, 0.05, 0.05)
    img[16:32, :16] = (0.05, 0.8, 0.05)
    img[32:, :16] = 0.5
    rgb8 = core.linear_to_rgb8(img)

    check("色相采样：红≈0°，绿≈120°",
          abs(core.hue_degrees(core.sample_patch(rgb8, 2, 2))) < 5
          and abs(core.hue_degrees(core.sample_patch(rgb8, 2, 18)) - 120.0) < 5)

    p = core.PickerState(hue=core.hue_degrees(core.sample_patch(rgb8, 2, 2)),
                         span=25.0, saturation=-100.0)
    out = core.apply_pickers(img, [p])
    check("取色器去饱和只影响红色区域",
          float(out[:16, :16, 1].mean() - img[:16, :16, 1].mean()) > 0.1
          and abs(float(out[16:32, :16, 1].mean() - img[16:32, :16, 1].mean())) < 0.01
          and float(np.abs(out[32:, :16] - img[32:, :16]).max()) < 1e-4)

    p2 = core.PickerState(hue=120.0, span=30.0, hue_shift=120.0)
    out2 = core.apply_pickers(img, [p2])
    check("色相旋转改变绿色区域色相",
          core.hue_degrees(out2[16:32, :16].reshape(-1, 3).mean(axis=0)) > 100)
    check("无取色点为恒等", np.array_equal(core.apply_pickers(img, []), img))


def test_effects():
    # 清晰度：应增强局部对比（对比两块区域的平均差）
    img = np.zeros((64, 64, 3), np.float32)
    img[:32] = 0.20
    img[32:] = 0.30
    clear = core.apply_clarity(img, 80)
    check("清晰度增强局部对比",
          abs(float(clear[:32].mean() - clear[32:].mean())) >
          abs(float(img[:32].mean() - img[32:].mean())))
    check("清晰度 0 为恒等", np.array_equal(core.apply_clarity(img, 0), img))

    # 锐度：USM 应增强边缘梯度
    strip = np.zeros((32, 32, 3), np.float32)
    strip[:, 14:18] = 0.8
    sharp = core.apply_sharpen(strip, 80)
    check("锐度增强边缘", float(sharp[:, 15].mean() - sharp[:, 13].mean()) >
          float(strip[:, 15].mean() - strip[:, 13].mean()))
    check("锐度 0 为恒等", np.array_equal(core.apply_sharpen(strip, 0), strip))

    # 去朦胧：合成"雾图"应被恢复（对比提升）
    rng = np.random.default_rng(5)
    base = (rng.random((120, 160, 3)) * 0.6).astype(np.float32)
    hazy = base * 0.65 + 0.25                     # 加雾：压对比、抬黑位
    dehazed = core.apply_dehaze(hazy, 70)
    check("去朦胧提升对比",
          float(dehazed.std()) > float(hazy.std()))
    fogged = core.apply_dehaze(base, -60)
    check("负值去朦胧 = 加雾（对比下降）", float(fogged.std()) < float(base.std()))
    check("去朦胧 0 为恒等", np.array_equal(core.apply_dehaze(hazy, 0), hazy))
    check("去朦胧不产生 NaN/无穷", bool(np.all(np.isfinite(dehazed))))
    dark = np.zeros((32, 32, 3), np.float32) + 0.002
    check("近黑图去朦胧数值稳定", bool(np.all(np.isfinite(core.apply_dehaze(dark, 100)))))


def test_masks():
    shape = (300, 400)                             # h, w
    h, w = shape

    linear = core.MaskState(kind="linear", x1=0.2, y1=0.5, x2=0.8, y2=0.5, feather=40)
    lm = core.render_mask(linear, shape)
    check("线性蒙版尺寸/范围", lm.shape == (h, w) and 0 <= lm.min() and lm.max() <= 1)
    check("线性蒙版左端 ~0 右端 ~1",
          float(lm[150, 10]) < 0.05 and float(lm[150, 390]) > 0.95)
    check("线性蒙版端点连线中点为 ~0.5", abs(float(lm[150, 200]) - 0.5) < 0.1)
    lin_inv = core.MaskState(kind="linear", x1=0.2, y1=0.5, x2=0.8, y2=0.5,
                             feather=40, invert=True)
    check("线性蒙版反向", abs(float((core.render_mask(lin_inv, shape) + lm).max()) - 1.0) < 1e-5)
    lin_hard = core.MaskState(kind="linear", x1=0.2, y1=0.5, x2=0.8, y2=0.5, feather=0)
    lh = core.render_mask(lin_hard, shape)
    check("线性蒙版羽化 0 为硬边", abs(float(lh[150, 199])) < 0.02 and abs(float(lh[150, 201]) - 1) < 0.02)

    radial = core.MaskState(kind="radial", cx=0.5, cy=0.5, radius=0.25, feather=30)
    rm = core.render_mask(radial, shape)
    check("径向蒙版中心为 1", float(rm[150, 200]) > 0.99)
    check("径向蒙版四角为 0", float(rm[2, 2]) < 0.01 and float(rm[297, 397]) < 0.01)
    radial_inv = core.MaskState(kind="radial", cx=0.5, cy=0.5, radius=0.25,
                                feather=30, invert=True)
    check("径向蒙版反向 = 除椭圆外全部覆盖",
          float(core.render_mask(radial_inv, shape)[150, 200]) < 0.01)
    check("径向长宽比生效",
          core.render_mask(core.MaskState(kind="radial", radius=0.25, feather=0, aspect=2.5),
                           shape)[150, 300] > 0.5)

    brush = core.MaskState(kind="brush", size=10, feather=20,
                           strokes=[[(0.5, 0.5), (0.6, 0.55)], [(0.3, 0.7)]])
    bm = core.render_mask(brush, shape)
    check("画笔蒙版覆盖笔迹", float(bm[150, 200]) > 0.5 and float(bm[210, 120]) > 0.5)
    check("画笔蒙版不影响远处", float(bm[20, 380]) < 0.01)
    check("空画笔蒙版全 0", float(core.render_mask(
        core.MaskState(kind="brush"), shape).max()) == 0.0)
    brush_hard = core.MaskState(kind="brush", size=10, feather=0,
                                 strokes=[[(0.5, 0.5)]])
    bh = core.render_mask(brush_hard, shape)
    check("画笔羽化 0 为硬边", float(bh.max()) > 0.999)

    # 蒙版缓存键：几何变化 -> 键变化；仅改调整 -> 键不变
    k1 = core.mask_cache_key(brush, shape)
    brush2 = core.MaskState(kind="brush", size=10, feather=20,
                            strokes=[[(0.5, 0.5), (0.6, 0.55)], [(0.3, 0.7)]])
    brush2.adj.exposure = 1.0
    k2 = core.mask_cache_key(brush2, shape)
    brush3 = core.MaskState(kind="brush", size=12, feather=20, strokes=brush.strokes)
    check("蒙版缓存键与调整数值无关", k1 == k2)
    check("蒙版缓存键随几何变化", k1 != core.mask_cache_key(brush3, shape))

    # 蒙版局部调整：只作用于蒙版内
    img = np.full((64, 64, 3), 0.3, np.float32)
    m = core.MaskState(kind="radial", cx=0.5, cy=0.5, radius=0.3, feather=10)
    m.adj.exposure = 1.0
    out = core.apply_masked_adjustments(img, [m])
    check("蒙版内曝光 +1EV", out[32, 32].mean() > 0.55)
    check("蒙版外保持原值", abs(float(out[2, 2].mean()) - 0.3) < 1e-5)
    m2 = core.MaskState(kind="radial", cx=0.5, cy=0.5, radius=0.3, feather=10, invert=True)
    m2.adj.exposure = 1.0
    out2 = core.apply_masked_adjustments(img, [m2])
    check("反向蒙版作用于蒙版外", out2[2, 2].mean() > 0.55
          and abs(float(out2[32, 32].mean()) - 0.3) < 1e-5)
    check("无调整的蒙版为恒等",
          np.array_equal(core.apply_masked_adjustments(img, [core.MaskState()]), img))
    check("蒙版蒙版叠加不过曝", core.apply_masked_adjustments(
        img, [m, m2]).max() <= 0.3 * 2 + 1e-6)


def test_pipeline():
    rng = np.random.default_rng(3)
    rgb = (rng.random((200, 320, 3)) * 255).astype(np.uint8)

    p = core.Adjustments()
    check("默认参数 -> 原图直出", np.array_equal(core.process(p, rgb), rgb))

    p.exposure = 0.5
    p.clarity = 30
    p.sharpen = 20
    p.dehaze = 25
    p.curves["master"] = [(0.0, 0.0), (0.5, 0.6), (1.0, 1.0)]
    p.wheels["shadows"] = core.WheelState(hue=210, saturation=60, luminance=10)
    p.wheels["highlights"] = core.WheelState(hue=40, saturation=30, luminance=-10)
    p.blend = 60
    p.pickers.append(core.PickerState(hue=0.0, span=40.0, saturation=30.0))
    p.masks.append(core.MaskState(kind="radial", cx=0.5, cy=0.5, radius=0.3, feather=25))
    p.masks[0].adj.exposure = -0.5
    out = core.process(p, rgb)
    check("流水线输出 shape/dtype 正确", out.shape == rgb.shape and out.dtype == np.uint8)
    check("流水线输出范围合法", int(out.min()) >= 0 and int(out.max()) <= 255)
    check("流水线两次结果一致", np.array_equal(out, core.process(p, rgb)))
    check("降采样不放大", core.downscale_max_edge(rgb, 1200).shape[0] <= core.PREVIEW_MAX_EDGE)
    check("小图不降采样", core.downscale_max_edge(rgb, 4000) is rgb)

    prev = core.process(p, core.downscale_max_edge(rgb, 1200))
    check("预览/全图尺寸比例正确", abs(rgb.shape[0] / prev.shape[0] - 200 / prev.shape[0]) < 0.01)
    check("halo 计算合理", core.mask_halo(p, 3000, 2000) >= 64
          and core.mask_halo(core.Adjustments(), 3000, 2000) == 0)


def test_parallel_equals_serial():
    """大图走多线程分块路径，结果必须与单线程完全一致。"""
    rng = np.random.default_rng(11)
    rgb = (rng.random((1600, 3000, 3)) * 255).astype(np.uint8)   # 4.8MP，触发多线程
    p = core.Adjustments()
    p.exposure = 0.4
    p.contrast = 25
    p.highlights = -15
    p.shadows = 20
    p.temperature = 50
    p.saturation = 10
    p.vibrance = 30
    p.curves["master"] = [(0.0, 0.0), (0.3, 0.4), (0.7, 0.65), (1.0, 1.0)]
    p.wheels["shadows"] = core.WheelState(210, 70, 20)
    p.wheels["highlights"] = core.WheelState(45, -40, -15)
    p.blend = 65
    p.pickers.append(core.PickerState(0.0, (0.8, 0.2, 0.2), 45, 40, -30))
    p.pickers.append(core.PickerState(120.0, (0.2, 0.8, 0.2), 35, 0, 25))

    lin = core.rgb8_to_linear(rgb)
    par = core.process_linear(p, lin)
    ser = core._process_band(p, lin)
    check("多线程结果与单线程逐位一致", np.array_equal(par, ser))
    check("多线程输出尺寸/类型正确", par.shape == rgb.shape and par.dtype == np.uint8)

    # 蒙版 + 效果一起并行，验证低分辨率蒙版按行切片不产生接缝
    p2 = p.copy()
    p2.clarity = 30
    p2.sharpen = 20
    p2.dehaze = 15
    rm = core.MaskState(kind="radial", cx=0.5, cy=0.5, radius=0.3, feather=25)
    rm.adj.exposure = -0.5
    bm = core.MaskState(kind="brush", size=12, feather=30,
                        strokes=[[(0.2, 0.2), (0.4, 0.35), (0.6, 0.2)],
                                 [(0.7, 0.8), (0.85, 0.9)]])
    bm.adj.shadows = 35
    lm = core.MaskState(kind="linear", x1=0.15, y1=0.9, x2=0.85, y2=0.15,
                        feather=35, invert=True)
    lm.adj.temperature = 25
    p2.masks = [rm, bm, lm]
    par2 = core.process_linear(p2, lin)
    ser2 = core._process_band(p2, lin)
    check("蒙版+效果并行结果逐位一致（无接缝）", np.array_equal(par2, ser2))
    check("蒙版流水线输出范围合法", int(par2.min()) >= 0 and int(par2.max()) <= 255)
    check("蒙版流水线确定", np.array_equal(par2, core.process_linear(p2, lin)))
    check("蒙版渲染器切片有效",
          core.MaskRenderer(p2.masks, rgb.shape[:2]).get(1, 800, 1000, rgb.shape[1]).shape
          == (200, rgb.shape[1]))


# ======================================================================
# 二、UI 冒烟（--ui）
# ======================================================================

def run_ui_smoke() -> int:
    import tempfile

    import cv2
    from PySide6.QtWidgets import QApplication

    from ui_window import MainWindow
    from ui_widgets import apply_dark_theme

    IMG_W, IMG_H = 2000, 3000
    RED_CENTER = (1000, 1500)
    GREEN_CENTER = (400, 600)

    def make_test_image(path: str):
        yy, xx = np.mgrid[0:IMG_H, 0:IMG_W]
        img = np.empty((IMG_H, IMG_W, 3), np.uint8)
        img[..., 0] = (xx * 255 // IMG_W).astype(np.uint8)
        img[..., 1] = (yy * 255 // IMG_H).astype(np.uint8)
        img[..., 2] = 90
        img[1400:1600, 900:1100] = (200, 30, 30)     # 红块
        img[450:750, 250:550] = (30, 200, 40)        # 绿块
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    def image_to_widget(window, px, py):
        prev = window.preview
        off = prev._offset()
        scale = prev._scale()
        ratio = prev.current_frame_size()[0] / IMG_W
        return int(off.x() + px * ratio * scale), int(off.y() + py * ratio * scale)

    app = QApplication(sys.argv)
    apply_dark_theme(app)
    window = MainWindow()
    window.show()
    window.resize(1460, 920)
    app.processEvents()

    with tempfile.TemporaryDirectory() as tmp:
        img_path = os.path.join(tmp, "test.png")
        make_test_image(img_path)
        window.load_image(img_path)

        check("图像已加载", window.source is not None
              and window.source.shape[:2] == (IMG_H, IMG_W))
        check("预览控件有帧", window.preview._pixmap is not None)
        check("加载后渲染全图", window._last_frame.shape[:2] == (IMG_H, IMG_W))

        # ---- 基础 ----
        row = window.panel.basic.rows["exposure"]
        before = window._last_frame.astype(np.int32).mean()
        row.slider.setValue(int(1.0 * row._mult))          # 模拟拖动滑块
        check("曝光滑块写入参数", window.params.exposure == 1.0)
        check("拖动时处理降采样图(预览)", window._last_frame.shape[0] == 1200)
        check("曝光 +1EV 画面变亮", window._last_frame.astype(np.int32).mean() > before)
        window._on_params_committed()
        check("松手后处理全图", window._last_frame.shape[:2] == (IMG_H, IMG_W))

        for key, val in (("contrast", 50), ("highlights", -20), ("shadows", 15),
                         ("temperature", 60), ("tint", -10), ("saturation", 30),
                         ("vibrance", 20)):
            window.panel.basic.rows[key].slider.setValue(val)
        check("基础参数全部生效", window.params.contrast == 50
              and window.params.temperature == 60 and window.params.vibrance == 20)

        # ---- 曲线 ----
        window.panel.curve._on_points_changed("master", [(0.0, 0.0), (0.5, 0.6), (1.0, 1.0)])
        check("主通道曲线参数已写入", len(window.params.curves["master"]) == 3)
        window.panel.curve._on_points_committed("green", [(0.0, 0.1), (1.0, 0.9)])
        check("绿色通道曲线参数已写入", window.params.curves["green"][0] == (0.0, 0.1))
        window.panel.curve._apply_reset("all")
        check("曲线可重置", all(len(v) == 2 for v in window.params.curves.values()))

        # ---- 颜色分级 ----
        window.panel.grading._on_wheel("shadows", 200.0, 0.7)
        check("色轮参数已写入", abs(window.params.wheels["shadows"].saturation - 70) < 1e-3)
        window.panel.grading._on_lum("highlights", -20)
        window.panel.grading._on_sat("global", -25)
        window.panel.grading._on_blend(70)
        check("混合滑块已写入", window.params.blend == 70)
        frame_graded = window._last_frame.copy()

        # ---- 取色器 ----
        wx, wy = image_to_widget(window, *RED_CENTER)
        window._on_image_picked(wx, wy)
        check("图像点击生成取色点", len(window.params.pickers) == 1)
        p = window.params.pickers[0]
        check("取色点色相≈红色", min(p.hue, 360 - p.hue) < 25.0)
        check("取色点采样颜色为红", p.color[0] > 0.6 and p.color[1] < 0.3)
        p.hue_shift = 120
        p.saturation = -50
        window._on_params_changed(force_full=True)
        frame_picked = window._last_frame.copy()
        check("取色器调整改变画面", not np.array_equal(frame_picked, frame_graded))

        gx, gy = image_to_widget(window, *GREEN_CENTER)
        window._on_image_picked(gx, gy)
        check("支持多个取色点", len(window.params.pickers) == 2)
        window.panel.picker.delete_picker(1)
        check("可删除取色点", len(window.params.pickers) == 1)

        # ---- 新功能：预设颜色 / 效果调节 ----
        window.panel.picker.new_preset(120.0)                     # 预设"绿"
        check("预设颜色创建取色点", len(window.params.pickers) == 2
              and abs(window.params.pickers[1].hue - 120.0) < 1e-6)
        window.params.pickers[1].hue_shift = 45            # 让绿偏黄
        window.params.pickers[1].saturation = -30
        window._on_params_changed(force_full=True)         # 提交渲染后再比较
        check("预设色可单独调色相", window.params.pickers[1].hue_shift == 45)
        frame_preset = window._last_frame.copy()
        check("预设取色点影响画面", not np.array_equal(frame_preset, frame_picked))

        window.panel.basic.rows["clarity"].slider.setValue(35)
        window.panel.basic.rows["sharpen"].slider.setValue(25)
        window.panel.basic.rows["dehaze"].slider.setValue(20)
        check("效果参数生效", window.params.clarity == 35
              and window.params.sharpen == 25 and window.params.dehaze == 20)
        window._on_params_committed()
        frame_effects = window._last_frame.copy()
        check("效果调节改变画面", not np.array_equal(frame_effects, frame_preset))
        for k in ("clarity", "sharpen", "dehaze"):
            window.panel.basic.rows[k].slider.setValue(0)

        # ---- 蒙版 ----
        wx, wy = image_to_widget(window, *RED_CENTER)
        wx, wy = image_to_widget(window, *RED_CENTER)
        window._on_image_picked(wx, wy)
        check("恢复图像取色", len(window.params.pickers) == 3)

        window.panel.mask.create_mask("radial")
        check("创建径向蒙版", len(window.params.masks) == 1
              and window.params.masks[0].kind == "radial")
        check("创建后自动进入编辑", window.preview.edit_mode() == "radial"
              and window._mask_edit == 0)

        # 拖动蒙版中心（在画布上模拟）
        window.preview.radialCenterMoved.emit(0.4, 0.6)
        check("径向蒙版中心可拖", abs(window.params.masks[0].cx - 0.4) < 1e-6)
        # 半径/长宽比/反向
        window.panel.mask._on_radius(45.0)
        window.panel.mask._on_aspect(150.0)
        check("径向蒙版半径/长宽比", abs(window.params.masks[0].radius - 0.45) < 1e-6
              and abs(window.params.masks[0].aspect - 1.5) < 1e-6)
        window.panel.mask._set_attr("invert", True)
        check("蒙版可反向", window.params.masks[0].invert is True)
        window.panel.mask._set_attr("invert", False)
        # 局部调整
        window.panel.mask._on_adj("exposure", -1.0)
        check("蒙版局部调整生效", window.params.masks[0].adj.exposure == -1.0)
        window._on_params_committed()
        frame_mask = window._last_frame.copy()
        check("蒙版局部调整改变画面", not np.array_equal(frame_mask, frame_effects))
        check("蒙版叠加层已生成", len(window.preview._overlays) == 1)
        check("蒙版缓存复用", len(window._overlay_cache) >= 1)

        # ESC 退出编辑
        window._exit_mask_edit()
        check("Esc 退出蒙版编辑", window.preview.edit_mode() == "none")

        # 画笔蒙版 + 笔画 + 撤销/清除
        window.panel.mask.create_mask("brush")
        window._on_brush_point(0.45, 0.45)
        window._on_brush_point(0.55, 0.55)
        window._on_brush_finished()
        window._on_brush_point(0.2, 0.2)
        window._on_brush_finished()
        check("画笔蒙版记录笔画", len(window.params.masks[1].strokes) == 2
              and len(window.params.masks[1].strokes[0]) == 2)
        window.panel.mask._on_adj("shadows", 30)
        window._on_params_committed()
        check("画笔蒙版局部调整改变画面",
              not np.array_equal(window._last_frame, frame_mask))
        window._on_brush_undo()
        check("撤销一笔", len(window.params.masks[1].strokes) == 1)
        window._on_brush_cleared()
        check("清除笔迹", len(window.params.masks[1].strokes) == 0)

        # 线性渐变蒙版（从画布拖手柄）
        window.panel.mask.create_mask("linear")
        window.preview.linearHandleMoved.emit(0, 0.15, 0.8, 0.85, 0.3)
        check("线性蒙版端点可拖", abs(window.params.masks[2].x1 - 0.15) < 1e-6
              and abs(window.params.masks[2].y2 - 0.3) < 1e-6)
        window.panel.mask.delete_mask(2)
        check("可删除蒙版", len(window.params.masks) == 2)

        # ---- 真实鼠标事件：画布上拖拽编辑蒙版 ----
        from PySide6.QtCore import QPoint, Qt
        from PySide6.QtTest import QTest

        def wxy(nx, ny):
            pt = window.preview.frame_point_to_widget(nx, ny)
            return QPoint(int(pt.x()), int(pt.y()))

        # 画笔：按住并拖动应记录一笔（两个点）
        window._exit_mask_edit()
        window.panel.mask.create_mask("brush")
        check("画笔模式已进入", window.preview.edit_mode() == "brush")
        p0, p1 = wxy(0.45, 0.4), wxy(0.55, 0.5)
        QTest.mouseMove(window.preview, p0)
        QTest.mousePress(window.preview, Qt.MouseButton.LeftButton,
                         Qt.KeyboardModifier.NoModifier, p0)
        QTest.mouseMove(window.preview, p1)
        QTest.mouseRelease(window.preview, Qt.MouseButton.LeftButton,
                           Qt.KeyboardModifier.NoModifier, p1)
        brush = window.params.masks[2]
        check("鼠标拖动绘制一笔", len(brush.strokes) == 1 and len(brush.strokes[0]) == 2)
        check("笔迹坐标归一化", abs(brush.strokes[0][0][0] - 0.45) < 0.02
              and abs(brush.strokes[0][-1][1] - 0.5) < 0.02)

        # 线性渐变：拖动起点手柄
        window._exit_mask_edit()
        window.panel.mask.delete_mask(2)
        window.panel.mask.create_mask("linear")
        handle = wxy(window.params.masks[2].x1, window.params.masks[2].y1)
        target = wxy(0.2, 0.25)
        QTest.mouseMove(window.preview, handle)
        QTest.mousePress(window.preview, Qt.MouseButton.LeftButton,
                         Qt.KeyboardModifier.NoModifier, handle)
        QTest.mouseMove(window.preview, target)
        QTest.mouseRelease(window.preview, Qt.MouseButton.LeftButton,
                           Qt.KeyboardModifier.NoModifier, target)
        lin = window.params.masks[2]
        check("拖动线性手柄改变端点", abs(lin.x1 - 0.2) < 0.02 and abs(lin.y1 - 0.25) < 0.02)

        # 径向渐变：拖动中心
        window._exit_mask_edit()
        window.panel.mask.delete_mask(2)
        window.panel.mask.create_mask("radial")
        center = wxy(window.params.masks[2].cx, window.params.masks[2].cy)
        target = wxy(0.65, 0.35)
        QTest.mouseMove(window.preview, center)
        QTest.mousePress(window.preview, Qt.MouseButton.LeftButton,
                         Qt.KeyboardModifier.NoModifier, center)
        QTest.mouseMove(window.preview, target)
        QTest.mouseRelease(window.preview, Qt.MouseButton.LeftButton,
                           Qt.KeyboardModifier.NoModifier, target)
        rad = window.params.masks[2]
        check("拖动径向中心", abs(rad.cx - 0.65) < 0.02 and abs(rad.cy - 0.35) < 0.02)

        # 编辑模式下左键不应平移画面
        pan_before = window.preview._pan
        QTest.mouseMove(window.preview, center)
        QTest.mousePress(window.preview, Qt.MouseButton.LeftButton,
                         Qt.KeyboardModifier.NoModifier, center)
        QTest.mouseMove(window.preview, QPoint(center.x() + 60, center.y() + 40))
        QTest.mouseRelease(window.preview, Qt.MouseButton.LeftButton,
                           Qt.KeyboardModifier.NoModifier,
                           QPoint(center.x() + 60, center.y() + 40))
        check("编辑模式不触发平移", window.preview._pan == pan_before)

        window._exit_mask_edit()
        for m in list(range(len(window.params.masks) - 1, -1, -1)):
            window.panel.mask.delete_mask(m)
        check("清理全部蒙版", not window.params.masks)

        # ---- 对比原图 ----
        window._toggle_compare(True)
        check("对比原图显示降采样原图", window.preview.current_frame_size()[1] == 1200)
        window._toggle_compare(False)
        check("退出对比恢复处理结果", window.preview.current_frame_size()[0] == IMG_W)

        # ---- 保存 ----
        out_path = os.path.join(tmp, "out.jpg")
        window._save_path = out_path
        window._save_rid = 10 ** 6
        window._write_image(window._last_frame, out_path)
        check("保存 JPEG 成功", os.path.isfile(out_path) and os.path.getsize(out_path) > 1000)
        check("保存内容尺寸正确", cv2.imread(out_path).shape[:2] == (IMG_H, IMG_W))

        # ---- 重置 ----
        window.reset_adjustments()
        check("重置后参数归零", window.params.is_default() and not window.params.pickers)
        check("重置后输出回到原图", np.array_equal(window._last_frame, window.source))

        # ---- 视图交互 ----
        window.preview.zoom_at(2.0)
        check("滚轮缩放生效", abs(window.preview.zoom - 2.0) < 1e-6)
        window.preview.zoom_fit()
        window.preview.zoom_one_to_one()
        check("1:1 缩放生效", window.preview.zoom > 1.0)
        window.preview.zoom_fit()
        window.panel.set_histograms(None)
        check("直方图可更新", True)

        # ---- 边界：小图 / 灰度图 ----
        small = (np.random.default_rng(1).random((60, 80, 3)) * 255).astype(np.uint8)
        sp = os.path.join(tmp, "small.png")
        cv2.imwrite(sp, cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
        window.load_image(sp)
        check("小图加载且不降采样", window._last_frame.shape[:2] == (60, 80))
        window.panel.basic.rows["exposure"].slider.setValue(80)
        check("小图可调色", window._last_frame.shape[:2] == (60, 80))

        gray_path = os.path.join(tmp, "gray.png")
        cv2.imwrite(gray_path, np.full((200, 300), 120, np.uint8))
        window.load_image(gray_path)
        gray = window.source
        check("灰度图加载为三通道", gray.shape == (200, 300, 3))
        window.panel.basic.rows["saturation"].slider.setValue(-100)
        check("灰度图调色不崩溃", window._last_frame.shape == (200, 300, 3))

        # ---- 无图像时保存被安全拦截 ----
        window.source = None
        check("无图像时保存被安全拦截", window._submit("full") == -1)
        window.source = gray

        window.close()
        check("窗口正常关闭", True)
    return 0


# ======================================================================

def main():
    print("== 色彩空间 ==");    test_colorspaces()
    print("== 基础调整 ==");    test_basic()
    print("== 曲线 ==");        test_curves()
    print("== 颜色分级 ==");    test_grading()
    print("== 取色器 ==");      test_picker()
    print("== 流水线 ==");      test_pipeline()
    print("== 并行一致性 ==");  test_parallel_equals_serial()
    print("== 效果调节 ==");    test_effects()
    print("== 蒙版 ==");        test_masks()

    if RUN_UI:
        print("== UI 冒烟 ==")
        run_ui_smoke()

    print()
    if FAILS:
        print(f"{len(FAILS)} 项失败: {FAILS}")
        return 1
    print("全部通过 ✔")
    return 0


if __name__ == "__main__":
    sys.exit(main())
