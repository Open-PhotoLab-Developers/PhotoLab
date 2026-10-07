"""PhotoLab UI：调整面板与主窗口。

内容分区：
  1. 基础面板    2. 曲线面板   3. 颜色分级面板   4. 取色器面板
  4b. 蒙版面板   5. 右侧总面板  6. 主窗口（工具栏/状态栏/调度/取色/蒙版/保存）
"""
from __future__ import annotations

import os
import time
import traceback

import cv2
import numpy as np

import core
from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (QButtonGroup, QFileDialog, QFrame, QGridLayout,
                               QGroupBox, QHBoxLayout, QLabel, QMainWindow,
                               QMessageBox, QPushButton, QScrollArea,
                               QSizePolicy, QSplitter, QToolBar, QVBoxLayout,
                               QWidget)

from core import (PREVIEW_MAX_EDGE, Adjustments, PickerState, downscale_max_edge,
                  hue_degrees, linear_histograms, mask_cache_key, render_mask,
                  sample_patch)
from ui_widgets import (CHANNEL_LABELS, CollapsibleSection, ColorWheel,
                        CurveEditor, FULL, PREVIEW, PreviewWidget,
                        ProcessorThread, SliderRow, mask_to_qimage)

IMAGE_FILTER = "图像文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;所有文件 (*.*)"

RANGE_LABELS = {"shadows": "阴影", "midtones": "中间调",
                "highlights": "高光", "global": "全局"}


# ======================================================================
# 1. 基础面板
# ======================================================================

class BasicPanel(QWidget):
    changed = Signal()
    committed = Signal()

    ROWS = (
        ("基础", (
            ("曝光", "exposure", -5.0, 5.0, 2),
            ("对比度", "contrast", -100, 100, 0),
            ("高光", "highlights", -100, 100, 0),
            ("阴影", "shadows", -100, 100, 0),
            ("白色", "whites", -100, 100, 0),
            ("黑色", "blacks", -100, 100, 0),
        )),
        ("颜色", (
            ("色温", "temperature", -100, 100, 0),
            ("色调", "tint", -100, 100, 0),
            ("饱和度", "saturation", -100, 100, 0),
            ("自然饱和度", "vibrance", -100, 100, 0),
        )),
        ("效果", (
            ("清晰度", "clarity", -100, 100, 0),
            ("锐度", "sharpen", 0, 100, 0),
            ("去朦胧", "dehaze", -100, 100, 0),
        )),
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self._params: Adjustments | None = None
        self._blocked = False
        self.rows: dict = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(8)
        for group_title, rows in self.ROWS:
            box = QGroupBox(group_title)
            lay = QVBoxLayout(box)
            lay.setSpacing(6)
            for label, name, lo, hi, decimals in rows:
                suffix = " EV" if name == "exposure" else ""
                row = SliderRow(label, lo, hi, 0.0, decimals=decimals,
                                suffix=suffix, default_value=0.0)
                row.valueChanged.connect(lambda v, n=name: self._on_changed(n, v))
                row.sliderReleased.connect(self.committed)
                self.rows[name] = row
                lay.addWidget(row)
            outer.addWidget(box)
        outer.addStretch(1)

    def set_params(self, params: Adjustments):
        self._params = params
        self.sync()

    def sync(self):
        if self._params is None:
            return
        self._blocked = True
        for name, row in self.rows.items():
            row.set_value(getattr(self._params, name))
        self._blocked = False

    def _on_changed(self, name: str, value: float):
        if self._blocked or self._params is None:
            return
        setattr(self._params, name, value)
        self.changed.emit()


# ======================================================================
# 2. 曲线面板
# ======================================================================

class CurvePanel(QWidget):
    changed = Signal()
    committed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._params: Adjustments | None = None
        self._blocked = False

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(8)

        ch_row = QHBoxLayout()
        self._channel_buttons = {}
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        for ch in ("master", "red", "green", "blue"):
            btn = QPushButton(CHANNEL_LABELS[ch])
            btn.setCheckable(True)
            btn.setChecked(ch == "master")
            btn.clicked.connect(lambda _=False, c=ch: self._select_channel(c))
            self._channel_buttons[ch] = btn
            self._group.addButton(btn)
            ch_row.addWidget(btn)
        reset_btn = QPushButton("重置")
        reset_btn.setToolTip("重置当前通道曲线")
        reset_btn.clicked.connect(self._reset_current)
        ch_row.addWidget(reset_btn)
        reset_all_btn = QPushButton("全部重置")
        reset_all_btn.setToolTip("重置全部通道曲线")
        reset_all_btn.clicked.connect(self._reset_all)
        ch_row.addWidget(reset_all_btn)
        lay.addLayout(ch_row)

        self.editor = CurveEditor()
        self.editor.pointsChanged.connect(self._on_points_changed)
        self.editor.pointsCommitted.connect(self._on_points_committed)
        lay.addWidget(self.editor, 1)

        hint = QLabel("双击曲线添加控制点，右键删除，拖动调整")
        hint.setObjectName("dim")
        hint.setWordWrap(True)
        lay.addWidget(hint)

    def set_params(self, params: Adjustments):
        self._params = params
        self.sync()

    def sync(self):
        if self._params is None:
            return
        self._blocked = True
        from core import default_curves
        for ch in ("master", "red", "green", "blue"):
            self.editor.set_points(ch, self._params.curves.get(ch)
                                   or default_curves()[ch])
        self._blocked = False

    def _select_channel(self, channel: str):
        self.editor.set_channel(channel)
        for ch, btn in self._channel_buttons.items():
            btn.setChecked(ch == channel)

    def set_histograms(self, hist):
        self.editor.set_histograms(hist)

    def _on_points_changed(self, channel: str, points):
        if self._params is None or self._blocked:
            return
        self._params.curves[channel] = [tuple(p) for p in points]
        self.changed.emit()

    def _on_points_committed(self, channel: str, points):
        if self._params is None:
            return
        self._params.curves[channel] = [tuple(p) for p in points]
        self.committed.emit()

    def _reset_current(self):
        self._apply_reset(self.editor.current_channel())

    def _reset_all(self):
        self._apply_reset("all")

    def _apply_reset(self, channel: str):
        if self._params is None:
            return
        from core import default_curves
        channels = ("master", "red", "green", "blue") if channel == "all" else (channel,)
        for ch in channels:
            self._params.curves[ch] = [(0.0, 0.0), (1.0, 1.0)]
            self.editor.set_points(ch, self._params.curves[ch])
        self.changed.emit()
        self.committed.emit()


# ======================================================================
# 3. 颜色分级面板
# ======================================================================

class GradingPanel(QWidget):
    changed = Signal()
    committed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._params: Adjustments | None = None
        self._blocked = False

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(6)

        grid = QGridLayout()
        grid.setSpacing(10)
        self.wheels: dict = {}
        self.sat_rows: dict = {}
        self.lum_rows: dict = {}
        positions = {"shadows": (0, 0), "midtones": (0, 1),
                     "highlights": (1, 0), "global": (1, 1)}
        for name in ("shadows", "midtones", "highlights", "global"):
            cell = QVBoxLayout()
            cell.setSpacing(4)
            caption = QLabel(RANGE_LABELS[name])
            caption.setObjectName("sectionTitle")
            caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
            cell.addWidget(caption)
            wheel = ColorWheel()
            wheel.hueSatChanged.connect(lambda h, s, n=name: self._on_wheel(n, h, s))
            wheel.wheelReleased.connect(self.committed)
            cell.addWidget(wheel, 0, Qt.AlignmentFlag.AlignCenter)
            sat_row = SliderRow("饱和度", -100, 100, 0)
            sat_row.valueChanged.connect(lambda v, n=name: self._on_sat(n, v))
            sat_row.sliderReleased.connect(self.committed)
            lum_row = SliderRow("亮度", -100, 100, 0)
            lum_row.valueChanged.connect(lambda v, n=name: self._on_lum(n, v))
            lum_row.sliderReleased.connect(self.committed)
            wrapper = QWidget()
            wl = QVBoxLayout(wrapper)
            wl.setContentsMargins(0, 0, 0, 0)
            wl.setSpacing(4)
            wl.addLayout(cell)
            wl.addWidget(sat_row)
            wl.addWidget(lum_row)
            grid.addWidget(wrapper, *positions[name])
            self.wheels[name] = wheel
            self.sat_rows[name] = sat_row
            self.lum_rows[name] = lum_row
        lay.addLayout(grid)

        box = QGroupBox("混合")
        bl = QVBoxLayout(box)
        self.blend_row = SliderRow("混合", 0, 100, 50)
        self.blend_row.valueChanged.connect(self._on_blend)
        self.blend_row.sliderReleased.connect(self.committed)
        bl.addWidget(self.blend_row)
        lay.addWidget(box)
        lay.addStretch(1)

    def set_params(self, params: Adjustments):
        self._params = params
        self.sync()

    def sync(self):
        if self._params is None:
            return
        self._blocked = True
        for name in ("shadows", "midtones", "highlights", "global"):
            w = self._params.wheels[name]
            self.sat_rows[name].set_value(w.saturation)
            self.lum_rows[name].set_value(w.luminance)
            self._refresh_wheel(name)
        self.blend_row.set_value(self._params.blend)
        self._blocked = False

    def _refresh_wheel(self, name: str):
        w = self._params.wheels[name]
        negative = w.saturation < 0
        self.wheels[name].set_hue_sat(w.hue + (180.0 if negative else 0.0),
                                      abs(w.saturation) / 100.0)

    def _on_wheel(self, name: str, hue: float, sat: float):
        if self._blocked or self._params is None:
            return
        w = self._params.wheels[name]
        w.hue = hue
        w.saturation = sat * 100.0
        self.sat_rows[name].set_value(w.saturation)
        self.changed.emit()

    def _on_sat(self, name: str, value: float):
        if self._blocked or self._params is None:
            return
        self._params.wheels[name].saturation = value
        self._refresh_wheel(name)
        self.changed.emit()

    def _on_lum(self, name: str, value: float):
        if self._blocked or self._params is None:
            return
        self._params.wheels[name].luminance = value
        self.changed.emit()

    def _on_blend(self, value: float):
        if self._blocked or self._params is None:
            return
        self._params.blend = value
        self.changed.emit()


# ======================================================================
# 4. 取色器面板
# ======================================================================

class _PickerRow(QFrame):
    def __init__(self, picker: PickerState, index: int, panel: "PickerPanel"):
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.index = index
        self.panel = panel

        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)

        head = QHBoxLayout()
        self.swatch = QPushButton()
        self.swatch.setFixedSize(26, 26)
        self.swatch.setToolTip("选中此取色点")
        self.swatch.clicked.connect(lambda: panel.select_row(index))
        self.info = QLabel()
        self.info.setObjectName("groupLabel")
        repick = QPushButton("重新取色")
        repick.setToolTip("然后在图像上单击以重新拾取颜色")
        repick.clicked.connect(lambda: panel.arm_repick(index))
        delete = QPushButton("✕")
        delete.setFixedWidth(30)
        delete.setToolTip("删除此取色点")
        delete.clicked.connect(lambda: panel.delete_picker(index))
        head.addWidget(self.swatch)
        head.addWidget(self.info, 1)
        head.addWidget(repick)
        head.addWidget(delete)
        lay.addLayout(head)

        self.span_row = SliderRow("影响范围", 5, 180, picker.span,
                                  default_value=30.0, suffix="°")
        self.span_row.valueChanged.connect(lambda v: self._set("span", v))
        self.span_row.sliderReleased.connect(panel.committed)
        # 色相偏移限制在 ±60°，即可令"绿"偏黄或偏青、"红"偏橙或偏品红……
        self.hue_row = SliderRow("色相偏移", -60, 60, picker.hue_shift,
                                 default_value=0.0, suffix="°")
        self.hue_row.valueChanged.connect(lambda v: self._set("hue_shift", v))
        self.hue_row.sliderReleased.connect(panel.committed)
        self.sat_row = SliderRow("饱和度", -100, 100, picker.saturation)
        self.sat_row.valueChanged.connect(lambda v: self._set("saturation", v))
        self.sat_row.sliderReleased.connect(panel.committed)
        self.lum_row = SliderRow("明亮度", -100, 100, picker.luminance)
        self.lum_row.valueChanged.connect(lambda v: self._set("luminance", v))
        self.lum_row.sliderReleased.connect(panel.committed)
        for r in (self.span_row, self.hue_row, self.sat_row, self.lum_row):
            lay.addWidget(r)

        self.update_from(picker)

    def _set(self, attr: str, value: float):
        picker = self.panel._params.pickers[self.index]
        setattr(picker, attr, value)
        self.panel.changed.emit()

    def update_from(self, picker: PickerState):
        r, g, b = [int(c * 255) for c in picker.color]
        self.swatch.setStyleSheet(f"background-color: rgb({r},{g},{b});")
        self.info.setText(f"色相 {picker.hue:.0f}°  范围 ±{picker.span:.0f}°")
        self.span_row.set_value(picker.span)
        self.hue_row.set_value(picker.hue_shift)
        self.sat_row.set_value(picker.saturation)
        self.lum_row.set_value(picker.luminance)
        selected = self.index == self.panel._selected
        self.setStyleSheet(
            "QFrame { border: 1px solid #3a3a41; border-radius: 6px; background: #232326; }"
            + ("QFrame { border: 1px solid #4c9be8; }" if selected else ""))


class PickerPanel(QWidget):
    changed = Signal()
    committed = Signal()
    pickRequested = Signal(object)     # 'new' 或 取色点索引
    pickCancelled = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._params: Adjustments | None = None
        self._selected: int | None = None
        self._target = None             # None | 'new' | int
        self._rows: list = []

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(6)

        top = QHBoxLayout()
        self.new_btn = QPushButton("＋ 从图像取色")
        self.new_btn.setCheckable(True)
        self.new_btn.clicked.connect(self.arm_new)
        top.addWidget(self.new_btn)
        self.hint = QLabel("在图像上单击拾取颜色")
        self.hint.setObjectName("dim")
        top.addWidget(self.hint, 1)
        lay.addLayout(top)

        # 预设颜色（每个可单独微调色相，如绿 -> 黄/青）
        preset_box = QGroupBox("预设颜色")
        preset_lay = QGridLayout(preset_box)
        preset_lay.setSpacing(4)
        self.preset_buttons = []
        for i, (name, hue) in enumerate(core.HS_PRESETS):
            btn = QPushButton(name)
            r, g, b = [int(c * 255) for c in core.preset_color(hue)]
            btn.setStyleSheet(f"background-color: rgb({r},{g},{b}); color: #101012;")
            btn.setToolTip(f"新建「{name}」取色点（色相 {hue:.0f}°，可再微调色相）")
            btn.clicked.connect(lambda _=False, h=hue: self.new_preset(h))
            preset_lay.addWidget(btn, i // 4, i % 4)
            self.preset_buttons.append(btn)
        lay.addWidget(preset_box)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.list_widget = QWidget()
        self.list_layout = QVBoxLayout(self.list_widget)
        self.list_layout.setContentsMargins(0, 0, 0, 0)
        self.list_layout.setSpacing(6)
        self.list_layout.addStretch(1)
        scroll.setWidget(self.list_widget)
        lay.addWidget(scroll, 1)

    def set_params(self, params: Adjustments):
        self._params = params
        self.sync()

    def sync(self):
        if self._params is None:
            return
        if len(self._params.pickers) != len(self._rows):
            self._rebuild()
        else:
            for row in self._rows:
                row.update_from(self._params.pickers[row.index])

    def _rebuild(self):
        while self.list_layout.count() > 1:      # 保留末尾 stretch
            item = self.list_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._rows = []
        for i, p in enumerate(self._params.pickers):
            row = _PickerRow(p, i, self)
            self.list_layout.insertWidget(self.list_layout.count() - 1, row)
            self._rows.append(row)

    # ------------------------------------------------------------ 交互
    def arm_new(self):
        self._target = "new"
        self.new_btn.setChecked(True)
        self.hint.setText("在图像上单击拾取颜色…")
        self.pickRequested.emit("new")

    def new_preset(self, hue_deg: float):
        """由预设色相直接创建取色点（无需在图像上取色）。"""
        if self._params is None:
            return
        state = PickerState(hue=hue_deg, color=core.preset_color(hue_deg),
                            span=30.0, saturation=0.0)
        self._params.pickers.append(state)
        self._selected = len(self._params.pickers) - 1
        self.sync()
        self.committed.emit()

    def arm_repick(self, index: int):
        self._target = index
        self._selected = index
        self.new_btn.setChecked(False)
        self.hint.setText(f"在图像上单击以更新取色点 {index + 1}…")
        self.pickRequested.emit(index)

    def cancel_pick(self):
        self._target = None
        self.new_btn.setChecked(False)
        self.hint.setText("在图像上单击拾取颜色")
        self.pickCancelled.emit()

    def select_row(self, index: int):
        self._selected = index
        self.sync()

    def delete_picker(self, index: int):
        if self._params is None or not (0 <= index < len(self._params.pickers)):
            return
        del self._params.pickers[index]
        if self._selected is not None and self._selected >= len(self._params.pickers):
            self._selected = None
        self.sync()
        self.committed.emit()

    def on_picked(self, color, hue_deg: float):
        """图像上完成取色后由主窗口调用，color 为 sRGB 三元组。"""
        if self._params is None:
            return
        if self._target == "new" or self._target is None:
            state = PickerState(hue=hue_deg, color=tuple(color), span=30.0,
                                saturation=25.0)
            self._params.pickers.append(state)
            self._selected = len(self._params.pickers) - 1
        else:
            p = self._params.pickers[int(self._target)]
            p.hue = hue_deg
            p.color = tuple(color)
        self.cancel_pick()
        self.sync()
        self.committed.emit()


# ======================================================================
# 4b. 蒙版面板（线性渐变 / 径向渐变 / 画笔）
# ======================================================================

MASK_LOCAL_ROWS = (
    ("曝光", "exposure", -5.0, 5.0, 2, " EV"),
    ("对比度", "contrast", -100, 100, 0, ""),
    ("高光", "highlights", -100, 100, 0, ""),
    ("阴影", "shadows", -100, 100, 0, ""),
    ("白色", "whites", -100, 100, 0, ""),
    ("黑色", "blacks", -100, 100, 0, ""),
    ("色温", "temperature", -100, 100, 0, ""),
    ("色调", "tint", -100, 100, 0, ""),
    ("饱和度", "saturation", -100, 100, 0, ""),
    ("清晰度", "clarity", -100, 100, 0, ""),
    ("锐度", "sharpen", 0, 100, 0, ""),
    ("去朦胧", "dehaze", -100, 100, 0, ""),
)
MASK_KIND_LABELS = {"linear": "线性渐变", "radial": "径向渐变", "brush": "画笔"}


class _MaskRow(QFrame):
    def __init__(self, mask: "core.MaskState", index: int, panel: "MaskPanel"):
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.index = index
        self.panel = panel

        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 5, 6, 5)
        lay.setSpacing(6)

        self.pick_btn = QPushButton()
        self.pick_btn.setFixedWidth(92)
        self.pick_btn.setCheckable(True)
        self.pick_btn.clicked.connect(lambda: panel.select_mask(index))
        lay.addWidget(self.pick_btn)

        self.edit_btn = QPushButton("编辑")
        self.edit_btn.setFixedWidth(48)
        self.edit_btn.setToolTip("在图像上拖动编辑该蒙版（Esc 退出）")
        self.edit_btn.clicked.connect(lambda: panel.request_edit(index))
        lay.addWidget(self.edit_btn)

        self.update_from(mask, False, False)

    def update_from(self, mask: "core.MaskState", selected: bool, editing: bool):
        self.pick_btn.setText(MASK_KIND_LABELS.get(mask.kind, mask.kind) +
                              ("（反）" if mask.invert else ""))
        self.pick_btn.setStyleSheet(
            "QPushButton { text-align: left; padding: 4px 6px; }"
            + ("QPushButton:checked { background-color: #2f6db3; border-color: #4c9be8; }"
               if selected else ""))
        self.pick_btn.setChecked(selected)
        self.edit_btn.setText("完成" if editing else "编辑")
        self.edit_btn.setChecked(editing)
        self.setStyleSheet(
            "QFrame { border: 1px solid #3a3a41; border-radius: 6px; background: #232326; }"
            + ("QFrame { border-color: #4c9be8; }" if selected else ""))


class MaskPanel(QWidget):
    changed = Signal()
    committed = Signal()
    overlayRefresh = Signal()             # 几何/羽化/反向变化 -> 重绘叠加层
    selectionChanged = Signal(int)
    editModeRequested = Signal(object)    # 蒙版索引 或 None（退出）
    brushCleared = Signal()
    brushUndone = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._params = None
        self._selected: int | None = None
        self._editing: int | None = None
        self._blocked = False
        self._rows: list = []
        self._param_rows: dict = {}

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(6)

        new_row = QHBoxLayout()
        for kind, label in (("linear", "线性渐变"), ("radial", "径向渐变"), ("brush", "画笔")):
            btn = QPushButton("＋ " + label)
            btn.setToolTip({"linear": "从一端到另一端的渐变蒙版",
                            "radial": "椭圆形渐变蒙版",
                            "brush": "用画笔涂抹出蒙版"}[kind])
            btn.clicked.connect(lambda _=False, k=kind: self.create_mask(k))
            new_row.addWidget(btn)
        lay.addLayout(new_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.list_widget = QWidget()
        self.list_layout = QVBoxLayout(self.list_widget)
        self.list_layout.setContentsMargins(0, 0, 0, 0)
        self.list_layout.setSpacing(4)
        self.list_layout.addStretch(1)
        scroll.setWidget(self.list_widget)
        lay.addWidget(scroll)

        # ---- 参数区（随选中蒙版动态构建）----
        self.params_box = QWidget()
        self.params_layout = QVBoxLayout(self.params_box)
        self.params_layout.setContentsMargins(0, 0, 0, 0)
        self.params_layout.setSpacing(6)
        lay.addWidget(self.params_box)

        self.hint = QLabel("创建蒙版后可在此调整参数与局部调整")
        self.hint.setObjectName("dim")
        self.hint.setWordWrap(True)
        lay.addWidget(self.hint)
        lay.addStretch(1)

    # ------------------------------------------------------------ 数据
    def set_params(self, params):
        self._params = params
        self.sync()

    def selected_index(self):
        return self._selected

    def sync(self):
        if self._params is None:
            return
        if self._selected is not None and self._selected >= len(self._params.masks):
            self._selected = len(self._params.masks) - 1 if self._params.masks else None
            self._editing = None
        if len(self._params.masks) != len(self._rows):
            self._rebuild_rows()
        else:
            for i, row in enumerate(self._rows):
                row.update_from(self._params.masks[i], i == self._selected,
                                i == self._editing)
        self._rebuild_params(want_focus=False)

    def _rebuild_rows(self):
        while self.list_layout.count() > 1:
            item = self.list_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._rows = []
        for i, m in enumerate(self._params.masks):
            row = _MaskRow(m, i, self)
            self.list_layout.insertWidget(self.list_layout.count() - 1, row)
            self._rows.append(row)

    # ------------------------------------------------------------ 面板操作
    def create_mask(self, kind: str):
        if self._params is None:
            return
        m = core.MaskState(kind=kind, name=MASK_KIND_LABELS[kind])
        if kind == "linear":
            m.x1, m.y1, m.x2, m.y2 = 0.3, 0.7, 0.7, 0.3
        elif kind == "radial":
            m.cx, m.cy, m.radius, m.aspect = 0.5, 0.5, 0.30, 1.0
        else:
            m.size, m.feather = 12.0, 25.0
        self._params.masks.append(m)
        self._selected = len(self._params.masks) - 1
        self.sync()
        self.selectionChanged.emit(self._selected)
        self.request_edit(self._selected)
        self.committed.emit()

    def select_mask(self, index: int):
        self._selected = index
        self.sync()
        self.selectionChanged.emit(index)

    def request_edit(self, index: int):
        self._editing = index
        self._selected = index
        self.sync()
        self.selectionChanged.emit(index)
        self.editModeRequested.emit(index)

    def exit_edit(self):
        if self._editing is None:
            return
        self._editing = None
        self.sync()
        self.editModeRequested.emit(None)

    def delete_mask(self, index: int):
        if self._params is None or not (0 <= index < len(self._params.masks)):
            return
        if self._editing == index:
            self._editing = None
            self.editModeRequested.emit(None)
        elif self._editing is not None and self._editing > index:
            self._editing -= 1
        del self._params.masks[index]
        if self._selected is not None and self._selected >= len(self._params.masks):
            self._selected = None
        self.sync()
        self.committed.emit()

    # ------------------------------------------------------------ 参数区
    def _current_mask(self):
        if self._params is None or self._selected is None:
            return None
        if self._selected >= len(self._params.masks):
            return None
        return self._params.masks[self._selected]

    def _clear_params(self):
        while self.params_layout.count():
            item = self.params_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._param_rows = {}
        self._geo_rows = []

    def _add_row(self, label, lo, hi, value, decimals, suffix, default,
                 on_change, on_release=None, geo=None):
        row = SliderRow(label, lo, hi, value, decimals=decimals, suffix=suffix,
                        default_value=default)
        row.valueChanged.connect(on_change)
        row.sliderReleased.connect(on_release or self.committed)
        self.params_layout.addWidget(row)
        if geo is not None:
            self._geo_rows.append((geo[0], geo[1], row))
        return row

    def _rebuild_params(self, want_focus=False):
        self._clear_params()
        m = self._current_mask()
        if m is None:
            self.hint.setText("创建蒙版后可在此调整参数与局部调整")
            return
        self.hint.setText(f"{MASK_KIND_LABELS.get(m.kind, m.kind)}蒙版："
                          "在图像上拖动编辑；Esc 退出编辑")

        head = QHBoxLayout()
        invert = QPushButton("反向")
        invert.setCheckable(True)
        invert.setChecked(m.invert)
        invert.setToolTip("反向：作用于蒙版之外的区域")
        invert.clicked.connect(lambda checked: self._set_attr("invert", checked))
        head.addWidget(invert)
        delete = QPushButton("删除")
        delete.setToolTip("删除该蒙版")
        delete.clicked.connect(lambda: self.delete_mask(self._selected))
        head.addWidget(delete)
        head.addStretch(1)
        holder = QWidget()
        hl = QVBoxLayout(holder)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.addLayout(head)
        self.params_layout.addWidget(holder)

        self._add_row("羽化", 0, 100, m.feather, 0, "", 30.0,
                      self._on_feather)

        if m.kind == "brush":
            self._add_row("笔刷大小", 1, 100, m.size, 0, "", 12.0, self._on_size)
            btns = QHBoxLayout()
            undo = QPushButton("撤销一笔")
            undo.clicked.connect(lambda: self.brushUndone.emit())
            clear = QPushButton("清除笔迹")
            clear.clicked.connect(lambda: self.brushCleared.emit())
            btns.addWidget(undo)
            btns.addWidget(clear)
            btns.addStretch(1)
            hb = QWidget()
            hbl = QVBoxLayout(hb)
            hbl.setContentsMargins(0, 0, 0, 0)
            hbl.addLayout(btns)
            self.params_layout.addWidget(hb)
        elif m.kind == "radial":
            self._add_row("半径", 5, 100, m.radius * 100, 0, "%", 30.0, self._on_radius,
                          geo=("radius", 100.0))
            self._add_row("长宽比", 20, 300, m.aspect * 100, 0, "%", 100.0, self._on_aspect,
                          geo=("aspect", 100.0))
        else:
            self._add_row("起点 X", 0, 100, m.x1 * 100, 0, "%", 30.0,
                          lambda v: self._on_linear(0, v), geo=("x1", 100.0))
            self._add_row("起点 Y", 0, 100, m.y1 * 100, 0, "%", 70.0,
                          lambda v: self._on_linear(1, v), geo=("y1", 100.0))
            self._add_row("终点 X", 0, 100, m.x2 * 100, 0, "%", 70.0,
                          lambda v: self._on_linear(2, v), geo=("x2", 100.0))
            self._add_row("终点 Y", 0, 100, m.y2 * 100, 0, "%", 30.0,
                          lambda v: self._on_linear(3, v), geo=("y2", 100.0))

        box = QGroupBox("局部调整")
        gl = QVBoxLayout(box)
        gl.setSpacing(6)
        for label, name, lo, hi, decimals, suffix in MASK_LOCAL_ROWS:
            row = SliderRow(label, lo, hi, getattr(m.adj, name), decimals=decimals,
                            suffix=suffix, default_value=0.0)
            row.valueChanged.connect(lambda v, n=name: self._on_adj(n, v))
            row.sliderReleased.connect(self.committed)
            gl.addWidget(row)
            self._param_rows[name] = row
        self.params_layout.addWidget(box)

    # ------------------------------------------------------------ 事件
    def _set_attr(self, attr, value):
        m = self._current_mask()
        if m is None:
            return
        setattr(m, attr, value)
        self.overlayRefresh.emit()
        self.sync()
        self.committed.emit()

    def _on_feather(self, v: float):
        m = self._current_mask()
        if m is None:
            return
        m.feather = v
        self.overlayRefresh.emit()
        self.changed.emit()

    def _on_size(self, v: float):
        m = self._current_mask()
        if m is None or m.kind != "brush":
            return
        m.size = v
        self.overlayRefresh.emit()
        self.changed.emit()

    def _on_radius(self, v: float):
        m = self._current_mask()
        if m is None:
            return
        m.radius = v / 100.0
        self.overlayRefresh.emit()
        self.changed.emit()

    def _on_aspect(self, v: float):
        m = self._current_mask()
        if m is None:
            return
        m.aspect = v / 100.0
        self.overlayRefresh.emit()
        self.changed.emit()

    def _on_linear(self, index: int, v: float):
        m = self._current_mask()
        if m is None or m.kind != "linear":
            return
        attr = ("x1", "y1", "x2", "y2")[index]
        setattr(m, attr, v / 100.0)
        self.overlayRefresh.emit()
        self.changed.emit()

    def _on_adj(self, name: str, v: float):
        m = self._current_mask()
        if m is None:
            return
        setattr(m.adj, name, v)
        self.changed.emit()

    def sync_values(self):
        """外部（画布拖拽）修改了几何后，只刷新数值不回填控件。"""
        self._rebuild_params()

    def update_geometry_sliders(self):
        """画布拖拽蒙版时，只更新几何滑块（避免重建导致拖动中断）。"""
        m = self._current_mask()
        if m is None or not self._geo_rows:
            return
        self._blocked = True
        for attr, mult, row in self._geo_rows:
            row.set_value(getattr(m, attr) * mult)
        self._blocked = False


# ======================================================================
# 5. 右侧总面板（可折叠分区）
# ======================================================================

class AdjustmentPanel(QWidget):
    changed = Signal()
    committed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.basic = BasicPanel()
        self.curve = CurvePanel()
        self.grading = GradingPanel()
        self.picker = PickerPanel()
        self.mask = MaskPanel()

        for p in (self.basic, self.curve, self.grading, self.picker):
            p.changed.connect(self.changed)
            p.committed.connect(self.committed)
        self.mask.changed.connect(self.changed)
        self.mask.committed.connect(self.committed)

        titles = {BasicPanel: "基础", CurvePanel: "曲线",
                  GradingPanel: "颜色分级", PickerPanel: "取色器",
                  MaskPanel: "蒙版"}
        container = QWidget()
        cl = QVBoxLayout(container)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(8)
        for p in (self.basic, self.curve, self.grading, self.picker, self.mask):
            cl.addWidget(CollapsibleSection(titles[type(p)], p, expanded=True))
        cl.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def set_params(self, params: Adjustments):
        for p in (self.basic, self.curve, self.grading, self.picker, self.mask):
            p.set_params(params)

    def sync(self):
        for p in (self.basic, self.curve, self.grading, self.picker, self.mask):
            p.sync()

    def set_histograms(self, hist):
        self.curve.set_histograms(hist)


# ======================================================================
# 6. 主窗口
# ======================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PhotoLab — 照片调色")
        self.resize(1460, 920)
        self.setAcceptDrops(True)

        # ---------------- 状态 ----------------
        self.params = Adjustments()
        self.source: np.ndarray | None = None        # 全图 uint8 RGB
        self.preview_src: np.ndarray | None = None   # 降采样图 uint8 RGB
        self._newest_rid = 0
        self._comparing = False
        self._last_frame: np.ndarray | None = None
        self._save_path: str | None = None
        self._save_rid = -1
        self._last_hist_time = 0.0
        self._last_overlay_time = 0.0
        self._last_overlay_time = 0.0

        # ---------------- 后台处理线程 ----------------
        self.processor = ProcessorThread()
        self.processor.frameReady.connect(self._on_frame_ready)
        self.processor.busyChanged.connect(self._on_busy_changed)
        self.processor.start()

        # ---------------- 界面 ----------------
        self.preview = PreviewWidget()
        self.preview.colorPicked.connect(self._on_image_picked)
        self.preview.zoomChanged.connect(self._update_status)

        self.panel = AdjustmentPanel()
        self.panel.set_params(self.params)
        self.panel.changed.connect(self._on_params_changed)
        self.panel.committed.connect(self._on_params_committed)
        self.panel.picker.pickRequested.connect(self._on_pick_requested)
        self.panel.picker.pickCancelled.connect(self._on_pick_cancelled)

        # ---- 蒙版 ----
        self._mask_sel: int | None = None
        self._mask_edit: int | None = None
        self._overlay_cache: dict = {}
        self._stroke_closed = False     # 画笔：上一笔是否已结束
        self.panel.mask.selectionChanged.connect(self._on_mask_selected)
        self.panel.mask.editModeRequested.connect(self._on_mask_edit_requested)
        self.panel.mask.overlayRefresh.connect(self._refresh_mask_overlays)
        self.panel.mask.brushCleared.connect(self._on_brush_cleared)
        self.panel.mask.brushUndone.connect(self._on_brush_undo)
        self.preview.linearHandleMoved.connect(self._on_linear_handle)
        self.preview.radialCenterMoved.connect(self._on_radial_center)
        self.preview.brushStrokePoint.connect(self._on_brush_point)
        self.preview.brushStrokeFinished.connect(self._on_brush_finished)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.preview)
        splitter.addWidget(self.panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([1040, 380])
        self.setCentralWidget(splitter)

        self._build_toolbar()
        self._build_statusbar()
        self._build_shortcuts()

        # 空闲后自动处理全图
        self._full_timer = QTimer(self)
        self._full_timer.setSingleShot(True)
        self._full_timer.setInterval(400)
        self._full_timer.timeout.connect(lambda: self._submit(FULL))

        self._update_status()

    # ------------------------------------------------------------ 工具栏
    def _build_toolbar(self):
        tb = QToolBar("主工具栏")
        tb.setMovable(False)
        tb.setIconSize(self.iconSize())
        self.addToolBar(tb)

        act_open = QAction("打开", self)
        act_open.setShortcut(QKeySequence.StandardKey.Open)
        act_open.triggered.connect(self.open_image)
        tb.addAction(act_open)

        act_save = QAction("保存", self)
        act_save.setShortcut(QKeySequence.StandardKey.Save)
        act_save.triggered.connect(self.save_image)
        tb.addAction(act_save)

        act_reset = QAction("重置", self)
        act_reset.setToolTip("重置所有调整 (Ctrl+R)")
        act_reset.triggered.connect(self.reset_adjustments)
        tb.addAction(act_reset)

        tb.addSeparator()

        self.act_compare = QAction("对比原图", self)
        self.act_compare.setCheckable(True)
        self.act_compare.setShortcut("\\")
        self.act_compare.triggered.connect(self._toggle_compare)
        tb.addAction(self.act_compare)

        tb.addSeparator()

        act_fit = QAction("适合窗口", self)
        act_fit.triggered.connect(self.preview.zoom_fit)
        tb.addAction(act_fit)

        act_1to1 = QAction("1:1", self)
        act_1to1.triggered.connect(self.preview.zoom_one_to_one)
        tb.addAction(act_1to1)

        act_in = QAction("放大", self)
        act_in.setShortcut(QKeySequence.StandardKey.ZoomIn)
        act_in.triggered.connect(lambda: self.preview.zoom_at(1.25))
        tb.addAction(act_in)

        act_out = QAction("缩小", self)
        act_out.setShortcut(QKeySequence.StandardKey.ZoomOut)
        act_out.triggered.connect(lambda: self.preview.zoom_at(0.8))
        tb.addAction(act_out)

        tb.addSeparator()

        self.act_pick = QAction("取色器", self)
        self.act_pick.setCheckable(True)
        self.act_pick.setToolTip("在图像上单击拾取目标颜色")
        self.act_pick.triggered.connect(self._toggle_pick_mode)
        tb.addAction(self.act_pick)

    def _build_statusbar(self):
        sb = self.statusBar()
        self.status_image = QLabel("未打开图像")
        self.status_zoom = QLabel("")
        self.status_proc = QLabel("")
        sb.addWidget(self.status_image, 1)
        sb.addPermanentWidget(self.status_zoom)
        sb.addPermanentWidget(self.status_proc)

    def _build_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+R"), self, self.reset_adjustments)
        QShortcut(QKeySequence("0"), self, self.preview.zoom_fit)
        QShortcut(QKeySequence("1"), self, self.preview.zoom_one_to_one)
        QShortcut(QKeySequence("Delete"), self, self._delete_selected_picker)
        QShortcut(QKeySequence("Escape"), self, self._exit_mask_edit)

    def _delete_selected_picker(self):
        idx = self.panel.picker._selected
        if idx is not None:
            self.panel.picker.delete_picker(idx)

    # ------------------------------------------------------------ 图像
    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开图像", "", IMAGE_FILTER)
        if path:
            self.load_image(path)

    def load_image(self, path: str):
        try:
            data = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)   # 兼容中文路径
        except Exception:
            img = None
        if img is None:
            QMessageBox.warning(self, "打开失败", f"无法解码图像：\n{path}")
            return
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.source = np.ascontiguousarray(rgb)
        self.preview_src = downscale_max_edge(self.source, PREVIEW_MAX_EDGE)

        self.processor.set_source(PREVIEW, self.preview_src)
        self.processor.set_source(FULL, self.source)
        self.preview.set_full_size(self.source.shape[1], self.source.shape[0])
        self.preview.zoom_fit()

        self._overlay_cache.clear()
        self._newest_rid = self.processor.pending_request()
        self._on_params_changed(force_full=True)
        self._refresh_mask_overlays()
        self._update_status()

    def reset_adjustments(self):
        self.params = Adjustments()
        self._mask_sel = None
        self._mask_edit = None
        self._overlay_cache.clear()
        self.preview.set_edit_mode("none")
        self.panel.set_params(self.params)
        self._on_params_changed(force_full=True)
        self._refresh_mask_overlays()
        self.statusBar().showMessage("已重置全部调整", 1500)

    def save_image(self):
        if self.source is None:
            QMessageBox.information(self, "保存", "请先打开一幅图像。")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "保存图像", "output.jpg",
            "JPEG (*.jpg);;PNG (*.png);;WebP (*.webp);;TIFF (*.tif)")
        if not path:
            return
        if not os.path.splitext(path)[1]:
            path += ".jpg"
        self._save_path = path
        self._save_rid = self._submit(FULL)
        self.status_proc.setText("正在渲染全图…")

    def _write_image(self, arr: np.ndarray, path: str):
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext in (".jpg", ".jpeg"):
                ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            elif ext == ".png":
                ok, buf = cv2.imencode(".png", bgr)
            elif ext == ".webp":
                ok, buf = cv2.imencode(".webp", bgr, [cv2.IMWRITE_WEBP_QUALITY, 95])
            elif ext in (".tif", ".tiff"):
                ok, buf = cv2.imencode(".tif", bgr)
            else:
                ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if ok:
                buf.tofile(path)          # numpy 写入，支持中文路径
            self.statusBar().showMessage(f"已保存：{path}", 4000)
        except Exception:
            traceback.print_exc()
            QMessageBox.warning(self, "保存失败", f"写入失败：\n{path}")

    # ------------------------------------------------------------ 调度
    def _submit(self, kind: str) -> int:
        if self.source is None:
            return -1
        return self.processor.submit(self.params, kind)

    def _on_params_changed(self, _value=None, force_full: bool = False):
        """参数变化：先刷新降采样预览，短暂空闲后处理全图。"""
        if self.source is None:
            return
        self._submit(PREVIEW)
        if force_full:
            self._full_timer.stop()
            self._submit(FULL)
        else:
            self._full_timer.start()

    def _on_params_committed(self):
        if self.source is None:
            return
        self._full_timer.stop()
        self._submit(FULL)

    def _on_busy_changed(self, busy: bool):
        self.status_proc.setText("⏳ 处理中…" if busy else "")

    def _on_frame_ready(self, arr: np.ndarray, rid: int, is_full: bool):
        if rid < self._newest_rid:
            return                                # 过期结果（已被更新的请求取代）
        self._newest_rid = rid
        self._last_frame = arr

        if self._comparing:
            if self.preview_src is not None:
                self.preview.set_frame(self.preview_src)
        else:
            self.preview.set_frame(arr)

        now = time.monotonic()
        if not self._comparing and now - self._last_hist_time > 0.3:
            self._last_hist_time = now
            try:
                self.panel.set_histograms(linear_histograms(arr))
            except Exception:
                traceback.print_exc()

        if self._save_path is not None and is_full and rid == self._save_rid:
            path, self._save_path = self._save_path, None
            self._write_image(arr, path)

        self._update_status()

    # ------------------------------------------------------------ 取色器
    def _toggle_pick_mode(self, checked: bool):
        if checked:
            self.panel.picker.arm_new()
        else:
            self.panel.picker.cancel_pick()
        self.act_pick.setChecked(self.panel.picker._target is not None)

    def _on_pick_requested(self, _target):
        self.preview.set_pick_mode(True)
        self.act_pick.setChecked(True)

    def _on_pick_cancelled(self):
        self.preview.set_pick_mode(False)
        self.act_pick.setChecked(False)

    def _on_image_picked(self, x: int, y: int):
        if self.source is None:
            return
        frame_w = self.preview.current_frame_size()[0]
        if frame_w <= 0:
            return
        fx, fy = self.preview.map_to_frame(x, y)
        scale = self.source.shape[1] / frame_w
        px = int(round(fx * scale))
        py = int(round(fy * scale))
        patch = max(3, int(round(self.source.shape[0] * 0.01)))
        color = sample_patch(self.source, px, py, patch=patch)
        self.panel.picker.on_picked(color, hue_degrees(color))
        self._on_params_changed()

    # ------------------------------------------------------------ 蒙版
    def _current_mask(self):
        if self._mask_sel is None:
            return None
        if not (0 <= self._mask_sel < len(self.params.masks)):
            return None
        return self.params.masks[self._mask_sel]

    def _on_mask_selected(self, index: int):
        self._mask_sel = index
        self._refresh_mask_overlays()

    def _on_mask_edit_requested(self, target):
        if target is None:
            self._exit_mask_edit()
            return
        self._mask_sel = int(target)
        self._mask_edit = int(target)
        m = self._current_mask()
        if m is None:
            return
        if self.preview._pick_mode:
            self.panel.picker.cancel_pick()
        self.preview.set_edit_mode(m.kind, self._mask_geometry(m))
        self._refresh_mask_overlays()
        self._update_status()

    @staticmethod
    def _mask_geometry(m) -> dict:
        if m.kind == "linear":
            return {"x1": m.x1, "y1": m.y1, "x2": m.x2, "y2": m.y2}
        if m.kind == "radial":
            return {"cx": m.cx, "cy": m.cy, "radius": m.radius, "aspect": m.aspect}
        return {"size": m.size}

    def _exit_mask_edit(self):
        if self._mask_edit is None:
            return
        self._mask_edit = None
        self.preview.set_edit_mode("none")
        self.panel.mask.exit_edit()

    def _on_linear_handle(self, handle: int, x1: float, y1: float,
                          x2: float, y2: float):
        m = self._current_mask()
        if m is None or m.kind != "linear" or self._mask_edit is None:
            return
        m.x1, m.y1, m.x2, m.y2 = x1, y1, x2, y2
        self._after_mask_geometry_change(m)

    def _on_radial_center(self, nx: float, ny: float):
        m = self._current_mask()
        if m is None or m.kind != "radial" or self._mask_edit is None:
            return
        m.cx, m.cy = nx, ny
        self._after_mask_geometry_change(m)

    def _on_brush_point(self, nx: float, ny: float):
        m = self._current_mask()
        if m is None or m.kind != "brush" or self._mask_edit is None:
            return
        if not m.strokes or self._stroke_closed:
            m.strokes.append([])          # 新的一笔
            self._stroke_closed = False
        m.strokes[-1].append((nx, ny))
        now = time.monotonic()
        if now - self._last_overlay_time > 0.05:
            self._last_overlay_time = now
            self._refresh_mask_overlays()
            self._on_params_changed()

    def _on_brush_finished(self):
        self._stroke_closed = True
        m = self._current_mask()
        if m is not None and m.strokes and not m.strokes[-1]:
            m.strokes.pop()
        self._refresh_mask_overlays()
        self._on_params_committed()

    def _on_brush_cleared(self):
        m = self._current_mask()
        if m is None or m.kind != "brush":
            return
        m.strokes.clear()
        self._refresh_mask_overlays()
        self._on_params_committed()

    def _on_brush_undo(self):
        m = self._current_mask()
        if m is None or m.kind != "brush" or not m.strokes:
            return
        m.strokes.pop()
        self._refresh_mask_overlays()
        self._on_params_committed()

    def _after_mask_geometry_change(self, m):
        self.preview._edit_info = self._mask_geometry(m)
        self.preview.update()
        self.panel.mask.update_geometry_sliders()
        self._refresh_mask_overlays()
        self._on_params_changed()

    def _refresh_mask_overlays(self):
        """重绘蒙版叠加层（按缓存键缓存，避免重复渲染；低分辨率渲染足够显示）。"""
        if self.source is None or not self.params.masks:
            self.preview.set_overlays([])
            return
        shape = self.source.shape[:2]
        overlays = []
        for i, m in enumerate(self.params.masks):
            key = (mask_cache_key(m, shape), i == self._mask_sel)
            img = self._overlay_cache.get(key)
            if img is None:
                img = mask_to_qimage(render_mask(m, shape, max_edge=768),
                                     selected=(i == self._mask_sel))
                self._overlay_cache[key] = img
            overlays.append(img)
        self.preview.set_overlays(overlays)

    # ------------------------------------------------------------ 对比
    def _toggle_compare(self, checked: bool):
        self._comparing = checked
        if checked:
            if self.preview_src is not None:
                self.preview.set_frame(self.preview_src)
            self.status_proc.setText("对比原图")
        else:
            self.status_proc.setText("")
            if self._last_frame is not None:
                self.preview.set_frame(self._last_frame)

    # ------------------------------------------------------------ 其它
    def _update_status(self):
        if self.source is None:
            self.status_image.setText("未打开图像")
            self.status_zoom.setText("")
            return
        h, w = self.source.shape[:2]
        self.status_image.setText(f"{w} × {h} px")
        if self._last_frame is not None:
            fh, fw = self._last_frame.shape[:2]
            tag = "全图" if (fw, fh) == (w, h) else f"预览 {max(fw, fh)}px"
            self.status_proc.setText(f"{tag} {fw}×{fh}")
        self.status_zoom.setText(f"缩放 {self.preview.zoom * 100:.0f}%")
        if self._mask_edit is not None:
            self.status_image.setText(f"{w} × {h} px   ·   蒙版编辑中：拖动编辑，Esc 退出")

    def closeEvent(self, event):
        self.processor.stop()
        super().closeEvent(event)
