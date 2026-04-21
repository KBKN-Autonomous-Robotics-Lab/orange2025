#!/usr/bin/env python3
"""
intensity_filter_gui.py
反射強度フィルタ GUI ツール

intensity_filter_node の intensity_min / intensity_max パラメータを
tkinter スライダーでリアルタイム変更する。

Usage:
    python3 intensity_filter_gui.py
"""

import subprocess
import tkinter as tk

# ============================================================
#  パラメータ送信
# ============================================================

def set_param(name, value):
    """ros2 param set で intensity_filter ノードにパラメータを送る"""
    subprocess.Popen(
        ['ros2', 'param', 'set', '/intensity_filter', name, str(float(value))],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# ============================================================
#  GUI
# ============================================================

class IntensityFilterGUI:

    def __init__(self, root):
        self.root = root
        self.root.title('Intensity Filter')
        self.root.resizable(False, False)

        PAD = 12
        SLIDER_LEN = 400

        # ----- min スライダー -----
        tk.Label(root, text='intensity_min', anchor='w').grid(
            row=0, column=0, padx=PAD, pady=(PAD, 2), sticky='w'
        )
        self.min_var = tk.DoubleVar(value=0.0)
        self.min_label = tk.Label(root, text='0', width=5, anchor='e')
        self.min_label.grid(row=0, column=2, padx=(0, PAD))

        self.min_slider = tk.Scale(
            root,
            from_=0, to=255,
            resolution=1,
            orient=tk.HORIZONTAL,
            length=SLIDER_LEN,
            variable=self.min_var,
            showvalue=False,
            command=self._on_min_changed,
        )
        self.min_slider.grid(row=0, column=1, pady=(PAD, 2))

        # ----- max スライダー -----
        tk.Label(root, text='intensity_max', anchor='w').grid(
            row=1, column=0, padx=PAD, pady=(2, PAD), sticky='w'
        )
        self.max_var = tk.DoubleVar(value=255.0)
        self.max_label = tk.Label(root, text='255', width=5, anchor='e')
        self.max_label.grid(row=1, column=2, padx=(0, PAD))

        self.max_slider = tk.Scale(
            root,
            from_=0, to=255,
            resolution=1,
            orient=tk.HORIZONTAL,
            length=SLIDER_LEN,
            variable=self.max_var,
            showvalue=False,
            command=self._on_max_changed,
        )
        self.max_slider.grid(row=1, column=1, pady=(2, PAD))

        # ----- 範囲表示ラベル -----
        self.range_label = tk.Label(root, text='range: 0 ～ 255', fg='gray40')
        self.range_label.grid(row=2, column=0, columnspan=3, pady=(0, PAD))

    # ----------------------------------------------------------
    #  コールバック
    # ----------------------------------------------------------

    def _on_min_changed(self, val):
        v = int(float(val))
        # min が max を超えないよう制限
        max_v = int(self.max_var.get())
        if v > max_v:
            v = max_v
            self.min_var.set(v)
        self.min_label.config(text=str(v))
        self._update_range_label()
        set_param('intensity_min', v)

    def _on_max_changed(self, val):
        v = int(float(val))
        # max が min を下回らないよう制限
        min_v = int(self.min_var.get())
        if v < min_v:
            v = min_v
            self.max_var.set(v)
        self.max_label.config(text=str(v))
        self._update_range_label()
        set_param('intensity_max', v)

    def _update_range_label(self):
        lo = int(self.min_var.get())
        hi = int(self.max_var.get())
        self.range_label.config(text=f'range: {lo} ～ {hi}')


# ============================================================
#  エントリーポイント
# ============================================================

def main():
    root = tk.Tk()
    IntensityFilterGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
