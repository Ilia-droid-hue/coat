# coating_simulator_project/coating_simulator/gui/frames/control.py
"""
Frame containing control buttons for the simulation.
Фрейм, содержащий кнопки управления симуляцией.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import traceback

class ControlFrame(ttk.LabelFrame):
    """
    LabelFrame containing control buttons like Run, Export, Show Params, Show Geometry.
    LabelFrame, содержащий кнопки управления, такие как Запуск, Экспорт, Показать параметры, Показать геометрию.
    """
    def __init__(self, master, run_callback, export_callback, show_params_callback, show_geometry_callback, **kwargs): # <<< Добавлен show_geometry_callback
        """
        Initializes the ControlFrame.
        Инициализирует ControlFrame.

        Args:
            master: Parent widget. Родительский виджет.
            run_callback: Function to call when the 'Run' button is pressed.
                          Функция, вызываемая при нажатии кнопки 'Запустить'.
            export_callback: Function to call when the 'Export CSV' button is pressed.
                             Функция, вызываемая при нажатии кнопки 'Экспорт CSV'.
            show_params_callback: Function to call when the 'Show Params' button is pressed.
                                  Функция, вызываемая при нажатии кнопки 'Показать параметры'.
            show_geometry_callback: Function to call when 'Show Geometry' is pressed. # <<< Новый аргумент
                                    Функция, вызываемая при нажатии 'Показать геометрию'.
            **kwargs: Additional arguments for ttk.LabelFrame.
                      Дополнительные аргументы для ttk.LabelFrame.
        """
        super().__init__(master, text="Управление", **kwargs)
        self.columnconfigure((0, 1), weight=1)

        # Store callbacks
        self._run_simulation = run_callback
        self._export_csv = export_callback
        self._show_params = show_params_callback
        self._show_geometry = show_geometry_callback # <<< Сохраняем новый колбэк

        # --- Widgets ---
        self.btn_run = ttk.Button(self, text="Запустить", command=self._safe_run_callback)
        self.btn_export = ttk.Button(self, text="Экспорт CSV", command=self._safe_export_callback, state=tk.DISABLED)
        self.btn_check = ttk.Button(self, text="Показать параметры", command=self._safe_show_params_callback)
        self.btn_show_geometry = ttk.Button(self, text="Показать геометрию", command=self._safe_show_geometry_callback) # <<< Новая кнопка

        # --- Layout ---
        # Изменим немного компоновку, чтобы разместить 4 кнопки
        self.btn_run.grid(row=0, column=0, sticky=tk.EW, padx=5, pady=(5, 2))
        self.btn_export.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=(5, 2))
        self.btn_check.grid(row=1, column=0, sticky=tk.EW, padx=5, pady=(2, 5))
        self.btn_show_geometry.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=(2, 5)) # <<< Размещение новой кнопки

    def _safe_run_callback(self):
        if self._run_simulation:
            try:
                self._run_simulation()
            except Exception as e:
                messagebox.showerror("Ошибка Запуска", f"Произошла ошибка при запуске симуляции:\n{e}", parent=self)
                traceback.print_exc()

    def _safe_export_callback(self):
        if self._export_csv:
             try:
                 self._export_csv()
             except Exception as e:
                 messagebox.showerror("Ошибка Экспорта", f"Произошла ошибка при экспорте CSV:\n{e}", parent=self)
                 traceback.print_exc()

    def _safe_show_params_callback(self):
        if self._show_params:
             try:
                 self._show_params()
             except Exception as e:
                 messagebox.showerror("Ошибка Параметров", f"Произошла ошибка при показе параметров:\n{e}", parent=self)
                 traceback.print_exc()

    def _safe_show_geometry_callback(self): # <<< Новый безопасный колбэк
        if self._show_geometry:
            try:
                self._show_geometry()
            except Exception as e:
                messagebox.showerror("Ошибка Геометрии", f"Произошла ошибка при отображении геометрии:\n{e}", parent=self)
                traceback.print_exc()

    def enable_export_button(self):
        self.btn_export.config(state=tk.NORMAL)

    def disable_export_button(self):
        self.btn_export.config(state=tk.DISABLED)

    def enable_run_button(self):
        self.btn_run.config(state=tk.NORMAL)

    def disable_run_button(self):
        self.btn_run.config(state=tk.DISABLED)

