# coding: utf-8
# Файл: coating_simulator/gui/results_gui/tab_inverse_task_panel.py
"""
Содержит класс InverseTaskPanelTab для вкладки "Обратная задача" в SettingsPanel.
"""
import tkinter as tk
from tkinter import ttk

class InverseTaskPanelTab(ttk.Frame):
    def __init__(self, master, settings_vars, callbacks, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.settings_vars = settings_vars
        self.callbacks = callbacks
        
        self.columnconfigure(0, weight=1)
        self._create_widgets()

    def _create_widgets(self):
        pady_label_frame = (5, 5)
        pady_widget_group_sep = (5, 2)
        pady_widget_internal = (1, 1)
        internal_frame_padding = (5, 5, 5, 5)

        inverse_problem_frame = ttk.LabelFrame(self, text="Обратная задача (по профилям)", padding=internal_frame_padding)
        inverse_problem_frame.pack(fill=tk.X, expand=False, pady=pady_label_frame)
        inverse_problem_frame.columnconfigure(1, weight=1)

        btn_load_profiles = ttk.Button(inverse_problem_frame, text="Загрузить профиль(и)",
                                       command=self.callbacks['load_profiles_callback'])
        btn_load_profiles.grid(row=0, column=0, columnspan=2, pady=pady_widget_internal, sticky=tk.EW)
        
        self.label_loaded_files = ttk.Label(inverse_problem_frame, 
                                            textvariable=self.settings_vars['loaded_files_text_var'],
                                            wraplength=220, justify=tk.LEFT, font=('TkDefaultFont', 8))
        self.label_loaded_files.grid(row=1, column=0, columnspan=2, pady=pady_widget_internal, sticky=tk.W)
        
        ttk.Label(inverse_problem_frame, text="Метод реконстр.:").grid(row=2, column=0, padx=(0,5), pady=pady_widget_internal, sticky=tk.W)
        self.combo_reconstruction = ttk.Combobox(inverse_problem_frame,
                                                 textvariable=self.settings_vars['reconstruction_method_var'],
                                                 values=["Линейная", "Кубический сплайн"],
                                                 state='readonly', width=14)
        self.combo_reconstruction.grid(row=2, column=1, padx=2, pady=pady_widget_internal, sticky=tk.EW)
        # Если нужен _on_settings_change
        # self.combo_reconstruction.bind("<<ComboboxSelected>>", self.callbacks['_on_settings_change'])


        self.btn_reconstruct_map = ttk.Button(inverse_problem_frame, text="Построить карту по профилям",
                                              command=self.callbacks['_internal_reconstruct_map_callback'])
        self.btn_reconstruct_map.grid(row=3, column=0, columnspan=2, pady=(pady_widget_group_sep[0],0), sticky=tk.EW)
