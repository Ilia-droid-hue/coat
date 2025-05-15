# coding: utf-8
# Файл: coating_simulator/gui/results_gui/tab_auto_uniformity_panel.py
"""
Содержит класс AutoUniformityPanelTab для вкладки "Авторавномерность" в SettingsPanel.
"""
import tkinter as tk
from tkinter import ttk

class AutoUniformityPanelTab(ttk.Frame):
    def __init__(self, master, settings_vars, callbacks, validation_commands, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.settings_vars = settings_vars
        self.callbacks = callbacks
        self.vcmd = validation_commands
        
        self.columnconfigure(0, weight=1)
        self._create_widgets()

    def _create_widgets(self):
        pady_label_frame = (5, 5)
        pady_widget_group_sep = (5, 2) # Сохраняем для консистентности
        pady_widget_internal = (1, 1)
        internal_frame_padding = (5, 5, 5, 5)
        
        auto_frame = ttk.LabelFrame(self, text="Авторавномерность", padding=internal_frame_padding)
        auto_frame.pack(fill=tk.X, expand=False, pady=pady_label_frame) 
        auto_frame.columnconfigure(1, weight=1)

        ttk.Label(auto_frame, text="Метод:").grid(row=0, column=0, padx=(0,5), pady=pady_widget_internal, sticky=tk.W)
        self.combo_auto_mode = ttk.Combobox(auto_frame, 
                                            textvariable=self.settings_vars['auto_uniformity_mode_var'],
                                            values=['Маска', 'Источник'], state='readonly', width=10)
        self.combo_auto_mode.grid(row=0, column=1, padx=2, pady=pady_widget_internal, sticky=tk.EW)
        # Если для этого комбобокса нужен _on_settings_change, его нужно добавить и передать
        # self.combo_auto_mode.bind("<<ComboboxSelected>>", self.callbacks['_on_settings_change'])


        ttk.Label(auto_frame, text="Высота маски (мм):").grid(row=1, column=0, padx=(0,5), pady=pady_widget_internal, sticky=tk.W)
        self.entry_mask_height = ttk.Entry(auto_frame, 
                                           textvariable=self.settings_vars['auto_uniformity_mask_height_var'],
                                           width=6, validate='key', 
                                           validatecommand=self.vcmd['float_positive_or_zero'])
        self.entry_mask_height.grid(row=1, column=1, padx=2, pady=pady_widget_internal, sticky=tk.EW)
        # Если нужен _on_settings_change_entry
        # self.entry_mask_height.bind("<KeyRelease>", self.callbacks['_on_settings_change_entry'])

        self.btn_calc_mask = ttk.Button(auto_frame, text="Рассчитать", 
                                        command=self.callbacks['calculate_mask_callback'])
        self.btn_calc_mask.grid(row=2, column=0, columnspan=2, pady=(pady_widget_group_sep[0],0), sticky=tk.EW)
