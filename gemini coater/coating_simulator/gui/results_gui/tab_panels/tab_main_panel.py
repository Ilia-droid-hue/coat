# coding: utf-8
# Файл: coating_simulator/gui/results_gui/tab_panels/tab_main_panel.py
"""
Содержит класс MainPanelTab для основной вкладки в SettingsPanel.
Поля ROI и Colormap перенесены вправо от чекбоксов.
"""
import tkinter as tk
from tkinter import ttk

class MainPanelTab(ttk.Frame):
    def __init__(self, master, settings_vars, callbacks, validation_commands, helper_methods, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.settings_vars = settings_vars
        self.callbacks = callbacks
        self.vcmd = validation_commands
        self.helpers = helper_methods
        
        self.config_const = self.helpers.get('config_constants', {})
        self.TARGET_DISK = self.config_const.get('TARGET_DISK', "диск (FALLBACK)")
        self.TARGET_DOME = self.config_const.get('TARGET_DOME', "купол (FALLBACK)")
        self.TARGET_PLANETARY = self.config_const.get('TARGET_PLANETARY', "планетарный (FALLBACK)")
        self.TARGET_LINEAR = self.config_const.get('TARGET_LINEAR', "линейное перемещение (FALLBACK)")

        self.current_target_type = self.TARGET_DISK 

        self.columnconfigure(0, weight=1) 
        self.rowconfigure(0, weight=0) 
        self.rowconfigure(1, weight=0) 
        self.rowconfigure(2, weight=0) 
        self.rowconfigure(3, weight=0) 
        self.rowconfigure(4, weight=0) 

        self._create_widgets()
        self.update_all_field_states()

    def _is_scipy_available(self):
        if 'is_scipy_available' in self.helpers:
            return self.helpers['is_scipy_available']()
        try: from scipy.signal import savgol_filter; return True # noqa
        except ImportError: return False

    def update_all_field_states(self):
        self.update_linear_profile_fields_state()
        self.update_roi_fields_visibility_and_state(self.current_target_type) 
        self.update_cmap_range_fields_state()
        self.update_idletasks() 

    def _toggle_roi_visibility(self):
        self.update_roi_fields_visibility_and_state(self.current_target_type)
        if self.callbacks.get('_on_settings_change'):
            self.callbacks['_on_settings_change']() 
        self.update_idletasks() 

    def _toggle_cmap_visibility(self):
        self.update_cmap_range_fields_state()
        if self.callbacks.get('_on_settings_change'):
            self.callbacks['_on_settings_change']()
        self.update_idletasks() 

    def _create_widgets(self):
        pady_label_frame = (5, 5)
        pady_widget_internal = (3, 3)
        internal_frame_padding = (5, 5, 5, 5)
        label_padx = (0, 10) 
        entry_padx = (0, 5) 
        
        entry_width_short = 7
        entry_width_combo = 20

        current_main_row = 0 

        # --- Секция: Расчет равномерности U (по профилю) ---
        method_frame = ttk.LabelFrame(self, text="Расчет равномерности U (по профилю)", padding=internal_frame_padding)
        method_frame.grid(row=current_main_row, column=0, sticky=tk.EW, pady=pady_label_frame); current_main_row += 1
        method_frame.columnconfigure(0, weight=0) 
        method_frame.columnconfigure(1, weight=1) 
        
        ttk.Label(method_frame, text="Метод:").grid(row=0, column=0, padx=label_padx, pady=pady_widget_internal, sticky=tk.W)
        combo_method_uniformity = ttk.Combobox(method_frame, textvariable=self.settings_vars['uniformity_method_var'],
                                               values=list(self.settings_vars['uniformity_formulas'].keys()),
                                               state='readonly', width=entry_width_short) 
        combo_method_uniformity.grid(row=0, column=1, padx=entry_padx, pady=pady_widget_internal, sticky=tk.W) 
        combo_method_uniformity.bind("<<ComboboxSelected>>", self.callbacks['_on_settings_change'])
        
        self.label_formula_display = ttk.Label(method_frame, textvariable=self.settings_vars['selected_formula_text_var'],
                                               font=('TkDefaultFont', 8), foreground="grey", wraplength=230)
        self.label_formula_display.grid(row=1, column=0, columnspan=2, padx=label_padx, pady=(0, 3), sticky=tk.W)
        
        stats_profile_frame = ttk.Frame(method_frame)
        stats_profile_frame.grid(row=2, column=0, columnspan=2, sticky=tk.EW, pady=(3,1))
        stats_profile_frame.columnconfigure(0, weight=1, uniform="stats_col")
        stats_profile_frame.columnconfigure(1, weight=1, uniform="stats_col")
        stats_profile_frame.columnconfigure(2, weight=1, uniform="stats_col")
        self.label_t_max = ttk.Label(stats_profile_frame, textvariable=self.settings_vars['profile_t_max_var'], font=('TkDefaultFont', 8))
        self.label_t_max.grid(row=0, column=0, sticky=tk.W, padx=(0,2))
        self.label_t_min = ttk.Label(stats_profile_frame, textvariable=self.settings_vars['profile_t_min_var'], font=('TkDefaultFont', 8))
        self.label_t_min.grid(row=0, column=1, sticky=tk.W, padx=2)
        self.label_t_mean = ttk.Label(stats_profile_frame, textvariable=self.settings_vars['profile_t_mean_var'], font=('TkDefaultFont', 8))
        self.label_t_mean.grid(row=0, column=2, sticky=tk.W, padx=(2,0))

        # --- Секция: Настройки профиля для U и вида ---
        self.profile_config_outer_frame = ttk.LabelFrame(self, text="Настройки профиля для U и вида", padding=internal_frame_padding)
        self.profile_config_outer_frame.grid(row=current_main_row, column=0, sticky=tk.EW, pady=pady_label_frame); current_main_row += 1
        self.profile_config_outer_frame.columnconfigure(0, weight=0) 
        self.profile_config_outer_frame.columnconfigure(1, weight=1) 

        self.label_num_circular_profiles = ttk.Label(self.profile_config_outer_frame, text="Профилей (круг, 1-6):")
        self.entry_num_circular_profiles = ttk.Entry(
            self.profile_config_outer_frame, textvariable=self.settings_vars['num_circular_profiles_var'],
            width=entry_width_short, validate='key', validatecommand=self.vcmd['num_circular_profiles']
        )
        self.entry_num_circular_profiles.bind("<KeyRelease>", self.callbacks['_on_settings_change_entry'])

        self.linear_profile_type_radios_frame = ttk.Frame(self.profile_config_outer_frame)
        self.linear_profile_type_radios = {}
        radio_options = [("Горизонт. (X)", "horizontal"), ("Вертикал. (Y)", "vertical"), ("Оба", "both")]
        for i, (text, val) in enumerate(radio_options):
            rb = ttk.Radiobutton(self.linear_profile_type_radios_frame, text=text,
                                 variable=self.settings_vars['linear_profile_type_var'], value=val,
                                 command=self.callbacks['_on_settings_change']) 
            rb.pack(side=tk.LEFT, padx=(0,10), anchor=tk.W)
            self.linear_profile_type_radios[val] = rb
            
        self.label_linear_x_offset = ttk.Label(self.profile_config_outer_frame, text="Y для X-проф. (мм):")
        self.entry_linear_x_offset = ttk.Entry(
            self.profile_config_outer_frame, textvariable=self.settings_vars['linear_x_profile_y_offset_var'],
            width=entry_width_short, validate='key', validatecommand=self.vcmd['float_general']
        )
        self.label_linear_y_offset = ttk.Label(self.profile_config_outer_frame, text="X для Y-проф. (мм):")
        self.entry_linear_y_offset = ttk.Entry(
            self.profile_config_outer_frame, textvariable=self.settings_vars['linear_y_profile_x_offset_var'],
            width=entry_width_short, validate='key', validatecommand=self.vcmd['float_general']
        )
        self.entry_linear_x_offset.bind("<KeyRelease>", self.callbacks['_on_settings_change_entry'])
        self.entry_linear_y_offset.bind("<KeyRelease>", self.callbacks['_on_settings_change_entry'])
        
        self.smoothing_label = ttk.Label(self.profile_config_outer_frame, text="Сглаживание:")
        smoothing_options = ["Без сглаживания", "Полином. аппрокс."]
        if self._is_scipy_available(): 
            smoothing_options.insert(0, "Savitzky-Golay")
            self.settings_vars['smoothing_method_var'].set("Savitzky-Golay")
        else:
            self.settings_vars['smoothing_method_var'].set("Полином. аппрокс.")
        
        self.combo_smoothing = ttk.Combobox(self.profile_config_outer_frame, 
                                             textvariable=self.settings_vars['smoothing_method_var'],
                                             values=smoothing_options, state='readonly', width=entry_width_combo)
        self.combo_smoothing.bind("<<ComboboxSelected>>", self.callbacks['_on_settings_change'])

        self.savgol_frame = ttk.Frame(self.profile_config_outer_frame)
        self.savgol_frame.columnconfigure((0,2), weight=0) 
        self.savgol_frame.columnconfigure((1,3), weight=0) 
        ttk.Label(self.savgol_frame, text="Окно:").grid(row=0, column=0, padx=(0,2), pady=pady_widget_internal, sticky=tk.W)
        self.entry_savgol_window = ttk.Entry(self.savgol_frame, textvariable=self.settings_vars['savgol_window_var'],
                                             width=entry_width_short-2, validate='key', validatecommand=self.vcmd['digits_only']) 
        self.entry_savgol_window.grid(row=0, column=1, padx=(0,10), pady=pady_widget_internal, sticky=tk.W) 
        ttk.Label(self.savgol_frame, text="Полином:").grid(row=0, column=2, padx=(0,2), pady=pady_widget_internal, sticky=tk.W)
        self.entry_savgol_polyorder = ttk.Entry(self.savgol_frame, textvariable=self.settings_vars['savgol_polyorder_var'],
                                                width=entry_width_short-2, validate='key', validatecommand=self.vcmd['digits_only']) 
        self.entry_savgol_polyorder.grid(row=0, column=3, padx=(0,0), pady=pady_widget_internal, sticky=tk.W) 
        self.entry_savgol_window.bind("<KeyRelease>", self.callbacks['_on_settings_change_entry'])
        self.entry_savgol_polyorder.bind("<KeyRelease>", self.callbacks['_on_settings_change_entry'])

        self.label_polyfit_degree = ttk.Label(self.profile_config_outer_frame, text="Степень полин.:")
        self.entry_polyfit_degree = ttk.Entry(self.profile_config_outer_frame, 
                                              textvariable=self.settings_vars['polyfit_degree_var'],
                                              width=entry_width_short, validate='key', 
                                              validatecommand=self.vcmd['digits_only'])
        self.entry_polyfit_degree.bind("<KeyRelease>", self.callbacks['_on_settings_change_entry'])
        
        self.chk_raw_profile = ttk.Checkbutton(self.profile_config_outer_frame, text="Показать сырой профиль",
                                                variable=self.settings_vars['show_raw_profile_var'],
                                                command=self.callbacks['recalculate_callback'])
        self.chk_percent = ttk.Checkbutton(self.profile_config_outer_frame, text="Покрытие в %",
                                            variable=self.settings_vars['display_percent_var'],
                                            command=self.callbacks['recalculate_callback'])
        self.chk_logscale = ttk.Checkbutton(self.profile_config_outer_frame, text="Лог. шкала (карта)",
                                             variable=self.settings_vars['use_logscale_var'],
                                             command=self.callbacks['recalculate_callback'])

        # --- Секция: Зона интереса для U (ROI) ---
        self.roi_frame = ttk.LabelFrame(self, text="Зона интереса для U", padding=internal_frame_padding)
        self.roi_frame.grid(row=current_main_row, column=0, sticky=tk.EW, pady=pady_label_frame); current_main_row += 1
        # Конфигурация колонок для roi_frame: 0-чекбокс, 1-метка, 2-поле ввода
        self.roi_frame.columnconfigure(0, weight=0) 
        self.roi_frame.columnconfigure(1, weight=0) # Для меток полей ввода
        self.roi_frame.columnconfigure(2, weight=1) # Для полей ввода

        self.chk_roi_enabled = ttk.Checkbutton(self.roi_frame, text="Учитывать зону U",
                                               variable=self.settings_vars['roi_enabled_var'],
                                               command=self._toggle_roi_visibility) 
        self.chk_roi_enabled.grid(row=0, column=0, sticky=tk.W, padx=label_padx, pady=pady_widget_internal)
        
        self.chk_roi_show_on_map = ttk.Checkbutton(self.roi_frame, text="Отображать зону", 
                                                   variable=self.settings_vars['roi_show_on_map_var'],
                                                   command=self.callbacks['recalculate_callback']) 
        self.chk_roi_show_on_map.grid(row=1, column=0, sticky=tk.W, padx=label_padx, pady=pady_widget_internal)

        # Фреймы для полей ввода ROI больше не нужны как отдельные контейнеры для grid/grid_remove
        # Мы будем размещать метки и поля ввода прямо в self.roi_frame
        self.label_roi_d_min = ttk.Label(self.roi_frame, text="Dmin (мм):")
        self.entry_roi_d_min = ttk.Entry(self.roi_frame, textvariable=self.settings_vars['roi_d_min_var'], width=entry_width_short, validate='key', validatecommand=self.vcmd['digits_or_empty'])
        self.entry_roi_d_min.bind("<KeyRelease>", self.callbacks['_on_settings_change_entry'])
        
        self.label_roi_d_max = ttk.Label(self.roi_frame, text="Dmax (мм):")
        self.entry_roi_d_max = ttk.Entry(self.roi_frame, textvariable=self.settings_vars['roi_d_max_var'], width=entry_width_short, validate='key', validatecommand=self.vcmd['digits_or_empty'])
        self.entry_roi_d_max.bind("<KeyRelease>", self.callbacks['_on_settings_change_entry'])

        self.label_roi_width_x = ttk.Label(self.roi_frame, text="Ширина X (мм):")
        self.entry_roi_width_x = ttk.Entry(self.roi_frame, textvariable=self.settings_vars['roi_width_x_var'], width=entry_width_short, validate='key', validatecommand=self.vcmd['digits_or_empty'])
        self.entry_roi_width_x.bind("<KeyRelease>", self.callbacks['_on_settings_change_entry'])

        self.label_roi_height_y = ttk.Label(self.roi_frame, text="Высота Y (мм):")
        self.entry_roi_height_y = ttk.Entry(self.roi_frame, textvariable=self.settings_vars['roi_height_y_var'], width=entry_width_short, validate='key', validatecommand=self.vcmd['digits_or_empty'])
        self.entry_roi_height_y.bind("<KeyRelease>", self.callbacks['_on_settings_change_entry'])

        self.label_roi_offset_x = ttk.Label(self.roi_frame, text="X₀ (мм):")
        self.entry_roi_offset_x = ttk.Entry(self.roi_frame, textvariable=self.settings_vars['roi_offset_x_var'], width=entry_width_short, validate='key', validatecommand=self.vcmd['float_or_empty'])
        self.entry_roi_offset_x.bind("<KeyRelease>", self.callbacks['_on_settings_change_entry'])

        self.label_roi_offset_y = ttk.Label(self.roi_frame, text="Y₀ (мм):")
        self.entry_roi_offset_y = ttk.Entry(self.roi_frame, textvariable=self.settings_vars['roi_offset_y_var'], width=entry_width_short, validate='key', validatecommand=self.vcmd['float_or_empty'])
        self.entry_roi_offset_y.bind("<KeyRelease>", self.callbacks['_on_settings_change_entry'])
        
        # --- Секция: Настройки цветовой карты ---
        self.cmap_frame = ttk.LabelFrame(self, text="Настройки цветовой карты", padding=internal_frame_padding)
        self.cmap_frame.grid(row=current_main_row, column=0, sticky=tk.EW, pady=pady_label_frame); current_main_row += 1
        self.cmap_frame.columnconfigure(0, weight=0)
        self.cmap_frame.columnconfigure(1, weight=0) # Для меток полей ввода
        self.cmap_frame.columnconfigure(2, weight=1) # Для полей ввода

        self.chk_cmap_manual_range = ttk.Checkbutton(self.cmap_frame, text="Ручной диапазон", 
                                                     variable=self.settings_vars['cmap_manual_range_var'],
                                                     command=self._toggle_cmap_visibility) 
        self.chk_cmap_manual_range.grid(row=0, column=0, sticky=tk.W, padx=label_padx, pady=pady_widget_internal)
        
        self.label_cmap_min = ttk.Label(self.cmap_frame, text="Мин. знач.:")
        self.entry_cmap_min_val = ttk.Entry(self.cmap_frame, textvariable=self.settings_vars['cmap_min_val_var'], width=entry_width_short, validate='key', validatecommand=self.vcmd['float_or_empty'])
        self.entry_cmap_min_val.bind("<KeyRelease>", self.callbacks['_on_settings_change_entry'])
        
        self.label_cmap_max = ttk.Label(self.cmap_frame, text="Макс. знач.:")
        self.entry_cmap_max_val = ttk.Entry(self.cmap_frame, textvariable=self.settings_vars['cmap_max_val_var'], width=entry_width_short, validate='key', validatecommand=self.vcmd['float_or_empty'])
        self.entry_cmap_max_val.bind("<KeyRelease>", self.callbacks['_on_settings_change_entry'])
        
        # --- Панель управления (кнопки) ---
        control_panel = ttk.Frame(self, padding=(0, 5, 0, 0))
        control_panel.grid(row=current_main_row, column=0, sticky=tk.EW, pady=pady_label_frame); current_main_row += 1
        control_panel.columnconfigure((0, 1), weight=1, uniform="controlgroup")
        btn_update = ttk.Button(control_panel, text="Обновить графики", command=self.callbacks['recalculate_callback'])
        btn_update.grid(row=0, column=0, padx=(0,2), pady=(1,0), sticky=tk.EW)
        
        self.btn_export_excel = ttk.Button(control_panel, text="Экспорт в Excel",
                                           command=self.callbacks['export_excel_callback'], state=tk.DISABLED)
        self.btn_export_excel.grid(row=0, column=1, padx=(2,0), pady=(1,0), sticky=tk.EW)
        
        self.update_profile_options_layout(self.TARGET_DISK) 
        self.update_cmap_range_fields_state() 

    def update_roi_fields_visibility_and_state(self, target_type):
        current_target_for_roi = target_type if target_type is not None else self.current_target_type
        roi_is_enabled = self.settings_vars['roi_enabled_var'].get()
        
        chk_show_state = tk.NORMAL if roi_is_enabled else tk.DISABLED
        if hasattr(self, 'chk_roi_show_on_map'): 
            self.chk_roi_show_on_map.config(state=chk_show_state)
            if not roi_is_enabled:
                self.settings_vars['roi_show_on_map_var'].set(False)

        is_circular_roi_type = (current_target_for_roi in [self.TARGET_DISK, self.TARGET_DOME, self.TARGET_PLANETARY])
        is_rectangular_roi_type = (current_target_for_roi == self.TARGET_LINEAR)

        # Сначала скрываем все поля ROI
        for widget in [self.label_roi_d_min, self.entry_roi_d_min, self.label_roi_d_max, self.entry_roi_d_max,
                       self.label_roi_width_x, self.entry_roi_width_x, self.label_roi_height_y, self.entry_roi_height_y,
                       self.label_roi_offset_x, self.entry_roi_offset_x, self.label_roi_offset_y, self.entry_roi_offset_y]:
            if hasattr(self, widget.winfo_name()): # Проверяем существование виджета
                 widget.grid_remove()

        current_roi_row = 0 # Начинаем с первой доступной строки под чекбоксами в roi_frame
        
        # Поля для чекбокса "Учитывать зону U" и "Отображать зону" уже размещены в _create_widgets
        # Их строки (0 и 1) не меняются. Поля ввода начинаются с row=0 *внутри* их контейнера.
        # Мы будем размещать метки и поля в колонках 1 и 2 самого roi_frame, начиная с row=0
        
        if roi_is_enabled:
            if is_circular_roi_type:
                self.label_roi_d_min.grid(in_=self.roi_frame, row=0, column=1, sticky=tk.W, padx=(10,5), pady=(0,0)) # Отступ слева от чекбокса
                self.entry_roi_d_min.grid(in_=self.roi_frame, row=0, column=2, sticky=tk.W, padx=(0,5), pady=(0,0))
                self.entry_roi_d_min.config(state=tk.NORMAL)

                self.label_roi_d_max.grid(in_=self.roi_frame, row=1, column=1, sticky=tk.W, padx=(10,5), pady=(0,0))
                self.entry_roi_d_max.grid(in_=self.roi_frame, row=1, column=2, sticky=tk.W, padx=(0,5), pady=(0,0))
                self.entry_roi_d_max.config(state=tk.NORMAL)
            elif is_rectangular_roi_type:
                # Размещаем метки и поля для прямоугольного ROI
                # Эти виджеты будут размещаться, начиная с row=0, column=1 и column=2 в self.roi_frame
                # Чтобы они были справа от чекбоксов, чекбоксы должны быть в column=0
                self.label_roi_width_x.grid(in_=self.roi_frame, row=0, column=1, sticky=tk.W, padx=(10,5), pady=(0,0))
                self.entry_roi_width_x.grid(in_=self.roi_frame, row=0, column=2, sticky=tk.W, padx=(0,5), pady=(0,0))
                self.entry_roi_width_x.config(state=tk.NORMAL)

                self.label_roi_height_y.grid(in_=self.roi_frame, row=1, column=1, sticky=tk.W, padx=(10,5), pady=(0,0))
                self.entry_roi_height_y.grid(in_=self.roi_frame, row=1, column=2, sticky=tk.W, padx=(0,5), pady=(0,0))
                self.entry_roi_height_y.config(state=tk.NORMAL)
                
                # Для смещений можно использовать следующие строки (2 и 3) в колонках 1 и 2
                self.label_roi_offset_x.grid(in_=self.roi_frame, row=2, column=1, sticky=tk.W, padx=(10,5), pady=(0,0))
                self.entry_roi_offset_x.grid(in_=self.roi_frame, row=2, column=2, sticky=tk.W, padx=(0,5), pady=(0,0))
                self.entry_roi_offset_x.config(state=tk.NORMAL)

                self.label_roi_offset_y.grid(in_=self.roi_frame, row=3, column=1, sticky=tk.W, padx=(10,5), pady=(0,0))
                self.entry_roi_offset_y.grid(in_=self.roi_frame, row=3, column=2, sticky=tk.W, padx=(0,5), pady=(0,0))
                self.entry_roi_offset_y.config(state=tk.NORMAL)
        else: # roi_is_enabled is False
            # Просто деактивируем все поля, так как они уже скрыты
            for entry_widget in [self.entry_roi_d_min, self.entry_roi_d_max, self.entry_roi_width_x, 
                                 self.entry_roi_height_y, self.entry_roi_offset_x, self.entry_roi_offset_y]:
                if hasattr(self, entry_widget.winfo_name()):
                    entry_widget.config(state=tk.DISABLED)
        
        if hasattr(self, 'roi_frame'): 
            self.roi_frame.update_idletasks()


    def update_cmap_range_fields_state(self):
        is_manual = self.settings_vars['cmap_manual_range_var'].get()
        fields_state = tk.NORMAL if is_manual else tk.DISABLED
        
        # Сначала скрываем метки и поля
        if hasattr(self, 'label_cmap_min'): self.label_cmap_min.grid_remove()
        if hasattr(self, 'entry_cmap_min_val'): self.entry_cmap_min_val.grid_remove()
        if hasattr(self, 'label_cmap_max'): self.label_cmap_max.grid_remove()
        if hasattr(self, 'entry_cmap_max_val'): self.entry_cmap_max_val.grid_remove()

        if is_manual:
            # Размещаем метки и поля справа от чекбокса "Ручной диапазон"
            # Чекбокс в row=0, column=0. Поля будут в row=0, col=1,2 и row=1, col=1,2
            if hasattr(self, 'label_cmap_min'):
                self.label_cmap_min.grid(in_=self.cmap_frame, row=0, column=1, sticky=tk.W, padx=(10,5), pady=(0,0))
            if hasattr(self, 'entry_cmap_min_val'):
                self.entry_cmap_min_val.grid(in_=self.cmap_frame, row=0, column=2, sticky=tk.W, padx=(0,5), pady=(0,0))
                self.entry_cmap_min_val.config(state=fields_state)
            
            if hasattr(self, 'label_cmap_max'):
                self.label_cmap_max.grid(in_=self.cmap_frame, row=1, column=1, sticky=tk.W, padx=(10,5), pady=(0,0))
            if hasattr(self, 'entry_cmap_max_val'):
                self.entry_cmap_max_val.grid(in_=self.cmap_frame, row=1, column=2, sticky=tk.W, padx=(0,5), pady=(0,0))
                self.entry_cmap_max_val.config(state=fields_state)
        else: # Если не ручной, деактивируем (они уже скрыты)
            if hasattr(self, 'entry_cmap_min_val'): self.entry_cmap_min_val.config(state=tk.DISABLED)
            if hasattr(self, 'entry_cmap_max_val'): self.entry_cmap_max_val.config(state=tk.DISABLED)


        if hasattr(self, 'cmap_frame'): 
            self.cmap_frame.update_idletasks()

    def update_linear_profile_fields_state(self):
        # ... (без изменений) ...
        profile_type = self.settings_vars['linear_profile_type_var'].get()
        state_x_offset = tk.DISABLED; state_y_offset = tk.DISABLED
        if profile_type == "horizontal": state_x_offset = tk.NORMAL
        elif profile_type == "vertical": state_y_offset = tk.NORMAL
        elif profile_type == "both": state_x_offset = tk.NORMAL; state_y_offset = tk.NORMAL
        if hasattr(self, 'entry_linear_x_offset'): self.entry_linear_x_offset.config(state=state_x_offset)
        if hasattr(self, 'entry_linear_y_offset'): self.entry_linear_y_offset.config(state=state_y_offset)

    def update_profile_options_layout(self, target_type: str | None):
        # ... (логика размещения элементов профиля и сглаживания остается) ...
        self.current_target_type = target_type 
        pady_widget_internal = (3, 3) 
        pady_group_separator = (5, 3) 
        current_row_in_profile_config_frame = 0 
        label_padx = (0, 10) 
        entry_padx = (0, 5)
        checkbox_padx = (0,0) 

        for widget_list in [
            (self.label_num_circular_profiles, self.entry_num_circular_profiles),
            (getattr(self, 'linear_profile_type_radios_frame', None),),
            (getattr(self, 'label_linear_x_offset', None), getattr(self, 'entry_linear_x_offset', None)),
            (getattr(self, 'label_linear_y_offset', None), getattr(self, 'entry_linear_y_offset', None)),
            (getattr(self, 'savgol_frame', None),),
            (getattr(self, 'label_polyfit_degree', None), getattr(self, 'entry_polyfit_degree', None))
        ]:
            for widget in widget_list:
                if widget and hasattr(widget, 'grid_remove'): widget.grid_remove()
        
        if target_type in [self.TARGET_DISK, self.TARGET_DOME, self.TARGET_PLANETARY]:
            self.label_num_circular_profiles.grid(in_=self.profile_config_outer_frame, row=current_row_in_profile_config_frame, column=0, padx=label_padx, pady=pady_widget_internal, sticky=tk.W)
            self.entry_num_circular_profiles.grid(in_=self.profile_config_outer_frame, row=current_row_in_profile_config_frame, column=1, padx=entry_padx, pady=pady_widget_internal, sticky=tk.W) 
            current_row_in_profile_config_frame += 1
        elif target_type == self.TARGET_LINEAR:
            ttk.Label(self.profile_config_outer_frame, text="Тип профиля:").grid(in_=self.profile_config_outer_frame, row=current_row_in_profile_config_frame, column=0, padx=label_padx, pady=pady_widget_internal, sticky=tk.W)
            if hasattr(self, 'linear_profile_type_radios_frame'): self.linear_profile_type_radios_frame.grid(in_=self.profile_config_outer_frame, row=current_row_in_profile_config_frame, column=1, padx=entry_padx, pady=pady_widget_internal, sticky=tk.W)
            current_row_in_profile_config_frame +=1
            self.label_linear_x_offset.grid(in_=self.profile_config_outer_frame, row=current_row_in_profile_config_frame, column=0, padx=label_padx, pady=pady_widget_internal, sticky=tk.W)
            self.entry_linear_x_offset.grid(in_=self.profile_config_outer_frame, row=current_row_in_profile_config_frame, column=1, padx=entry_padx, pady=pady_widget_internal, sticky=tk.W) 
            current_row_in_profile_config_frame +=1 
            self.label_linear_y_offset.grid(in_=self.profile_config_outer_frame, row=current_row_in_profile_config_frame, column=0, padx=label_padx, pady=pady_widget_internal, sticky=tk.W)
            self.entry_linear_y_offset.grid(in_=self.profile_config_outer_frame, row=current_row_in_profile_config_frame, column=1, padx=entry_padx, pady=pady_widget_internal, sticky=tk.W) 
            current_row_in_profile_config_frame +=1
        
        self.update_linear_profile_fields_state()
        self.smoothing_label.grid(in_=self.profile_config_outer_frame, row=current_row_in_profile_config_frame, column=0, padx=label_padx, pady=(pady_group_separator[0], pady_widget_internal[1]), sticky=tk.W)
        self.combo_smoothing.grid(in_=self.profile_config_outer_frame, row=current_row_in_profile_config_frame, column=1, padx=entry_padx, pady=(pady_group_separator[0], pady_widget_internal[1]), sticky=tk.W)
        current_row_in_profile_config_frame += 1 
        
        smoothing_param_frame_gridded = False
        if '_update_smoothing_options_display_internal' in self.helpers:
            smoothing_param_frame_gridded = self.helpers['_update_smoothing_options_display_internal'](
                self.profile_config_outer_frame, 
                self.savgol_frame, 
                self.label_polyfit_degree, 
                self.entry_polyfit_degree, 
                self.settings_vars['smoothing_method_var'].get(),
                current_row_in_profile_config_frame 
            )
        if smoothing_param_frame_gridded: current_row_in_profile_config_frame += 1 
        
        self.chk_raw_profile.grid(in_=self.profile_config_outer_frame, row=current_row_in_profile_config_frame, column=0, columnspan=2, padx=checkbox_padx, pady=(pady_group_separator[0] if not smoothing_param_frame_gridded else pady_widget_internal[0], pady_widget_internal[1]), sticky=tk.W)
        current_row_in_profile_config_frame += 1
        self.chk_percent.grid(in_=self.profile_config_outer_frame, row=current_row_in_profile_config_frame, column=0, columnspan=2, padx=checkbox_padx, pady=pady_widget_internal, sticky=tk.W)
        current_row_in_profile_config_frame += 1
        self.chk_logscale.grid(in_=self.profile_config_outer_frame, row=current_row_in_profile_config_frame, column=0, columnspan=2, padx=checkbox_padx, pady=pady_widget_internal, sticky=tk.W)

        self.update_roi_fields_visibility_and_state(target_type) 
        self.update_cmap_range_fields_state() # Обновляем состояние полей cmap при смене типа мишени
        self.update_idletasks()
