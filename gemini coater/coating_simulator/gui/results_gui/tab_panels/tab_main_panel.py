# coding: utf-8
# Файл: coating_simulator/gui/results_gui/tab_panels/tab_main_panel.py
"""
Содержит класс MainPanelTab для основной вкладки в SettingsPanel.
"""
import tkinter as tk
from tkinter import ttk

# УБРАН блок try-except для импорта TARGET_*, они будут получены из helpers

class MainPanelTab(ttk.Frame):
    def __init__(self, master, settings_vars, callbacks, validation_commands, helper_methods, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.settings_vars = settings_vars
        self.callbacks = callbacks
        self.vcmd = validation_commands
        self.helpers = helper_methods
        
        # Получаем константы из helpers, переданных из SettingsPanel
        self.config_const = self.helpers.get('config_constants', {})
        self.TARGET_DISK = self.config_const.get('TARGET_DISK', "диск (FALLBACK)") # Фоллбэк на случай отсутствия
        self.TARGET_DOME = self.config_const.get('TARGET_DOME', "купол (FALLBACK)")
        self.TARGET_PLANETARY = self.config_const.get('TARGET_PLANETARY', "планетарный (FALLBACK)")
        self.TARGET_LINEAR = self.config_const.get('TARGET_LINEAR', "линейное перемещение (FALLBACK)")

        self.current_target_type = self.TARGET_DISK # Инициализируем значением по умолчанию

        self.columnconfigure(0, weight=1) 
        self._create_widgets()
        
        # Устанавливаем начальное состояние после создания всех виджетов
        self.update_all_field_states()


    def _is_scipy_available(self):
        if 'is_scipy_available' in self.helpers:
            return self.helpers['is_scipy_available']()
        try: from scipy.signal import savgol_filter; return True # noqa
        except ImportError: return False

    def update_all_field_states(self):
        """Обновляет состояние всех динамических полей на вкладке."""
        self.update_linear_profile_fields_state()
        # Передаем current_target_type, который уже должен быть установлен
        self.update_roi_fields_visibility_and_state(self.current_target_type) 
        self.update_cmap_range_fields_state()

    def _create_widgets(self):
        pady_label_frame = (5, 5)
        pady_widget_group_sep = (5, 2) 
        pady_widget_internal = (1, 1)
        internal_frame_padding = (5, 5, 5, 5)
        label_padx = (0, 5) 
        entry_padx = (0, 5) 
        entry_width_num_profiles = 5
        entry_width_roi_circular = 6
        entry_width_roi_rectangular = 6
        entry_width_cmap = 6

        current_main_row = 0 

        # --- Секция: Расчет равномерности U (по профилю) ---
        method_frame = ttk.LabelFrame(self, text="Расчет равномерности U (по профилю)", padding=internal_frame_padding)
        method_frame.grid(row=current_main_row, column=0, sticky=tk.EW, pady=pady_label_frame); current_main_row += 1
        method_frame.columnconfigure(1, weight=1)
        
        ttk.Label(method_frame, text="Метод:").grid(row=0, column=0, padx=label_padx, pady=pady_widget_internal, sticky=tk.W)
        combo_method_uniformity = ttk.Combobox(method_frame, textvariable=self.settings_vars['uniformity_method_var'],
                                               values=list(self.settings_vars['uniformity_formulas'].keys()),
                                               state='readonly', width=5)
        combo_method_uniformity.grid(row=0, column=1, padx=entry_padx, pady=pady_widget_internal, sticky=tk.EW)
        combo_method_uniformity.bind("<<ComboboxSelected>>", self.callbacks['_on_settings_change'])
        self.label_formula_display = ttk.Label(method_frame, textvariable=self.settings_vars['selected_formula_text_var'],
                                               font=('TkDefaultFont', 8), foreground="grey", wraplength=220)
        self.label_formula_display.grid(row=1, column=0, columnspan=2, padx=label_padx, pady=(pady_widget_internal[0], 3), sticky=tk.W)
        stats_profile_frame = ttk.Frame(method_frame)
        stats_profile_frame.grid(row=2, column=0, columnspan=2, sticky=tk.EW, pady=(3,0))
        stats_profile_frame.columnconfigure((0,1,2), weight=1, uniform="statsgroup")
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

        self.label_num_circular_profiles = ttk.Label(self.profile_config_outer_frame, text="Кол-во профилей (круг, 1-6):")
        self.entry_num_circular_profiles = ttk.Entry(
            self.profile_config_outer_frame, textvariable=self.settings_vars['num_circular_profiles_var'],
            width=entry_width_num_profiles, validate='key', validatecommand=self.vcmd['num_circular_profiles']
        )
        self.entry_num_circular_profiles.bind("<KeyRelease>", self.callbacks['_on_settings_change_entry'])

        self.linear_profile_type_radios_frame = ttk.Frame(self.profile_config_outer_frame)
        self.linear_profile_type_radios = {}
        radio_options = [
            ("Горизонтальный (по X)", "horizontal"),
            ("Вертикальный (по Y)", "vertical"),
            ("Оба", "both")
        ]
        for i, (text, val) in enumerate(radio_options):
            rb = ttk.Radiobutton(self.linear_profile_type_radios_frame, text=text,
                                 variable=self.settings_vars['linear_profile_type_var'], value=val,
                                 command=self.callbacks['_on_settings_change']) 
            rb.pack(side=tk.LEFT, padx=2, anchor=tk.W)
            self.linear_profile_type_radios[val] = rb
            
        self.label_linear_x_offset = ttk.Label(self.profile_config_outer_frame, text="Y-координата X-профиля (мм):")
        self.entry_linear_x_offset = ttk.Entry(
            self.profile_config_outer_frame, textvariable=self.settings_vars['linear_x_profile_y_offset_var'],
            width=6, validate='key', validatecommand=self.vcmd['float_general']
        )
        self.label_linear_y_offset = ttk.Label(self.profile_config_outer_frame, text="X-координата Y-профиля (мм):")
        self.entry_linear_y_offset = ttk.Entry(
            self.profile_config_outer_frame, textvariable=self.settings_vars['linear_y_profile_x_offset_var'],
            width=6, validate='key', validatecommand=self.vcmd['float_general']
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
                                             values=smoothing_options, state='readonly')
        self.combo_smoothing.bind("<<ComboboxSelected>>", self.callbacks['_on_settings_change'])

        self.savgol_frame = ttk.Frame(self.profile_config_outer_frame)
        ttk.Label(self.savgol_frame, text="Окно (нечет):").grid(row=0, column=0, padx=label_padx, pady=pady_widget_internal, sticky=tk.W)
        self.entry_savgol_window = ttk.Entry(self.savgol_frame, textvariable=self.settings_vars['savgol_window_var'],
                                             validate='key', validatecommand=self.vcmd['digits_only']) 
        self.entry_savgol_window.grid(row=0, column=1, padx=entry_padx, pady=pady_widget_internal, sticky=tk.EW)
        ttk.Label(self.savgol_frame, text="Полином:").grid(row=0, column=2, padx=label_padx, pady=pady_widget_internal, sticky=tk.W)
        self.entry_savgol_polyorder = ttk.Entry(self.savgol_frame, textvariable=self.settings_vars['savgol_polyorder_var'],
                                                validate='key', validatecommand=self.vcmd['digits_only']) 
        self.entry_savgol_polyorder.grid(row=0, column=3, padx=entry_padx, pady=pady_widget_internal, sticky=tk.EW) 
        self.savgol_frame.columnconfigure(1, weight=1) 
        self.savgol_frame.columnconfigure(3, weight=1) 
        self.entry_savgol_window.bind("<KeyRelease>", self.callbacks['_on_settings_change_entry'])
        self.entry_savgol_polyorder.bind("<KeyRelease>", self.callbacks['_on_settings_change_entry'])

        self.label_polyfit_degree = ttk.Label(self.profile_config_outer_frame, text="Степень полин.:")
        self.entry_polyfit_degree = ttk.Entry(self.profile_config_outer_frame, 
                                              textvariable=self.settings_vars['polyfit_degree_var'],
                                              validate='key', 
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
        self.roi_frame.columnconfigure(1, weight=1) 

        self.chk_roi_enabled = ttk.Checkbutton(self.roi_frame, text="Учитывать зону при расчете U",
                                               variable=self.settings_vars['roi_enabled_var'],
                                               command=self.callbacks['_on_settings_change'])
        self.chk_roi_enabled.grid(row=0, column=0, columnspan=2, sticky=tk.W, padx=label_padx, pady=pady_widget_internal)
        
        self.chk_roi_show_on_map = ttk.Checkbutton(self.roi_frame, text="Отображать зону на карте",
                                                   variable=self.settings_vars['roi_show_on_map_var'],
                                                   command=self.callbacks['recalculate_callback']) 
        self.chk_roi_show_on_map.grid(row=1, column=0, columnspan=2, sticky=tk.W, padx=label_padx, pady=pady_widget_internal)

        # ИЗМЕНЕНО: ROI фреймы создаются, но НЕ размещаются здесь. Их размещением управляет update_roi_fields_visibility_and_state
        self.roi_circular_frame = ttk.Frame(self.roi_frame)
        self.roi_circular_frame.columnconfigure(1, weight=1) 
        ttk.Label(self.roi_circular_frame, text="Мин. диаметр Dmin (мм):").grid(row=0, column=0, padx=label_padx, pady=pady_widget_internal, sticky=tk.W)
        self.entry_roi_d_min = ttk.Entry(self.roi_circular_frame, textvariable=self.settings_vars['roi_d_min_var'], width=entry_width_roi_circular, validate='key', validatecommand=self.vcmd['digits_or_empty'])
        self.entry_roi_d_min.grid(row=0, column=1, padx=entry_padx, pady=pady_widget_internal, sticky=tk.EW)
        self.entry_roi_d_min.bind("<KeyRelease>", self.callbacks['_on_settings_change_entry'])
        ttk.Label(self.roi_circular_frame, text="Макс. диаметр Dmax (мм):").grid(row=1, column=0, padx=label_padx, pady=pady_widget_internal, sticky=tk.W)
        self.entry_roi_d_max = ttk.Entry(self.roi_circular_frame, textvariable=self.settings_vars['roi_d_max_var'], width=entry_width_roi_circular, validate='key', validatecommand=self.vcmd['digits_or_empty'])
        self.entry_roi_d_max.grid(row=1, column=1, padx=entry_padx, pady=pady_widget_internal, sticky=tk.EW)
        self.entry_roi_d_max.bind("<KeyRelease>", self.callbacks['_on_settings_change_entry'])

        self.roi_rectangular_frame = ttk.Frame(self.roi_frame)
        self.roi_rectangular_frame.columnconfigure(1, weight=1) 
        ttk.Label(self.roi_rectangular_frame, text="Ширина зоны X (мм):").grid(row=0, column=0, padx=label_padx, pady=pady_widget_internal, sticky=tk.W)
        self.entry_roi_width_x = ttk.Entry(self.roi_rectangular_frame, textvariable=self.settings_vars['roi_width_x_var'], width=entry_width_roi_rectangular, validate='key', validatecommand=self.vcmd['digits_or_empty'])
        self.entry_roi_width_x.grid(row=0, column=1, padx=entry_padx, pady=pady_widget_internal, sticky=tk.EW)
        self.entry_roi_width_x.bind("<KeyRelease>", self.callbacks['_on_settings_change_entry'])
        ttk.Label(self.roi_rectangular_frame, text="Высота зоны Y (мм):").grid(row=1, column=0, padx=label_padx, pady=pady_widget_internal, sticky=tk.W)
        self.entry_roi_height_y = ttk.Entry(self.roi_rectangular_frame, textvariable=self.settings_vars['roi_height_y_var'], width=entry_width_roi_rectangular, validate='key', validatecommand=self.vcmd['digits_or_empty'])
        self.entry_roi_height_y.grid(row=1, column=1, padx=entry_padx, pady=pady_widget_internal, sticky=tk.EW)
        self.entry_roi_height_y.bind("<KeyRelease>", self.callbacks['_on_settings_change_entry'])
        ttk.Label(self.roi_rectangular_frame, text="Смещение центра X₀ (мм):").grid(row=2, column=0, padx=label_padx, pady=pady_widget_internal, sticky=tk.W)
        self.entry_roi_offset_x = ttk.Entry(self.roi_rectangular_frame, textvariable=self.settings_vars['roi_offset_x_var'], width=entry_width_roi_rectangular, validate='key', validatecommand=self.vcmd['float_or_empty'])
        self.entry_roi_offset_x.grid(row=2, column=1, padx=entry_padx, pady=pady_widget_internal, sticky=tk.EW)
        self.entry_roi_offset_x.bind("<KeyRelease>", self.callbacks['_on_settings_change_entry'])
        ttk.Label(self.roi_rectangular_frame, text="Смещение центра Y₀ (мм):").grid(row=3, column=0, padx=label_padx, pady=pady_widget_internal, sticky=tk.W)
        self.entry_roi_offset_y = ttk.Entry(self.roi_rectangular_frame, textvariable=self.settings_vars['roi_offset_y_var'], width=entry_width_roi_rectangular, validate='key', validatecommand=self.vcmd['float_or_empty'])
        self.entry_roi_offset_y.grid(row=3, column=1, padx=entry_padx, pady=pady_widget_internal, sticky=tk.EW)
        self.entry_roi_offset_y.bind("<KeyRelease>", self.callbacks['_on_settings_change_entry'])

        # --- Секция: Настройки цветовой карты ---
        self.cmap_frame = ttk.LabelFrame(self, text="Настройки цветовой карты", padding=internal_frame_padding)
        self.cmap_frame.grid(row=current_main_row, column=0, sticky=tk.EW, pady=pady_label_frame); current_main_row += 1
        self.cmap_frame.columnconfigure(1, weight=1)

        self.chk_cmap_manual_range = ttk.Checkbutton(self.cmap_frame, text="Ручной диапазон карты",
                                                     variable=self.settings_vars['cmap_manual_range_var'],
                                                     command=self.callbacks['_on_settings_change'])
        self.chk_cmap_manual_range.grid(row=0, column=0, columnspan=2, sticky=tk.W, padx=label_padx, pady=pady_widget_internal)

        self.cmap_manual_fields_frame = ttk.Frame(self.cmap_frame) 
        # ИЗМЕНЕНО: cmap_manual_fields_frame НЕ размещается здесь, а в update_cmap_range_fields_state
        self.cmap_manual_fields_frame.columnconfigure(1, weight=1) 
        
        ttk.Label(self.cmap_manual_fields_frame, text="Мин. значение:").grid(row=0, column=0, padx=label_padx, pady=pady_widget_internal, sticky=tk.W)
        self.entry_cmap_min_val = ttk.Entry(self.cmap_manual_fields_frame, textvariable=self.settings_vars['cmap_min_val_var'], width=entry_width_cmap, validate='key', validatecommand=self.vcmd['float_or_empty'])
        self.entry_cmap_min_val.grid(row=0, column=1, padx=entry_padx, pady=pady_widget_internal, sticky=tk.EW)
        self.entry_cmap_min_val.bind("<KeyRelease>", self.callbacks['_on_settings_change_entry'])
        ttk.Label(self.cmap_manual_fields_frame, text="Макс. значение:").grid(row=1, column=0, padx=label_padx, pady=pady_widget_internal, sticky=tk.W)
        self.entry_cmap_max_val = ttk.Entry(self.cmap_manual_fields_frame, textvariable=self.settings_vars['cmap_max_val_var'], width=entry_width_cmap, validate='key', validatecommand=self.vcmd['float_or_empty'])
        self.entry_cmap_max_val.grid(row=1, column=1, padx=entry_padx, pady=pady_widget_internal, sticky=tk.EW)
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

    def update_roi_fields_visibility_and_state(self, target_type):
        current_target_for_roi = target_type if target_type is not None else self.current_target_type
        roi_is_enabled = self.settings_vars['roi_enabled_var'].get()
        
        chk_show_state = tk.NORMAL if roi_is_enabled else tk.DISABLED
        if hasattr(self, 'chk_roi_show_on_map'): 
            self.chk_roi_show_on_map.config(state=chk_show_state)
            if not roi_is_enabled:
                self.settings_vars['roi_show_on_map_var'].set(False)

        # Управление видимостью фреймов ROI и состоянием их полей
        show_circular = roi_is_enabled and (current_target_for_roi in [self.TARGET_DISK, self.TARGET_DOME, self.TARGET_PLANETARY])
        show_rectangular = roi_is_enabled and (current_target_for_roi == self.TARGET_LINEAR)

        if hasattr(self, 'roi_circular_frame'):
            if show_circular:
                self.roi_circular_frame.grid(in_=self.roi_frame, row=2, column=0, columnspan=2, sticky=tk.EW, padx=(20,0), pady=(2,0))
                for child in self.roi_circular_frame.winfo_children():
                    if isinstance(child, ttk.Entry): child.config(state=tk.NORMAL)
            else:
                self.roi_circular_frame.grid_remove() 
                for child in self.roi_circular_frame.winfo_children(): 
                    if isinstance(child, ttk.Entry): child.config(state=tk.DISABLED)

        if hasattr(self, 'roi_rectangular_frame'):
            if show_rectangular:
                self.roi_rectangular_frame.grid(in_=self.roi_frame, row=2, column=0, columnspan=2, sticky=tk.EW, padx=(20,0), pady=(2,0))
                for child in self.roi_rectangular_frame.winfo_children():
                    if isinstance(child, ttk.Entry): child.config(state=tk.NORMAL)
            else:
                self.roi_rectangular_frame.grid_remove() 
                for child in self.roi_rectangular_frame.winfo_children(): 
                    if isinstance(child, ttk.Entry): child.config(state=tk.DISABLED)


    def update_cmap_range_fields_state(self):
        is_manual = self.settings_vars['cmap_manual_range_var'].get()
        fields_state = tk.NORMAL if is_manual else tk.DISABLED
        
        if hasattr(self, 'cmap_manual_fields_frame'): # Проверяем существование фрейма
            if is_manual:
                # Размещаем фрейм, если он еще не размещен или был удален
                self.cmap_manual_fields_frame.grid(in_=self.cmap_frame, row=1, column=0, columnspan=2, sticky=tk.EW, padx=(20,0))
            else:
                self.cmap_manual_fields_frame.grid_remove() # Скрываем фрейм

            # Управляем состоянием полей внутри фрейма
            if hasattr(self, 'entry_cmap_min_val'): self.entry_cmap_min_val.config(state=fields_state)
            if hasattr(self, 'entry_cmap_max_val'): self.entry_cmap_max_val.config(state=fields_state)


    def update_linear_profile_fields_state(self):
        profile_type = self.settings_vars['linear_profile_type_var'].get()
        
        state_x_offset = tk.DISABLED
        state_y_offset = tk.DISABLED

        if profile_type == "horizontal":
            state_x_offset = tk.NORMAL
        elif profile_type == "vertical":
            state_y_offset = tk.NORMAL
        elif profile_type == "both":
            state_x_offset = tk.NORMAL
            state_y_offset = tk.NORMAL
        
        if hasattr(self, 'entry_linear_x_offset'): self.entry_linear_x_offset.config(state=state_x_offset)
        if hasattr(self, 'entry_linear_y_offset'): self.entry_linear_y_offset.config(state=state_y_offset)


    def update_profile_options_layout(self, target_type: str | None):
        self.current_target_type = target_type 

        pady_widget_internal = (2, 2) 
        pady_group_separator = (5, 3) 
        current_row = 0 
        label_padx = (0, 5) 
        entry_padx = (0, 5) 

        self.label_num_circular_profiles.grid_remove()
        self.entry_num_circular_profiles.grid_remove()
        
        if hasattr(self, 'linear_profile_type_radios_frame'): self.linear_profile_type_radios_frame.grid_remove()
        self.label_linear_x_offset.grid_remove()
        self.entry_linear_x_offset.grid_remove()
        self.label_linear_y_offset.grid_remove()
        self.entry_linear_y_offset.grid_remove()
        
        self.savgol_frame.grid_remove()
        self.label_polyfit_degree.grid_remove()
        self.entry_polyfit_degree.grid_remove()

        if target_type in [self.TARGET_DISK, self.TARGET_DOME, self.TARGET_PLANETARY]:
            self.label_num_circular_profiles.grid(in_=self.profile_config_outer_frame, row=current_row, column=0, padx=label_padx, pady=pady_widget_internal, sticky=tk.W)
            self.entry_num_circular_profiles.grid(in_=self.profile_config_outer_frame, row=current_row, column=1, padx=entry_padx, pady=pady_widget_internal, sticky=tk.EW) 
            current_row += 1
        elif target_type == self.TARGET_LINEAR:
            ttk.Label(self.profile_config_outer_frame, text="Тип профиля:").grid(in_=self.profile_config_outer_frame, row=current_row, column=0, padx=label_padx, pady=pady_widget_internal, sticky=tk.W)
            if hasattr(self, 'linear_profile_type_radios_frame'): self.linear_profile_type_radios_frame.grid(in_=self.profile_config_outer_frame, row=current_row, column=1, padx=entry_padx, pady=pady_widget_internal, sticky=tk.EW)
            current_row +=1
            
            self.label_linear_x_offset.grid(in_=self.profile_config_outer_frame, row=current_row, column=0, padx=label_padx, pady=pady_widget_internal, sticky=tk.W)
            self.entry_linear_x_offset.grid(in_=self.profile_config_outer_frame, row=current_row, column=1, padx=entry_padx, pady=pady_widget_internal, sticky=tk.EW) 
            current_row +=1 
            self.label_linear_y_offset.grid(in_=self.profile_config_outer_frame, row=current_row, column=0, padx=label_padx, pady=pady_widget_internal, sticky=tk.W)
            self.entry_linear_y_offset.grid(in_=self.profile_config_outer_frame, row=current_row, column=1, padx=entry_padx, pady=pady_widget_internal, sticky=tk.EW) 
            current_row +=1
        
        self.update_linear_profile_fields_state()

        self.smoothing_label.grid(in_=self.profile_config_outer_frame, row=current_row, column=0, padx=label_padx, pady=(pady_group_separator[0], pady_widget_internal[1]), sticky=tk.W)
        self.combo_smoothing.grid(in_=self.profile_config_outer_frame, row=current_row, column=1, padx=entry_padx, pady=(pady_group_separator[0], pady_widget_internal[1]), sticky=tk.EW)
        current_row += 1 
        
        smoothing_param_frame_gridded = False
        if '_update_smoothing_options_display_internal' in self.helpers:
            smoothing_param_frame_gridded = self.helpers['_update_smoothing_options_display_internal'](
                self.profile_config_outer_frame, 
                self.savgol_frame, 
                self.label_polyfit_degree, 
                self.entry_polyfit_degree, 
                self.settings_vars['smoothing_method_var'].get(),
                current_row 
            )
        else:
            print("ПРЕДУПРЕЖДЕНИЕ (tab_main_panel.py): Метод '_update_smoothing_options_display_internal' не найден в self.helpers.")

        if smoothing_param_frame_gridded:
            current_row += 1 
        
        self.chk_raw_profile.grid(in_=self.profile_config_outer_frame, row=current_row, column=0, columnspan=2, pady=(pady_group_separator[0], pady_widget_internal[1]), sticky=tk.W)
        current_row += 1
        self.chk_percent.grid(in_=self.profile_config_outer_frame, row=current_row, column=0, columnspan=2, pady=pady_widget_internal, sticky=tk.W)
        current_row += 1
        self.chk_logscale.grid(in_=self.profile_config_outer_frame, row=current_row, column=0, columnspan=2, pady=pady_widget_internal, sticky=tk.W)

        self.update_roi_fields_visibility_and_state(target_type)
