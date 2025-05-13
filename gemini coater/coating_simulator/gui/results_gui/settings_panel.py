# coding: utf-8
# Файл: coating_simulator/gui/results_gui/settings_panel.py
"""
Содержит класс SettingsPanel для панели настроек в ResultsWindow.
Улучшена компоновка и адаптивность элементов, исправлено размещение
динамически отображаемых параметров сглаживания.
"""
import tkinter as tk
from tkinter import ttk, messagebox
import traceback
import math

try:
    from ... import config
except ImportError:
    class ConfigMock: #type: ignore
        TARGET_DISK = "диск"
        TARGET_DOME = "купол"
        TARGET_PLANETARY = "планетарный"
        TARGET_LINEAR = "линейное перемещение"
        VIS_DEFAULT_PERCENT = True
        VIS_DEFAULT_LOGSCALE = False
    config = ConfigMock()
    print("ПРЕДУПРЕЖДЕНИЕ (settings_panel.py): Используется ConfigMock, т.к. не удалось импортировать config.")


class SettingsPanel(ttk.Frame):
    """
    Левая панель настроек для окна результатов.
    """
    def __init__(self, master, recalculate_callback, export_excel_callback,
                 calculate_mask_callback, load_profiles_callback, reconstruct_map_callback,
                 *args, **kwargs):
        super().__init__(master, padding=(5, 5, 5, 5), *args, **kwargs)
        self.columnconfigure(0, weight=1)

        self.recalculate_callback = recalculate_callback
        self.export_excel_callback = export_excel_callback
        self.calculate_mask_callback = calculate_mask_callback
        self.load_profiles_callback = load_profiles_callback
        self.reconstruct_map_callback = reconstruct_map_callback

        self._initialization_complete = False

        self.uniformity_formulas = {
            'U1': "U₁ = (Max-Min)/(Max+Min)", 'U2': "U₂ = (Max-Min)/Mean",
            'U3': "U₃ = StdDev/Mean (σ/t̄)", 'U4': "U₄ = Min/Max"
        }
        self.uniformity_method_var = tk.StringVar(value='U3')
        self.selected_formula_text_var = tk.StringVar(value=self.uniformity_formulas['U3'])
        self.profile_t_max_var = tk.StringVar(value="Max: -")
        self.profile_t_min_var = tk.StringVar(value="Min: -")
        self.profile_t_mean_var = tk.StringVar(value="Mean: -")

        self.smoothing_method_var = tk.StringVar(value='Savitzky-Golay')
        self.savgol_window_var = tk.StringVar(value='11')
        self.savgol_polyorder_var = tk.StringVar(value='3')
        self.polyfit_degree_var = tk.StringVar(value='5')

        self.display_percent_var = tk.BooleanVar(value=config.VIS_DEFAULT_PERCENT)
        self.use_logscale_var = tk.BooleanVar(value=config.VIS_DEFAULT_LOGSCALE)
        self.show_raw_profile_var = tk.BooleanVar(value=False)

        self.num_circular_profiles_var = tk.StringVar(value="1")
        self.linear_x_profile_y_offset_var = tk.StringVar(value="0.0")
        self.linear_y_profile_x_offset_var = tk.StringVar(value="0.0")

        self.auto_uniformity_mask_height_var = tk.StringVar(value='50.0')
        self.auto_uniformity_mode_var = tk.StringVar(value='Маска')

        self.loaded_files_text_var = tk.StringVar(value="Файлы не загружены")
        self.reconstruction_method_var = tk.StringVar(value="Линейная интерполяция")

        self.vcmd_int = (self.register(self._validate_positive_int), '%P')
        self.vcmd_odd_int = (self.register(self._validate_odd_positive_int), '%P')
        self.vcmd_float_positive_or_zero = (self.register(self._validate_positive_float_or_zero), '%P')
        self.vcmd_float_general = (self.register(self._validate_float_general), '%P')
        self.vcmd_num_circular_profiles = (self.register(self._validate_num_circular_profiles), '%P', '%d', '%s', '%S')

        self._create_widgets()
        self._initial_ui_update()

    def mark_initialization_complete(self):
        self._initialization_complete = True

    def _validate_positive_int(self, P_value):
        if P_value == "": return True
        try: return int(P_value) > 0
        except ValueError: return False

    def _validate_odd_positive_int(self, P_value):
        if P_value == "": return True
        try: val = int(P_value); return val > 0 and val % 2 != 0 and val >=3
        except ValueError: return False

    def _validate_positive_float_or_zero(self, P_value):
        if P_value == "": return True;
        if P_value == ".": return True;
        if P_value == "-": return True 
        if P_value.count('.') > 1: return False
        try: return float(P_value) >= 0.0
        except ValueError: return False
        
    def _validate_float_general(self, P_value):
        if P_value == "": return True;
        if P_value == "-": return True;
        if P_value == ".": return True
        if P_value.startswith("-.") and P_value.count('.') > 1 : return False
        if P_value.startswith("-") and P_value.count('.') > 1 : return False
        if not P_value.startswith("-") and P_value.count('.') > 1: return False
        try: float(P_value); return True
        except ValueError: return False

    def _validate_num_circular_profiles(self, P_value, action_code, current_text, inserted_text):
        if action_code == '1': 
            if not inserted_text.isdigit(): return False
            try:
                val = int(P_value)
                if len(P_value) == 1: return 1 <= val <= 6
                elif len(P_value) > 1: return False 
                return True 
            except ValueError: return False
        return True

    def _is_scipy_available(self):
        try: from scipy.signal import savgol_filter; return True
        except ImportError: return False

    def _create_widgets(self):
        pady_label_frame = (5, 5)
        pady_widget_group_sep = (5, 2)
        pady_widget_internal = (1, 1)
        internal_frame_padding = (5, 5, 5, 5)
        current_main_row = 0

        method_frame = ttk.LabelFrame(self, text="Расчет равномерности U (по профилю)", padding=internal_frame_padding)
        method_frame.grid(row=current_main_row, column=0, sticky=tk.EW, pady=pady_label_frame); current_main_row += 1
        method_frame.columnconfigure(1, weight=1)
        ttk.Label(method_frame, text="Метод:").grid(row=0, column=0, padx=(0,5), pady=pady_widget_internal, sticky=tk.W)
        combo_method_uniformity = ttk.Combobox(method_frame, textvariable=self.uniformity_method_var, values=list(self.uniformity_formulas.keys()), state='readonly', width=5)
        combo_method_uniformity.grid(row=0, column=1, padx=2, pady=pady_widget_internal, sticky=tk.EW)
        combo_method_uniformity.bind("<<ComboboxSelected>>", self._on_settings_change)
        self.label_formula_display = ttk.Label(method_frame, textvariable=self.selected_formula_text_var, font=('TkDefaultFont', 8), foreground="grey", wraplength=220)
        self.label_formula_display.grid(row=1, column=0, columnspan=2, padx=2, pady=(pady_widget_internal[0], 3), sticky=tk.W)
        stats_profile_frame = ttk.Frame(method_frame)
        stats_profile_frame.grid(row=2, column=0, columnspan=2, sticky=tk.EW, pady=(3,0))
        stats_profile_frame.columnconfigure((0,1,2), weight=1, uniform="statsgroup")
        self.label_t_max = ttk.Label(stats_profile_frame, textvariable=self.profile_t_max_var, font=('TkDefaultFont', 8))
        self.label_t_max.grid(row=0, column=0, sticky=tk.W, padx=(0,2))
        self.label_t_min = ttk.Label(stats_profile_frame, textvariable=self.profile_t_min_var, font=('TkDefaultFont', 8))
        self.label_t_min.grid(row=0, column=1, sticky=tk.W, padx=2)
        self.label_t_mean = ttk.Label(stats_profile_frame, textvariable=self.profile_t_mean_var, font=('TkDefaultFont', 8))
        self.label_t_mean.grid(row=0, column=2, sticky=tk.W, padx=(2,0))

        self.profile_config_outer_frame = ttk.LabelFrame(self, text="Настройки профиля для U и вида", padding=internal_frame_padding)
        self.profile_config_outer_frame.grid(row=current_main_row, column=0, sticky=tk.EW, pady=pady_label_frame); current_main_row += 1
        self.profile_config_outer_frame.columnconfigure(1, weight=1)

        self.label_num_circular_profiles = ttk.Label(self.profile_config_outer_frame, text="Кол-во профилей (круг, 1-6):")
        self.entry_num_circular_profiles = ttk.Entry(
            self.profile_config_outer_frame, textvariable=self.num_circular_profiles_var,
            width=5, validate='key', validatecommand=self.vcmd_num_circular_profiles
        )
        self.entry_num_circular_profiles.bind("<KeyRelease>", self._on_settings_change_entry)

        self.label_linear_x_offset = ttk.Label(self.profile_config_outer_frame, text="Y-смещ. для X-профиля (мм):")
        self.entry_linear_x_offset = ttk.Entry(
            self.profile_config_outer_frame, textvariable=self.linear_x_profile_y_offset_var,
            width=6, validate='key', validatecommand=self.vcmd_float_general
        )
        self.label_linear_y_offset = ttk.Label(self.profile_config_outer_frame, text="X-смещ. для Y-профиля (мм):")
        self.entry_linear_y_offset = ttk.Entry(
            self.profile_config_outer_frame, textvariable=self.linear_y_profile_x_offset_var,
            width=6, validate='key', validatecommand=self.vcmd_float_general
        )
        self.entry_linear_x_offset.bind("<KeyRelease>", self._on_settings_change_entry)
        self.entry_linear_y_offset.bind("<KeyRelease>", self._on_settings_change_entry)
        
        self.smoothing_label = ttk.Label(self.profile_config_outer_frame, text="Сглаживание:")
        smoothing_options = ["Без сглаживания", "Полином. аппрокс."]
        if self._is_scipy_available():
            smoothing_options.insert(0, "Savitzky-Golay"); self.smoothing_method_var.set("Savitzky-Golay")
        else: self.smoothing_method_var.set("Полином. аппрокс.")
        
        combo_smoothing = ttk.Combobox(self.profile_config_outer_frame, textvariable=self.smoothing_method_var, values=smoothing_options, state='readonly', width=16)
        combo_smoothing.bind("<<ComboboxSelected>>", self._on_settings_change)
        self.combo_smoothing_name = str(combo_smoothing)

        self.savgol_frame = ttk.Frame(self.profile_config_outer_frame)
        ttk.Label(self.savgol_frame, text="Окно (нечет):").grid(row=0, column=0, padx=(0,2), pady=pady_widget_internal, sticky=tk.W)
        self.entry_savgol_window = ttk.Entry(self.savgol_frame, textvariable=self.savgol_window_var, width=4, validate='key', validatecommand=self.vcmd_odd_int)
        self.entry_savgol_window.grid(row=0, column=1, padx=2, pady=pady_widget_internal, sticky=tk.EW)
        ttk.Label(self.savgol_frame, text="Полином:").grid(row=1, column=0, padx=(0,2), pady=pady_widget_internal, sticky=tk.W)
        self.entry_savgol_polyorder = ttk.Entry(self.savgol_frame, textvariable=self.savgol_polyorder_var, width=4, validate='key', validatecommand=self.vcmd_int)
        self.entry_savgol_polyorder.grid(row=1, column=1, padx=2, pady=pady_widget_internal, sticky=tk.EW)
        self.entry_savgol_window.bind("<KeyRelease>", self._on_settings_change_entry)
        self.entry_savgol_polyorder.bind("<KeyRelease>", self._on_settings_change_entry)

        self.polyfit_frame = ttk.Frame(self.profile_config_outer_frame)
        ttk.Label(self.polyfit_frame, text="Степень полин.:").grid(row=0, column=0, padx=(0,2), pady=pady_widget_internal, sticky=tk.W)
        self.entry_polyfit_degree = ttk.Entry(self.polyfit_frame, textvariable=self.polyfit_degree_var, width=4, validate='key', validatecommand=self.vcmd_int)
        self.entry_polyfit_degree.grid(row=0, column=1, padx=2, pady=pady_widget_internal, sticky=tk.EW)
        self.entry_polyfit_degree.bind("<KeyRelease>", self._on_settings_change_entry)
        
        chk_raw_profile = ttk.Checkbutton(self.profile_config_outer_frame, text="Показать сырой профиль", variable=self.show_raw_profile_var, command=self.recalculate_callback)
        self.chk_raw_profile_name = str(chk_raw_profile)
        chk_percent = ttk.Checkbutton(self.profile_config_outer_frame, text="Покрытие в %", variable=self.display_percent_var, command=self.recalculate_callback)
        self.chk_percent_name = str(chk_percent)
        chk_logscale = ttk.Checkbutton(self.profile_config_outer_frame, text="Лог. шкала (карта)", variable=self.use_logscale_var, command=self.recalculate_callback)
        self.chk_logscale_name = str(chk_logscale)

        auto_frame = ttk.LabelFrame(self, text="Авторавномерность", padding=internal_frame_padding)
        auto_frame.grid(row=current_main_row, column=0, sticky=tk.EW, pady=pady_label_frame); current_main_row += 1
        auto_frame.columnconfigure(1, weight=1)
        ttk.Label(auto_frame, text="Метод:").grid(row=0, column=0, padx=(0,5), pady=pady_widget_internal, sticky=tk.W)
        self.combo_auto_mode = ttk.Combobox(auto_frame, textvariable=self.auto_uniformity_mode_var, values=['Маска', 'Источник'], state='readonly', width=10)
        self.combo_auto_mode.grid(row=0, column=1, padx=2, pady=pady_widget_internal, sticky=tk.EW)
        ttk.Label(auto_frame, text="Высота маски (мм):").grid(row=1, column=0, padx=(0,5), pady=pady_widget_internal, sticky=tk.W)
        self.entry_mask_height = ttk.Entry(auto_frame, textvariable=self.auto_uniformity_mask_height_var, width=6, validate='key', validatecommand=self.vcmd_float_positive_or_zero)
        self.entry_mask_height.grid(row=1, column=1, padx=2, pady=pady_widget_internal, sticky=tk.EW)
        self.btn_calc_mask = ttk.Button(auto_frame, text="Рассчитать", command=self.calculate_mask_callback)
        self.btn_calc_mask.grid(row=2, column=0, columnspan=2, pady=(pady_widget_group_sep[0],0), sticky=tk.EW)

        inverse_problem_frame = ttk.LabelFrame(self, text="Обратная задача (по профилям)", padding=internal_frame_padding)
        inverse_problem_frame.grid(row=current_main_row, column=0, sticky=tk.EW, pady=pady_label_frame); current_main_row += 1
        inverse_problem_frame.columnconfigure(1, weight=1)
        btn_load_profiles = ttk.Button(inverse_problem_frame, text="Загрузить профиль(и)", command=self.load_profiles_callback)
        btn_load_profiles.grid(row=0, column=0, columnspan=2, pady=pady_widget_internal, sticky=tk.EW)
        self.label_loaded_files = ttk.Label(inverse_problem_frame, textvariable=self.loaded_files_text_var, wraplength=220, justify=tk.LEFT, font=('TkDefaultFont', 8))
        self.label_loaded_files.grid(row=1, column=0, columnspan=2, pady=pady_widget_internal, sticky=tk.W)
        ttk.Label(inverse_problem_frame, text="Метод реконстр.:").grid(row=2, column=0, padx=(0,5), pady=pady_widget_internal, sticky=tk.W)
        self.combo_reconstruction = ttk.Combobox(inverse_problem_frame, textvariable=self.reconstruction_method_var, values=["Линейная", "Кубический сплайн"], state='readonly', width=14)
        self.combo_reconstruction.grid(row=2, column=1, padx=2, pady=pady_widget_internal, sticky=tk.EW)
        self.btn_reconstruct_map = ttk.Button(inverse_problem_frame, text="Построить карту по профилям", command=self._internal_reconstruct_map_callback)
        self.btn_reconstruct_map.grid(row=3, column=0, columnspan=2, pady=(pady_widget_group_sep[0],0), sticky=tk.EW)

        control_panel = ttk.Frame(self, padding=(0, 5, 0, 0))
        control_panel.grid(row=current_main_row, column=0, sticky=tk.EW, pady=pady_label_frame); current_main_row += 1
        control_panel.columnconfigure((0, 1), weight=1, uniform="controlgroup")
        btn_update = ttk.Button(control_panel, text="Обновить графики", command=self.recalculate_callback)
        btn_update.grid(row=0, column=0, padx=(0,2), pady=(1,0), sticky=tk.EW)
        self.btn_export_excel = ttk.Button(control_panel, text="Экспорт в Excel", command=self.export_excel_callback, state=tk.DISABLED)
        self.btn_export_excel.grid(row=0, column=1, padx=(2,0), pady=(1,0), sticky=tk.EW)

    def _internal_reconstruct_map_callback(self):
        if self.reconstruct_map_callback:
            self.reconstruct_map_callback(self.reconstruction_method_var.get())

    def _initial_ui_update(self):
        self._update_formula_display_text_only_internal()
        self.update_profile_options(config.TARGET_DISK) 
        
    def _on_settings_change(self, event=None):
        self._update_formula_display_text_only_internal()
        # _update_smoothing_options_display_internal вызывается из update_profile_options
        # или при смене метода сглаживания
        if event and event.widget == self.profile_config_outer_frame.nametowidget(self.combo_smoothing_name):
            self._update_smoothing_options_display_internal() # Обновить, если изменился метод сглаживания

        if self.recalculate_callback and self._initialization_complete:
            self.recalculate_callback()

    def _on_settings_change_entry(self, event=None):
        if self.recalculate_callback and self._initialization_complete:
            self.after_idle(self.recalculate_callback)

    def _update_formula_display_text_only_internal(self):
        selected_key = self.uniformity_method_var.get()
        formula_text = self.uniformity_formulas.get(selected_key, "Неизвестная формула")
        self.selected_formula_text_var.set(formula_text)

    def _update_smoothing_options_display_internal(self, base_row_for_params: int) -> bool:
        """
        Обновляет видимость и размещение фреймов с параметрами сглаживания.
        Args:
            base_row_for_params: Строка в self.profile_config_outer_frame,
                                 на которой нужно разместить фрейм параметров сглаживания.
        Returns:
            bool: True, если фрейм параметров сглаживания был отображен, иначе False.
        """
        method = self.smoothing_method_var.get()
        gridded_a_frame = False

        self.savgol_frame.grid_remove()
        self.polyfit_frame.grid_remove()

        if method == "Savitzky-Golay":
            if self._is_scipy_available():
                self.savgol_frame.grid(in_=self.profile_config_outer_frame, row=base_row_for_params, column=0, columnspan=2, sticky=tk.EW, padx=(5,0), pady=(2,2)) # Добавлен pady
                gridded_a_frame = True
        elif method == "Полином. аппрокс.":
            self.polyfit_frame.grid(in_=self.profile_config_outer_frame, row=base_row_for_params, column=0, columnspan=2, sticky=tk.EW, padx=(5,0), pady=(2,2)) # Добавлен pady
            gridded_a_frame = True
        return gridded_a_frame


    def update_profile_options(self, target_type: str | None):
        pady_widget_internal = (2, 2) 
        pady_group_separator = (5, 3) 
        current_row = 0 # Локальный счетчик строк для self.profile_config_outer_frame

        # Сначала скрываем все элементы, которые могут меняться
        self.label_num_circular_profiles.grid_remove()
        self.entry_num_circular_profiles.grid_remove()
        self.label_linear_x_offset.grid_remove()
        self.entry_linear_x_offset.grid_remove()
        self.label_linear_y_offset.grid_remove()
        self.entry_linear_y_offset.grid_remove()
        self.savgol_frame.grid_remove()
        self.polyfit_frame.grid_remove()

        # Размещаем элементы выбора профиля
        if target_type in [config.TARGET_DISK, config.TARGET_DOME, config.TARGET_PLANETARY]:
            self.label_num_circular_profiles.grid(in_=self.profile_config_outer_frame, row=current_row, column=0, padx=(0,5), pady=pady_widget_internal, sticky=tk.W)
            self.entry_num_circular_profiles.grid(in_=self.profile_config_outer_frame, row=current_row, column=1, padx=2, pady=pady_widget_internal, sticky=tk.EW)
            current_row += 1
        elif target_type == config.TARGET_LINEAR:
            self.label_linear_x_offset.grid(in_=self.profile_config_outer_frame, row=current_row, column=0, padx=(0,5), pady=pady_widget_internal, sticky=tk.W)
            self.entry_linear_x_offset.grid(in_=self.profile_config_outer_frame, row=current_row, column=1, padx=2, pady=pady_widget_internal, sticky=tk.EW)
            current_row +=1 
            self.label_linear_y_offset.grid(in_=self.profile_config_outer_frame, row=current_row, column=0, padx=(0,5), pady=pady_widget_internal, sticky=tk.W)
            self.entry_linear_y_offset.grid(in_=self.profile_config_outer_frame, row=current_row, column=1, padx=2, pady=pady_widget_internal, sticky=tk.EW)
            current_row +=1
        
        # Размещаем группу сглаживания
        self.smoothing_label.grid(in_=self.profile_config_outer_frame, row=current_row, column=0, padx=(0,5), pady=(pady_group_separator[0], pady_widget_internal[1]), sticky=tk.W)
        combo_smoothing_widget = self.profile_config_outer_frame.nametowidget(self.combo_smoothing_name)
        combo_smoothing_widget.grid(in_=self.profile_config_outer_frame, row=current_row, column=1, padx=2, pady=(pady_group_separator[0], pady_widget_internal[1]), sticky=tk.EW)
        current_row += 1
        
        # Размещаем параметры сглаживания
        smoothing_param_frame_gridded = self._update_smoothing_options_display_internal(current_row) # Передаем текущую строку
        if smoothing_param_frame_gridded:
            current_row += 1 

        # Размещаем чекбоксы
        self.profile_config_outer_frame.nametowidget(self.chk_raw_profile_name).grid(in_=self.profile_config_outer_frame, row=current_row, column=0, columnspan=2, pady=(pady_group_separator[0], pady_widget_internal[1]), sticky=tk.W)
        current_row += 1
        self.profile_config_outer_frame.nametowidget(self.chk_percent_name).grid(in_=self.profile_config_outer_frame, row=current_row, column=0, columnspan=2, pady=pady_widget_internal, sticky=tk.W)
        current_row += 1
        self.profile_config_outer_frame.nametowidget(self.chk_logscale_name).grid(in_=self.profile_config_outer_frame, row=current_row, column=0, columnspan=2, pady=pady_widget_internal, sticky=tk.W)

        if self.recalculate_callback and self._initialization_complete:
             self.after_idle(self.recalculate_callback)

    def get_uniformity_method(self) -> str: return self.uniformity_method_var.get()
    def update_profile_stats(self, t_max: str, t_min: str, t_mean: str):
        self.profile_t_max_var.set(f"Max: {t_max}")
        self.profile_t_min_var.set(f"Min: {t_min}")
        self.profile_t_mean_var.set(f"Mean: {t_mean}")

    def get_view_settings(self) -> dict | None:
        smoothing_params_data = {'method': self.smoothing_method_var.get()}
        specific_params = {}
        try:
            if smoothing_params_data['method'] == "Savitzky-Golay":
                if not self._is_scipy_available(): smoothing_params_data['method'] = "Без сглаживания"
                else:
                    win_str = self.savgol_window_var.get(); poly_str = self.savgol_polyorder_var.get()
                    if not win_str or not poly_str: raise ValueError("Параметры SavGol не м.б. пустыми")
                    specific_params['window_length'] = int(win_str); specific_params['polyorder'] = int(poly_str)
                    if specific_params['window_length'] < 3 or specific_params['window_length'] % 2 == 0: raise ValueError("Окно SavGol >=3, нечет.")
                    if specific_params['polyorder'] >= specific_params['window_length']: raise ValueError("Порядок SavGol < окна")
            elif smoothing_params_data['method'] == "Полином. аппрокс.":
                deg_str = self.polyfit_degree_var.get()
                if not deg_str: raise ValueError("Степень полинома не м.б. пустой")
                specific_params['degree'] = int(deg_str)
                if specific_params['degree'] < 1: raise ValueError("Степень полинома >= 1")
            smoothing_params_data['params'] = specific_params
        except ValueError as e: messagebox.showerror("Ошибка ввода", f"Настройки сглаживания: {e}", parent=self); return None

        profile_params = {}
        if self.label_num_circular_profiles.winfo_ismapped():
            num_str = self.num_circular_profiles_var.get()
            try:
                if not num_str: num_val = 1
                else: num_val = int(num_str)
                if not (1 <= num_val <= 6):
                    corrected_val = min(max(1, num_val if num_str else 1), 6)
                    if num_val != corrected_val or not num_str:
                        self.num_circular_profiles_var.set(str(corrected_val))
                    num_val = corrected_val
                profile_params['num_circular_profiles'] = num_val
            except ValueError:
                self.num_circular_profiles_var.set("1")
                profile_params['num_circular_profiles'] = 1
        elif self.label_linear_x_offset.winfo_ismapped():
            try:
                x_offset_str = self.linear_x_profile_y_offset_var.get()
                y_offset_str = self.linear_y_profile_x_offset_var.get()
                profile_params['linear_x_profile_y_offset'] = float(x_offset_str if x_offset_str and x_offset_str != "-" else "0.0")
                profile_params['linear_y_profile_x_offset'] = float(y_offset_str if y_offset_str and y_offset_str != "-" else "0.0")
            except ValueError as e:
                profile_params['linear_x_profile_y_offset'] = 0.0
                profile_params['linear_y_profile_x_offset'] = 0.0
            
        return {
            'smoothing': smoothing_params_data,
            'display_percent': self.display_percent_var.get(),
            'use_logscale': self.use_logscale_var.get(),
            'show_raw_profile': self.show_raw_profile_var.get(),
            'profile_config_params': profile_params,
            'uniformity_method_key': self.uniformity_method_var.get()
        }

    def get_auto_uniformity_params(self) -> dict | None:
        try:
            height_str = self.auto_uniformity_mask_height_var.get()
            if not height_str: raise ValueError("Высота маски не м.б. пустой.")
            return {'mode': self.auto_uniformity_mode_var.get(), 'mask_height': float(height_str)}
        except ValueError: messagebox.showerror("Ошибка ввода", "Некорректная высота маски.", parent=self); return None
    def get_reconstruction_method(self) -> str: return self.reconstruction_method_var.get()
    def update_loaded_files_text(self, text: str): self.loaded_files_text_var.set(text)
    def enable_export_button(self): self.btn_export_excel.config(state=tk.NORMAL)
    def disable_export_button(self): self.btn_export_excel.config(state=tk.DISABLED)

if __name__ == '__main__':
    root = tk.Tk()
    root.title("Тест SettingsPanel (Layout Final Fix)")
    def mock_recalculate(): print("Вызван mock_recalculate")
    panel = SettingsPanel(root, mock_recalculate, lambda:None, lambda:None, lambda:None, lambda m:None)
    panel.mark_initialization_complete()
    panel.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    def check_view_settings():
        settings = panel.get_view_settings()
        print("View Settings:", settings)
    ttk.Button(root, text="Get View Settings", command=check_view_settings).pack(pady=5)
    current_mock_target_type = tk.StringVar(value=config.TARGET_DISK)
    def switch_target_type(event=None):
        target_type = current_mock_target_type.get()
        print(f"\nПереключение на '{target_type}' тип мишени (имитация)")
        panel.update_profile_options(target_type)
    mock_target_type_combo = ttk.Combobox(root, textvariable=current_mock_target_type,
                                          values=[config.TARGET_DISK, config.TARGET_LINEAR, config.TARGET_DOME, config.TARGET_PLANETARY],
                                          state='readonly')
    mock_target_type_combo.pack(pady=5)
    mock_target_type_combo.bind("<<ComboboxSelected>>", switch_target_type)
    panel.update_profile_options(current_mock_target_type.get())
    root.mainloop()
