# coding: utf-8
# Файл: coating_simulator/gui/results_gui/settings_panel.py
"""
Содержит класс SettingsPanel для панели настроек в ResultsWindow.
Элементы управления реорганизованы с использованием вкладок, каждая вкладка в своем файле,
расположенных в подпапке tab_panels.
Конфигурационные параметры импортируются напрямую из модуля config.
"""
import tkinter as tk
from tkinter import ttk, messagebox
import math

try:
    from ...config import (
        TARGET_DISK, TARGET_DOME, TARGET_PLANETARY, TARGET_LINEAR,
        VIS_DEFAULT_PERCENT, VIS_DEFAULT_LOGSCALE
    )
except ImportError as e:
    print(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось импортировать конфигурацию из ...config: {e}")
    TARGET_DISK = "диск (MOCK)" 
    TARGET_DOME = "купол (MOCK)"
    TARGET_PLANETARY = "планетарный (MOCK)"
    TARGET_LINEAR = "линейное перемещение (MOCK)"
    VIS_DEFAULT_PERCENT = True
    VIS_DEFAULT_LOGSCALE = False
    messagebox.showerror("Ошибка конфигурации", 
                         "Не удалось загрузить основные параметры конфигурации. Используются значения по умолчанию (MOCK).")

from .tab_panels.tab_main_panel import MainPanelTab
from .tab_panels.tab_auto_uniformity_panel import AutoUniformityPanelTab
from .tab_panels.tab_inverse_task_panel import InverseTaskPanelTab


class SettingsPanel(ttk.Frame):
    def __init__(self, master, recalculate_callback, export_excel_callback,
                 calculate_mask_callback, load_profiles_callback, reconstruct_map_callback,
                 *args, **kwargs):
        super().__init__(master, padding=(0, 0, 0, 0), *args, **kwargs)
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
        self._default_savgol_window = '11'
        self._default_savgol_polyorder = '3'
        self._default_polyfit_degree = '5'
        self.savgol_window_var = tk.StringVar(value=self._default_savgol_window) 
        self.savgol_polyorder_var = tk.StringVar(value=self._default_savgol_polyorder) 
        self.polyfit_degree_var = tk.StringVar(value=self._default_polyfit_degree) 

        self.display_percent_var = tk.BooleanVar(value=VIS_DEFAULT_PERCENT)
        self.use_logscale_var = tk.BooleanVar(value=VIS_DEFAULT_LOGSCALE)
        self.show_raw_profile_var = tk.BooleanVar(value=False)

        self._default_num_circular_profiles = "1"
        self.num_circular_profiles_var = tk.StringVar(value=self._default_num_circular_profiles)
        
        self.linear_x_profile_y_offset_var = tk.StringVar(value="0.0")
        self.linear_y_profile_x_offset_var = tk.StringVar(value="0.0")

        self.roi_enabled_var = tk.BooleanVar(value=False)
        self.roi_show_on_map_var = tk.BooleanVar(value=False)
        self._default_roi_d_min = "0"
        self._default_roi_d_max = "100"
        self._default_roi_width_x = "100"
        self._default_roi_height_y = "100"
        self._default_roi_offset_x = "0.0"
        self._default_roi_offset_y = "0.0"

        self.roi_d_min_var = tk.StringVar(value=self._default_roi_d_min)
        self.roi_d_max_var = tk.StringVar(value=self._default_roi_d_max)
        self.roi_width_x_var = tk.StringVar(value=self._default_roi_width_x)
        self.roi_height_y_var = tk.StringVar(value=self._default_roi_height_y)
        self.roi_offset_x_var = tk.StringVar(value=self._default_roi_offset_x)
        self.roi_offset_y_var = tk.StringVar(value=self._default_roi_offset_y)
        
        self.cmap_manual_range_var = tk.BooleanVar(value=False)
        self._default_cmap_min = "0.0"
        self._default_cmap_max = "1.0" 
        self.cmap_min_val_var = tk.StringVar(value=self._default_cmap_min)
        self.cmap_max_val_var = tk.StringVar(value=self._default_cmap_max)

        self.linear_profile_type_var = tk.StringVar(value="both") 

        self.auto_uniformity_mask_height_var = tk.StringVar(value='50.0')
        self.auto_uniformity_mode_var = tk.StringVar(value='Маска')
        self.loaded_files_text_var = tk.StringVar(value="Файлы не загружены")
        self.reconstruction_method_var = tk.StringVar(value="Линейная")

        self.vcmd_int = (self.register(self._validate_positive_int), '%P') 
        self.vcmd_odd_int = (self.register(self._validate_odd_positive_int), '%P')
        self.vcmd_float_positive_or_zero = (self.register(self._validate_positive_float_or_zero), '%P') 
        self.vcmd_float_general = (self.register(self._validate_float_general), '%P') 
        self.vcmd_num_circular_profiles = (self.register(self._validate_num_circular_profiles), '%P', '%d', '%s', '%S')
        self.vcmd_digits_only = (self.register(self._validate_digits_only), '%S', '%d') 
        self.vcmd_digits_or_empty = (self.register(self._validate_digits_or_empty), '%P') 
        self.vcmd_float_or_empty = (self.register(self._validate_float_or_empty), '%P') 

        self._create_notebook_and_instantiate_tabs()
        self._initial_ui_update() # Вызываем после создания вкладок

    def _create_notebook_and_instantiate_tabs(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(expand=True, fill='both', padx=5, pady=5)

        tab_main_frame = ttk.Frame(self.notebook, padding=(5,5))
        tab_auto_uniformity_frame = ttk.Frame(self.notebook, padding=(5,5))
        tab_inverse_task_frame = ttk.Frame(self.notebook, padding=(5,5))
        
        tab_main_frame.columnconfigure(0, weight=1)
        tab_auto_uniformity_frame.columnconfigure(0, weight=1)
        tab_inverse_task_frame.columnconfigure(0, weight=1)

        self.notebook.add(tab_main_frame, text='Основные')
        self.notebook.add(tab_auto_uniformity_frame, text='Авторавномерность')
        self.notebook.add(tab_inverse_task_frame, text='Обратная задача')

        settings_vars_for_tabs = {
            'uniformity_formulas': self.uniformity_formulas,
            'uniformity_method_var': self.uniformity_method_var,
            'selected_formula_text_var': self.selected_formula_text_var,
            'profile_t_max_var': self.profile_t_max_var,
            'profile_t_min_var': self.profile_t_min_var,
            'profile_t_mean_var': self.profile_t_mean_var,
            'smoothing_method_var': self.smoothing_method_var,
            'savgol_window_var': self.savgol_window_var,
            'savgol_polyorder_var': self.savgol_polyorder_var,
            'polyfit_degree_var': self.polyfit_degree_var,
            'display_percent_var': self.display_percent_var,
            'use_logscale_var': self.use_logscale_var,
            'show_raw_profile_var': self.show_raw_profile_var,
            'num_circular_profiles_var': self.num_circular_profiles_var,
            'linear_x_profile_y_offset_var': self.linear_x_profile_y_offset_var,
            'linear_y_profile_x_offset_var': self.linear_y_profile_x_offset_var,
            'roi_enabled_var': self.roi_enabled_var,
            'roi_show_on_map_var': self.roi_show_on_map_var,
            'roi_d_min_var': self.roi_d_min_var,
            'roi_d_max_var': self.roi_d_max_var,
            'roi_width_x_var': self.roi_width_x_var,
            'roi_height_y_var': self.roi_height_y_var,
            'roi_offset_x_var': self.roi_offset_x_var,
            'roi_offset_y_var': self.roi_offset_y_var,
            'cmap_manual_range_var': self.cmap_manual_range_var,
            'cmap_min_val_var': self.cmap_min_val_var,
            'cmap_max_val_var': self.cmap_max_val_var,
            'linear_profile_type_var': self.linear_profile_type_var,
            'auto_uniformity_mask_height_var': self.auto_uniformity_mask_height_var,
            'auto_uniformity_mode_var': self.auto_uniformity_mode_var,
            'loaded_files_text_var': self.loaded_files_text_var,
            'reconstruction_method_var': self.reconstruction_method_var,
        }
        
        callbacks_for_tabs = {
            'recalculate_callback': self.recalculate_callback,
            'export_excel_callback': self.export_excel_callback,
            'calculate_mask_callback': self.calculate_mask_callback,
            'load_profiles_callback': self.load_profiles_callback,
            '_internal_reconstruct_map_callback': self._internal_reconstruct_map_callback,
            '_on_settings_change': self._on_settings_change,
            '_on_settings_change_entry': self._on_settings_change_entry
        }

        validation_commands_for_tabs = {
            'int': self.vcmd_int, 
            'odd_int': self.vcmd_odd_int, 
            'float_positive_or_zero': self.vcmd_float_positive_or_zero,
            'float_general': self.vcmd_float_general,
            'num_circular_profiles': self.vcmd_num_circular_profiles,
            'digits_only': self.vcmd_digits_only,
            'digits_or_empty': self.vcmd_digits_or_empty, 
            'float_or_empty': self.vcmd_float_or_empty   
        }
        
        helper_methods_for_tabs = {
            'is_scipy_available': self._is_scipy_available,
            '_update_smoothing_options_display_internal': self._update_smoothing_options_display_internal,
            'config_constants': {
                'TARGET_DISK': TARGET_DISK,
                'TARGET_DOME': TARGET_DOME,
                'TARGET_PLANETARY': TARGET_PLANETARY,
                'TARGET_LINEAR': TARGET_LINEAR,
            }
        }

        self.main_tab_panel = MainPanelTab(tab_main_frame, settings_vars_for_tabs, callbacks_for_tabs, validation_commands_for_tabs, helper_methods_for_tabs)
        self.main_tab_panel.pack(expand=True, fill='both')

        self.auto_uniformity_tab_panel = AutoUniformityPanelTab(tab_auto_uniformity_frame, settings_vars_for_tabs, callbacks_for_tabs, validation_commands_for_tabs)
        self.auto_uniformity_tab_panel.pack(expand=True, fill='both')
        
        self.inverse_task_tab_panel = InverseTaskPanelTab(tab_inverse_task_frame, settings_vars_for_tabs, callbacks_for_tabs)
        self.inverse_task_tab_panel.pack(expand=True, fill='both')

    def _validate_positive_int(self, P_value): 
        if P_value == "": return True
        try: return int(P_value) > 0
        except ValueError: return False

    def _validate_odd_positive_int(self, P_value): 
        if P_value == "": return True
        try: val = int(P_value); return val > 0 and val % 2 != 0 and val >=3
        except ValueError: return False

    def _validate_positive_float_or_zero(self, P_value): 
        if P_value == "": return True
        if P_value == ".": return True
        if P_value.count('.') > 1: return False
        try: return float(P_value) >= 0.0
        except ValueError: return False
        
    def _validate_float_general(self, P_value): 
        if P_value == "": return True
        if P_value == "-": return True
        if P_value == ".": return True
        if P_value == "-.": return True
        if len(P_value) > 1 and P_value != "-.":
            if P_value.count('.') > 1: return False
        try: float(P_value); return True
        except ValueError: return False

    def _validate_num_circular_profiles(self, P_value, action_code, current_text, inserted_text):
        if action_code == '1': 
            if not inserted_text.isdigit(): return False
            try:
                widget = None
                if hasattr(self, 'main_tab_panel') and hasattr(self.main_tab_panel, 'entry_num_circular_profiles'):
                    widget = self.main_tab_panel.entry_num_circular_profiles
                
                if widget and widget.winfo_exists(): 
                    cursor_pos = widget.index(tk.INSERT)
                    temp_val_str = current_text[:cursor_pos] + inserted_text + current_text[cursor_pos:]
                else: 
                    temp_val_str = P_value 

                if len(temp_val_str) > 1 : return False
                val = int(temp_val_str)
                return 1 <= val <= 6
            except ValueError: return False
            except tk.TclError: return True 
        return True
        
    def _validate_digits_only(self, inserted_text, action_code): 
        if action_code == '1':
            return inserted_text.isdigit()
        return True
        
    def _validate_digits_or_empty(self, P_value): 
        return P_value == "" or P_value.isdigit()

    def _validate_float_or_empty(self, P_value): 
        if P_value == "" or P_value == "-" or P_value == "." or P_value == "-.": return True
        if P_value.count('.') > 1: return False
        try: float(P_value); return True
        except ValueError: return False

    def _is_scipy_available(self):
        try: from scipy.signal import savgol_filter; return True # noqa
        except ImportError: return False

    def _internal_reconstruct_map_callback(self):
        if self.reconstruct_map_callback:
            self.reconstruct_map_callback(self.reconstruction_method_var.get())

    def _initial_ui_update(self):
        self._update_formula_display_text_only_internal()
        if hasattr(self, 'main_tab_panel'):
             initial_target_type = TARGET_DISK 
             self.main_tab_panel.current_target_type = initial_target_type # Устанавливаем тип мишени в MainPanelTab
             self.main_tab_panel.update_all_field_states() # Вызываем единый метод обновления


    def _on_settings_change(self, event=None):
        self._update_formula_display_text_only_internal()
        if hasattr(self, 'main_tab_panel') and event and hasattr(self.main_tab_panel, 'combo_smoothing') and event.widget == self.main_tab_panel.combo_smoothing:
             self._update_smoothing_options_display_internal(
                 self.main_tab_panel.profile_config_outer_frame, 
                 self.main_tab_panel.savgol_frame,
                 self.main_tab_panel.label_polyfit_degree, 
                 self.main_tab_panel.entry_polyfit_degree, 
                 self.smoothing_method_var.get(),
                 self._get_current_smoothing_params_row()
             )
        if hasattr(self, 'main_tab_panel'):
            if event and hasattr(self.main_tab_panel, 'chk_roi_enabled') and \
               event.widget == self.main_tab_panel.chk_roi_enabled:
                # current_target_type уже должен быть актуален в main_tab_panel
                self.main_tab_panel.update_roi_fields_visibility_and_state(self.main_tab_panel.current_target_type) 
            
            if event and hasattr(self.main_tab_panel, 'linear_profile_type_radios') and \
               self.main_tab_panel.linear_profile_type_radios and \
               event.widget in self.main_tab_panel.linear_profile_type_radios.values():
                self.main_tab_panel.update_linear_profile_fields_state()

            if event and hasattr(self.main_tab_panel, 'chk_cmap_manual_range') and \
               event.widget == self.main_tab_panel.chk_cmap_manual_range:
                self.main_tab_panel.update_cmap_range_fields_state()


        if self.recalculate_callback and self._initialization_complete:
            self.recalculate_callback()

    def _on_settings_change_entry(self, event=None):
        if hasattr(self, '_after_id_recalc'):
            self.after_cancel(self._after_id_recalc)
        self._after_id_recalc = self.after(300, self._delayed_recalculate)

    def _delayed_recalculate(self):
        if self.recalculate_callback and self._initialization_complete:
            self.recalculate_callback()
            
    def _get_current_smoothing_params_row(self) -> int:
        if not hasattr(self, 'main_tab_panel') or not hasattr(self.main_tab_panel, 'combo_smoothing'): 
            return 2 
        
        combo_widget = self.main_tab_panel.combo_smoothing
        if combo_widget.winfo_exists() and combo_widget.winfo_manager() == "grid":
            try:
                return int(combo_widget.grid_info().get('row', 0)) + 1 
            except (KeyError, ValueError, tk.TclError): 
                pass 
        
        current_row_in_profile_config = 0
        try:
            if hasattr(self.main_tab_panel, 'label_num_circular_profiles') and \
               self.main_tab_panel.label_num_circular_profiles.winfo_ismapped() and \
               hasattr(self.main_tab_panel, 'entry_num_circular_profiles') and \
               self.main_tab_panel.entry_num_circular_profiles.winfo_exists() and \
               self.main_tab_panel.entry_num_circular_profiles.winfo_manager() == "grid":
                current_row_in_profile_config = max(current_row_in_profile_config, self.main_tab_panel.entry_num_circular_profiles.grid_info().get('row', -1) + 1)
            
            elif hasattr(self.main_tab_panel, 'label_linear_x_offset') and \
                 self.main_tab_panel.label_linear_x_offset.winfo_ismapped() and \
                 hasattr(self.main_tab_panel, 'entry_linear_x_offset') and \
                 self.main_tab_panel.entry_linear_x_offset.winfo_exists() and \
                 self.main_tab_panel.entry_linear_x_offset.winfo_manager() == "grid":
                
                base_row_linear = self.main_tab_panel.entry_linear_x_offset.grid_info().get('row', -1)
                
                if hasattr(self.main_tab_panel, 'linear_profile_type_radios_frame') and \
                   self.main_tab_panel.linear_profile_type_radios_frame.winfo_ismapped() and \
                   self.main_tab_panel.linear_profile_type_radios_frame.winfo_manager() == "grid":
                     base_row_linear = max(base_row_linear, self.main_tab_panel.linear_profile_type_radios_frame.grid_info().get('row', -1))

                if hasattr(self.main_tab_panel, 'label_linear_y_offset') and \
                   self.main_tab_panel.label_linear_y_offset.winfo_ismapped() and \
                   hasattr(self.main_tab_panel, 'entry_linear_y_offset') and \
                   self.main_tab_panel.entry_linear_y_offset.winfo_exists() and \
                   self.main_tab_panel.entry_linear_y_offset.winfo_manager() == "grid":
                    base_row_linear = max(base_row_linear, self.main_tab_panel.entry_linear_y_offset.grid_info().get('row', -1))
                current_row_in_profile_config = max(current_row_in_profile_config, base_row_linear + 1)
        except (tk.TclError, AttributeError, KeyError, ValueError):
             pass 
            
        row_for_combo_smoothing = current_row_in_profile_config
        return row_for_combo_smoothing + 1


    def _update_formula_display_text_only_internal(self):
        selected_key = self.uniformity_method_var.get()
        formula_text = self.uniformity_formulas.get(selected_key, "Неизвестная формула")
        self.selected_formula_text_var.set(formula_text)

    def _update_smoothing_options_display_internal(self, parent_frame, savgol_frame, 
                                                   polyfit_label, polyfit_entry, 
                                                   method: str, base_row_for_params: int) -> bool:
        gridded_a_frame = False 
        savgol_frame.grid_remove()
        polyfit_label.grid_remove()
        polyfit_entry.grid_remove()
        
        frame_padx = (0,0) 
        frame_pady = (2,2)
        label_padx = (0, 5) 
        entry_padx = (0, 5) 

        if method == "Savitzky-Golay":
            if self._is_scipy_available():
                savgol_frame.grid(in_=parent_frame, row=base_row_for_params, column=1, sticky=tk.EW, padx=frame_padx, pady=frame_pady)
                gridded_a_frame = True
        elif method == "Полином. аппрокс.":
            polyfit_label.grid(in_=parent_frame, row=base_row_for_params, column=0, sticky=tk.W, padx=label_padx, pady=frame_pady)
            polyfit_entry.grid(in_=parent_frame, row=base_row_for_params, column=1, sticky=tk.EW, padx=entry_padx, pady=frame_pady)
            gridded_a_frame = True
        return gridded_a_frame

    def mark_initialization_complete(self):
        self._initialization_complete = True

    def update_profile_options(self, target_type: str | None):
        if hasattr(self, 'main_tab_panel'):
            self.main_tab_panel.current_target_type = target_type 
            self.main_tab_panel.update_all_field_states() # Вызываем единый метод обновления
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
                if not self._is_scipy_available(): 
                    smoothing_params_data['method'] = "Без сглаживания"
                else:
                    win_str = self.savgol_window_var.get()
                    poly_str = self.savgol_polyorder_var.get()
                    
                    win_val = int(win_str if win_str else self._default_savgol_window)
                    poly_val = int(poly_str if poly_str else self._default_savgol_polyorder)
                    
                    specific_params['window_length'] = win_val
                    specific_params['polyorder'] = poly_val

                    if specific_params['window_length'] < 3 or specific_params['window_length'] % 2 == 0: 
                        raise ValueError("Окно SavGol должно быть >=3 и нечетным.")
                    if specific_params['polyorder'] >= specific_params['window_length']: 
                        raise ValueError("Порядок полинома SavGol должен быть меньше длины окна.")
            
            elif smoothing_params_data['method'] == "Полином. аппрокс.":
                deg_str = self.polyfit_degree_var.get()
                deg_val = int(deg_str if deg_str else self._default_polyfit_degree)
                
                specific_params['degree'] = deg_val
                if specific_params['degree'] < 1: 
                    raise ValueError("Степень полинома должна быть >= 1.")
            smoothing_params_data['params'] = specific_params
        except ValueError as e: 
            messagebox.showerror("Ошибка ввода", f"Настройки сглаживания: {e}", parent=self)
            return None

        profile_params = {}
        if hasattr(self, 'main_tab_panel') and hasattr(self.main_tab_panel, 'label_num_circular_profiles') and \
           self.main_tab_panel.label_num_circular_profiles.winfo_ismapped():
            num_str = self.num_circular_profiles_var.get()
            num_val_for_calc = int(self._default_num_circular_profiles) 
            if num_str:
                try:
                    num_val_entered = int(num_str)
                    corrected_val = min(max(1, num_val_entered), 6)
                    num_val_for_calc = corrected_val
                    if self.num_circular_profiles_var.get() != str(corrected_val):
                        self.num_circular_profiles_var.set(str(corrected_val)) 
                except ValueError:
                    self.num_circular_profiles_var.set(self._default_num_circular_profiles)
            profile_params['num_circular_profiles'] = num_val_for_calc
            
        elif hasattr(self, 'main_tab_panel') and hasattr(self.main_tab_panel, 'label_linear_x_offset') and \
             self.main_tab_panel.label_linear_x_offset.winfo_ismapped():
            profile_params['linear_profile_type'] = self.linear_profile_type_var.get()
            try:
                x_offset_str = self.linear_x_profile_y_offset_var.get()
                y_offset_str = self.linear_y_profile_x_offset_var.get()
                profile_params['linear_x_profile_y_offset'] = float(x_offset_str if x_offset_str and x_offset_str != "-" and x_offset_str != "-." else "0.0")
                profile_params['linear_y_profile_x_offset'] = float(y_offset_str if y_offset_str and y_offset_str != "-" and y_offset_str != "-." else "0.0")
            except ValueError: 
                self.linear_x_profile_y_offset_var.set("0.0") 
                self.linear_y_profile_x_offset_var.set("0.0") 
                profile_params['linear_x_profile_y_offset'] = 0.0
                profile_params['linear_y_profile_x_offset'] = 0.0
        
        roi_settings = {
            'enabled': self.roi_enabled_var.get(),
            'show_on_map': self.roi_show_on_map_var.get(),
            'type': None, 
            'params': {}
        }
        if roi_settings['enabled']:
            try:
                current_target_type = None 
                if hasattr(self, 'main_tab_panel') and hasattr(self.main_tab_panel, 'current_target_type'):
                    current_target_type = self.main_tab_panel.current_target_type

                if current_target_type in [TARGET_DISK, TARGET_DOME, TARGET_PLANETARY]:
                    roi_settings['type'] = 'circular'
                    d_min_str = self.roi_d_min_var.get()
                    d_max_str = self.roi_d_max_var.get()
                    d_min = float(d_min_str if d_min_str else self._default_roi_d_min)
                    d_max = float(d_max_str if d_max_str else self._default_roi_d_max)

                    if d_min < 0 or d_max < 0: raise ValueError("Диаметры ROI не могут быть отрицательными.")
                    if d_min >= d_max: raise ValueError("Мин. диаметр ROI должен быть меньше макс. диаметра.")
                    roi_settings['params'] = {'d_min': d_min, 'd_max': d_max}

                elif current_target_type == TARGET_LINEAR:
                    roi_settings['type'] = 'rectangular'
                    width_str = self.roi_width_x_var.get()
                    height_str = self.roi_height_y_var.get()
                    offset_x_str = self.roi_offset_x_var.get()
                    offset_y_str = self.roi_offset_y_var.get()

                    width = float(width_str if width_str else self._default_roi_width_x)
                    height = float(height_str if height_str else self._default_roi_height_y)
                    offset_x = float(offset_x_str if offset_x_str and offset_x_str != "-" and offset_x_str != "-." else self._default_roi_offset_x)
                    offset_y = float(offset_y_str if offset_y_str and offset_y_str != "-" and offset_y_str != "-." else self._default_roi_offset_y)

                    if width <= 0 or height <= 0: raise ValueError("Ширина и высота зоны ROI должны быть положительными.")
                    roi_settings['params'] = {'width': width, 'height': height, 'offset_x': offset_x, 'offset_y': offset_y}
            except ValueError as e:
                messagebox.showerror("Ошибка ввода ROI", str(e), parent=self)
                return None
        
        cmap_settings = {
            'manual_range': self.cmap_manual_range_var.get(),
            'min': None,
            'max': None
        }
        if cmap_settings['manual_range']:
            try:
                min_str = self.cmap_min_val_var.get()
                max_str = self.cmap_max_val_var.get()
                
                cmap_min = float(min_str if min_str and min_str != "-" and min_str != "-." else self._default_cmap_min)
                cmap_max = float(max_str if max_str and max_str != "-" and max_str != "-." else self._default_cmap_max)

                if cmap_min >= cmap_max: raise ValueError("Мин. значение карты должно быть меньше макс. значения.")
                cmap_settings['min'] = cmap_min
                cmap_settings['max'] = cmap_max
            except ValueError as e:
                messagebox.showerror("Ошибка ввода диапазона карты", str(e), parent=self)
                return None
            
        return {
            'smoothing': smoothing_params_data,
            'display_percent': self.display_percent_var.get(),
            'use_logscale': self.use_logscale_var.get(),
            'show_raw_profile': self.show_raw_profile_var.get(),
            'profile_config_params': profile_params,
            'uniformity_method_key': self.uniformity_method_var.get(),
            'roi': roi_settings, 
            'cmap': cmap_settings  
        }

    def get_auto_uniformity_params(self) -> dict | None: 
        try:
            height_str = self.auto_uniformity_mask_height_var.get()
            if not height_str or height_str == ".": raise ValueError("Высота маски не м.б. пустой или некорректной.")
            val = float(height_str)
            if val <0: raise ValueError("Высота маски не может быть отрицательной.")
            return {'mode': self.auto_uniformity_mode_var.get(), 'mask_height': val}
        except ValueError as e: 
            messagebox.showerror("Ошибка ввода", f"Некорректная высота маски: {e}", parent=self)
            self.auto_uniformity_mask_height_var.set("50.0")
            return None

    def get_reconstruction_method(self) -> str: return self.reconstruction_method_var.get()
    
    def update_loaded_files_text(self, text: str): self.loaded_files_text_var.set(text)
    
    def enable_export_button(self):
        if hasattr(self, 'main_tab_panel') and hasattr(self.main_tab_panel, 'btn_export_excel'):
            self.main_tab_panel.btn_export_excel.config(state=tk.NORMAL)
    
    def disable_export_button(self):
        if hasattr(self, 'main_tab_panel') and hasattr(self.main_tab_panel, 'btn_export_excel'):
            self.main_tab_panel.btn_export_excel.config(state=tk.DISABLED)


if __name__ == '__main__':
    root = tk.Tk()
    root.title("Тест SettingsPanel с разделенными вкладками")
    
    def mock_recalculate(): 
        settings = panel.get_view_settings()
        print("Вызван mock_recalculate. View Settings:", settings)
        if settings:
            print("ROI Params:", settings.get('roi'))
            print("Cmap Params:", settings.get('cmap'))

    def mock_export(): print("Вызван mock_export")
    def mock_calc_mask(): print("Вызван mock_calc_mask. Params:", panel.get_auto_uniformity_params())
    def mock_load_profiles(): print("Вызван mock_load_profiles"); panel.update_loaded_files_text("Загружен: test_profile.txt")
    def mock_reconstruct(method): print(f"Вызван mock_reconstruct с методом: {method}")

    initial_target_type = TARGET_DISK 

    panel = SettingsPanel(root, mock_recalculate, mock_export, mock_calc_mask, mock_load_profiles, mock_reconstruct)
    panel.mark_initialization_complete()
    panel.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def check_view_settings():
        settings = panel.get_view_settings()
        print("Ручной вызов get_view_settings():", settings)
        if settings:
            print("ROI Params:", settings.get('roi'))
            print("Cmap Params:", settings.get('cmap'))
    
    ttk.Button(root, text="Get View Settings (Manual)", command=check_view_settings).pack(pady=5)
    
    target_type_values = [TARGET_DISK, TARGET_LINEAR, TARGET_DOME, TARGET_PLANETARY]
    current_mock_target_type = tk.StringVar(value=initial_target_type)
    
    def switch_target_type(event=None):
        target_type = current_mock_target_type.get()
        print(f"\nПереключение на '{target_type}' тип мишени (имитация)")
        panel.update_profile_options(target_type) 
    
    mock_target_type_combo = ttk.Combobox(root, textvariable=current_mock_target_type,
                                          values=target_type_values,
                                          state='readonly')
    mock_target_type_combo.pack(pady=5)
    
    panel.update_profile_options(initial_target_type)
    panel.enable_export_button()

    root.mainloop()
