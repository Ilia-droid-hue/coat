# coating_simulator_project/coating_simulator/gui/frames/processing.py
"""
Frame for configuring processing parameters (RPM, speed, time, mini-batch size).
For linear target, Time and Passes are now interdependent.
Improved error handling and dependency updates.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import traceback
import math

# Используем относительные импорты
from ... import config # Импортируем константы

class ProcessingFrame(ttk.LabelFrame):
    """
    LabelFrame containing widgets for processing configuration.
    Its appearance changes based on the selected target type.
    """
    def __init__(self, master, target_frame_ref, emission_frame_ref, **kwargs):
        super().__init__(master, text="Режим обработки", **kwargs)
        self.columnconfigure(1, weight=1)

        self._target_frame = target_frame_ref
        self._emission_frame = emission_frame_ref
        self._last_edited_linear_param = 'time' 

        vcmd_float = (self.register(self._validate_float), '%P')
        vcmd_time_or_passes = (self.register(self._validate_positive_float_or_empty), '%P')
        vcmd_int_positive = (self.register(self._validate_positive_int), '%P')

        self.label_rpm1 = ttk.Label(self, text="RPM диска:")
        self.entry_rpm1 = ttk.Entry(self, validate='key', validatecommand=vcmd_float)
        self.entry_rpm1.insert(0, str(config.DEFAULT_PROCESSING_PARAMS["rpm"]))

        self.label_rpm2 = ttk.Label(self, text="Скорость (мм/с):")
        self.entry_rpm2_var = tk.StringVar(value=str(config.DEFAULT_PROCESSING_PARAMS["speed"]))
        self.entry_rpm2 = ttk.Entry(self, textvariable=self.entry_rpm2_var, validate='key', validatecommand=vcmd_float)

        self.label_time = ttk.Label(self, text="Время (с):")
        self.entry_time_var = tk.StringVar(value=str(config.DEFAULT_PROCESSING_PARAMS["time"]))
        self.entry_time = ttk.Entry(self, textvariable=self.entry_time_var, validate='key', validatecommand=vcmd_time_or_passes)

        self.label_passes = ttk.Label(self, text="Проходов:")
        self.entry_passes_var = tk.StringVar(value="N/A")
        self.entry_passes = ttk.Entry(self, textvariable=self.entry_passes_var, validate='key', validatecommand=vcmd_time_or_passes)

        self.label_mini_batch = ttk.Label(self, text="Шаг пересчета трансф.:")
        self.entry_mini_batch = ttk.Entry(self, validate='key', validatecommand=vcmd_int_positive)
        self.entry_mini_batch.insert(0, str(config.DEFAULT_PROCESSING_PARAMS["mini_batch_size"]))

        self.entry_time_var.trace_add("write", self._on_time_changed_by_user)
        self.entry_passes_var.trace_add("write", self._on_passes_changed_by_user)
        self.entry_rpm2_var.trace_add("write", self._on_speed_changed)

        self._is_updating_time = False
        self._is_updating_passes = False

        try:
             self.update_layout(self._target_frame.get_current_target_type())
        except AttributeError:
             print("Warning (ProcessingFrame init): Could not get initial target type from target_frame_ref.")

    def _validate_float(self, P):
        if P == "" or P == "-": return True
        if P == ".": return True
        if P.count('.') > 1: return False
        try: float(P); return True
        except ValueError: return False

    def _validate_positive_float_or_empty(self, P):
        if P == "": return True
        if P == ".": return True
        if P.count('.') > 1: return False
        try: return float(P) >= 0
        except ValueError: return False
        
    def _validate_positive_int(self, P):
        if P == "": return True
        try: return int(P) > 0
        except ValueError: return False

    def _on_time_changed_by_user(self, *args):
        if self._is_updating_time: return
        self._last_edited_linear_param = 'time'
        self._calculate_and_update_dependent_linear_param()

    def _on_passes_changed_by_user(self, *args):
        if self._is_updating_passes: return
        self._last_edited_linear_param = 'passes'
        self._calculate_and_update_dependent_linear_param()

    def _on_speed_changed(self, *args):
        self._calculate_and_update_dependent_linear_param()

    def _calculate_and_update_dependent_linear_param(self):
        if not self._target_frame or self._target_frame.get_current_target_type() != config.TARGET_LINEAR:
            if hasattr(self, 'entry_passes_var'):
                 self.entry_passes_var.set("N/A")
            return

        if self._is_updating_time or self._is_updating_passes:
            return

        try:
            target_params = self._target_frame.get_params()
            if not target_params:
                self._is_updating_time = True; self.entry_time_var.set(""); self._is_updating_time = False
                self._is_updating_passes = True; self.entry_passes_var.set(""); self._is_updating_passes = False
                return

            length_str = str(target_params.get('length', ''))
            speed_str = self.entry_rpm2_var.get()
            time_str = self.entry_time_var.get()
            passes_str = self.entry_passes_var.get()

            valid_length = False
            length = 0.0
            try:
                length = float(length_str)
                if length > 0:
                    valid_length = True
            except ValueError:
                pass # length_str is empty or not a number

            valid_speed = False
            speed = 0.0
            try:
                speed = float(speed_str)
                if speed >= 0: # Speed can be 0
                    valid_speed = True
            except ValueError:
                pass

            if not valid_length:
                self._is_updating_time = True; self.entry_time_var.set(""); self._is_updating_time = False
                self._is_updating_passes = True; self.entry_passes_var.set("N/A (длина!)"); self._is_updating_passes = False
                return
            
            if not valid_speed:
                self._is_updating_time = True; self.entry_time_var.set(""); self._is_updating_time = False
                self._is_updating_passes = True; self.entry_passes_var.set("N/A (скорость!)"); self._is_updating_passes = False
                return

            if self._last_edited_linear_param == 'time':
                if time_str:
                    try:
                        time_val = float(time_str)
                        if time_val < 0: raise ValueError("Время < 0")
                        
                        if speed > 1e-9: # Effectively non-zero speed
                            passes_calc = (speed * time_val) / length
                            self._is_updating_passes = True
                            self.entry_passes_var.set(f"{passes_calc:.2f}")
                            self._is_updating_passes = False
                        else: # Speed is zero
                            self._is_updating_passes = True
                            self.entry_passes_var.set("0.00" if time_val >= 0 else "N/A (время)")
                            self._is_updating_passes = False
                    except ValueError:
                        self._is_updating_passes = True; self.entry_passes_var.set(""); self._is_updating_passes = False
                else: # Time field is empty
                    self._is_updating_passes = True; self.entry_passes_var.set(""); self._is_updating_passes = False
            
            elif self._last_edited_linear_param == 'passes':
                if passes_str and passes_str.lower() != "n/a":
                    try:
                        passes_val = float(passes_str)
                        if passes_val < 0: raise ValueError("Проходы < 0")

                        if speed > 1e-9: # Speed must be positive to calculate time from passes
                            time_calc = (passes_val * length) / speed
                            self._is_updating_time = True
                            self.entry_time_var.set(f"{time_calc:.2f}")
                            self._is_updating_time = False
                        elif math.isclose(passes_val, 0): # If 0 passes, time is 0, regardless of speed
                             self._is_updating_time = True
                             self.entry_time_var.set("0.00")
                             self._is_updating_time = False
                        else: # Cannot calculate time if speed is zero and passes > 0
                            self._is_updating_time = True; self.entry_time_var.set("N/A (скорость)"); self._is_updating_time = False
                    except ValueError:
                        self._is_updating_time = True; self.entry_time_var.set(""); self._is_updating_time = False
                else: # Passes field is empty or N/A
                    self._is_updating_time = True; self.entry_time_var.set(""); self._is_updating_time = False
        
        except Exception as e:
            print(f"Error in _calculate_and_update_dependent_linear_param: {e}")
            traceback.print_exc()
            self._is_updating_time = True; self.entry_time_var.set(""); self._is_updating_time = False
            self._is_updating_passes = True; self.entry_passes_var.set("Ошибка"); self._is_updating_passes = False


    def update_layout(self, target_type: str):
        can_access_emission = hasattr(self, '_emission_frame') and self._emission_frame is not None
        
        for widget in (self.label_rpm1, self.entry_rpm1, self.label_rpm2, self.entry_rpm2,
                       self.label_time, self.entry_time, self.label_passes, self.entry_passes,
                       self.label_mini_batch, self.entry_mini_batch):
            widget.grid_remove()
        
        if can_access_emission: self._emission_frame.hide_particles_entry()
        
        self.entry_passes.config(state="disabled")

        current_row = 0
        if target_type in [config.TARGET_DISK, config.TARGET_DOME, config.TARGET_PLANETARY]:
            if target_type == config.TARGET_PLANETARY:
                self.label_rpm1.config(text="RPM планеты:")
                self.entry_rpm1.delete(0, tk.END); self.entry_rpm1.insert(0, str(config.DEFAULT_PROCESSING_PARAMS["rpm_disk"]))
                self.label_rpm2.config(text="RPM орбиты:")
                self.entry_rpm2_var.set(str(config.DEFAULT_PROCESSING_PARAMS["rpm_orbit"]))
            else: 
                self.label_rpm1.config(text="RPM диска:")
                self.entry_rpm1.delete(0, tk.END); self.entry_rpm1.insert(0, str(config.DEFAULT_PROCESSING_PARAMS["rpm"]))
                self.label_rpm2.grid_remove()
                self.entry_rpm2.grid_remove()

            self.label_rpm1.grid(row=current_row, column=0, sticky=tk.W, padx=(0, 5), pady=2)
            self.entry_rpm1.grid(row=current_row, column=1, sticky=tk.EW, pady=2); current_row += 1
            
            if target_type == config.TARGET_PLANETARY:
                self.label_rpm2.grid(row=current_row, column=0, sticky=tk.W, padx=(0, 5), pady=2)
                self.entry_rpm2.grid(row=current_row, column=1, sticky=tk.EW, pady=2); current_row += 1

            self.label_time.grid(row=current_row, column=0, sticky=tk.W, padx=(0, 5), pady=2)
            self.entry_time.grid(row=current_row, column=1, sticky=tk.EW, pady=2); current_row += 1
            
            self.label_mini_batch.grid(row=current_row, column=0, sticky=tk.W, padx=(0,5), pady=2)
            self.entry_mini_batch.grid(row=current_row, column=1, sticky=tk.EW, pady=2); current_row += 1
            if can_access_emission: self._emission_frame.show_particles_entry()

        elif target_type == config.TARGET_LINEAR:
            self.label_rpm2.config(text="Скорость (мм/с):")
            self.entry_rpm2_var.set(str(config.DEFAULT_PROCESSING_PARAMS["speed"]))
            self.label_rpm2.grid(row=current_row, column=0, sticky=tk.W, padx=(0, 5), pady=2)
            self.entry_rpm2.grid(row=current_row, column=1, sticky=tk.EW, pady=2); current_row += 1
            
            self.label_time.grid(row=current_row, column=0, sticky=tk.W, padx=(0, 5), pady=2)
            self.entry_time.grid(row=current_row, column=1, sticky=tk.EW, pady=2); current_row += 1
            
            self.label_passes.grid(row=current_row, column=0, sticky=tk.W, padx=(0, 5), pady=2)
            self.entry_passes.config(state="normal")
            self.entry_passes.grid(row=current_row, column=1, sticky=tk.EW, pady=2); current_row += 1
            
            self.label_mini_batch.grid(row=current_row, column=0, sticky=tk.W, padx=(0,5), pady=2)
            self.entry_mini_batch.grid(row=current_row, column=1, sticky=tk.EW, pady=2); current_row += 1
            
            if can_access_emission: self._emission_frame.show_particles_entry()
            self._last_edited_linear_param = 'time' # При переключении на линейный, считаем, что время было "отредактировано"
            self._calculate_and_update_dependent_linear_param()

        else:
             if can_access_emission: self._emission_frame.hide_particles_entry()

    def get_params(self) -> dict:
        params = {}
        if not self._target_frame:
             messagebox.showerror("Ошибка (Обработка)", "Не удалось получить ссылку на параметры мишени.", parent=self)
             raise RuntimeError("Missing target frame reference in ProcessingFrame")
        target_type = self._target_frame.get_current_target_type()
        try:
            time_str = self.entry_time_var.get()
            if not time_str or time_str.lower() == "n/a": raise ValueError("Время не может быть пустым или N/A.") # Добавил проверку на N/A
            params['time'] = float(time_str)
            if params['time'] < 0: raise ValueError("Время не может быть отрицательным.")

            mini_batch_str = self.entry_mini_batch.get()
            if not mini_batch_str: raise ValueError("Шаг пересчета трансформаций не может быть пустым.")
            params['mini_batch_size'] = int(mini_batch_str)
            if params['mini_batch_size'] <= 0: raise ValueError("Шаг пересчета трансформаций должен быть положительным.")

            if target_type in [config.TARGET_DISK, config.TARGET_DOME]:
                rpm1_str = self.entry_rpm1.get()
                if not rpm1_str: raise ValueError("RPM диска не может быть пустым.")
                params['rpm'] = float(rpm1_str)
            elif target_type == config.TARGET_LINEAR:
                speed_str = self.entry_rpm2_var.get()
                if not speed_str: raise ValueError("Скорость не может быть пустой.")
                params['speed'] = float(speed_str)
                if params['speed'] < 0: raise ValueError("Скорость не может быть отрицательной.")
            elif target_type == config.TARGET_PLANETARY:
                rpm_disk_str = self.entry_rpm1.get()
                rpm_orbit_str = self.entry_rpm2_var.get()
                if not rpm_disk_str: raise ValueError("RPM планеты не может быть пустым.")
                if not rpm_orbit_str: raise ValueError("RPM орбиты не может быть пустым.")
                params['rpm_disk'] = float(rpm_disk_str)
                params['rpm_orbit'] = float(rpm_orbit_str)
        except ValueError as e:
            messagebox.showerror("Ошибка ввода (Обработка)", f"Некорректное значение в параметрах обработки: {e}", parent=self)
            raise ValueError(f"Invalid processing parameter: {e}") from e
        except Exception as e:
             messagebox.showerror("Ошибка (Обработка)", f"Неожиданная ошибка при чтении параметров обработки: {e}", parent=self)
             raise RuntimeError(f"Unexpected error getting processing params: {e}") from e
        return params

if __name__ == '__main__':
    root = tk.Tk()
    root.title("Тест ProcessingFrame - Взаимозависимые Поля")
    class MockTargetFrame(ttk.Frame):
        def __init__(self, master):
            super().__init__(master)
            self.current_type = tk.StringVar(value=config.TARGET_LINEAR)
            self.length_var = tk.StringVar(value="100.0")
            ttk.Label(self, text="Mock Target:").pack()
            self.combo_tgt = ttk.Combobox(self, textvariable=self.current_type, values=config.TARGET_TYPES, state="readonly")
            self.combo_tgt.pack(pady=5)
            len_frame = ttk.Frame(self)
            ttk.Label(len_frame, text="Длина (L):").pack(side=tk.LEFT)
            self.entry_length = ttk.Entry(len_frame, textvariable=self.length_var, width=7)
            self.entry_length.pack(side=tk.LEFT)
            len_frame.pack()
            self.processing_frame_ref = None
            self.combo_tgt.bind("<<ComboboxSelected>>", self.notify_processing_type_change)
            self.length_var.trace_add("write", self.notify_processing_param_change)
        def get_current_target_type(self): return self.current_type.get()
        def get_params(self):
            try: return {'length': float(self.length_var.get()) if self.length_var.get() else 0.0, 'width': 50.0, 'target_type': self.get_current_target_type()}
            except ValueError: return None
        def get_entry_widget(self, name): return self.entry_length if name == 'length' else None
        def set_processing_frame_ref(self, processing_frame): self.processing_frame_ref = processing_frame
        def notify_processing_type_change(self, event=None):
             if self.processing_frame_ref: self.processing_frame_ref.update_layout(self.get_current_target_type())
        def notify_processing_param_change(self, *args):
            if self.processing_frame_ref: self.processing_frame_ref._calculate_and_update_dependent_linear_param()
    class MockEmissionFrame(ttk.Frame):
        def __init__(self, master): super().__init__(master)
        def show_particles_entry(self): pass
        def hide_particles_entry(self): pass
    mock_target = MockTargetFrame(root)
    mock_emission = MockEmissionFrame(None) 
    processing_frame = ProcessingFrame(root, target_frame_ref=mock_target, emission_frame_ref=mock_emission, padding=10)
    mock_target.set_processing_frame_ref(processing_frame)
    mock_emission.master = processing_frame 
    mock_target.pack(pady=10, fill=tk.X)
    processing_frame.pack(expand=True, fill=tk.BOTH, pady=10)
    mock_target.notify_processing_type_change()
    def print_proc_params():
        try:
            params = processing_frame.get_params()
            print("Параметры обработки:", params)
        except (ValueError, RuntimeError) as e: print("Ошибка получения параметров:", e)
    button = ttk.Button(root, text="Получить параметры обработки", command=print_proc_params)
    button.pack(pady=10)
    root.mainloop()
