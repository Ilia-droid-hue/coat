# coating_simulator_project/coating_simulator/gui/frames/processing.py
"""
Frame for configuring processing parameters (RPM, speed, time).
Фрейм для настройки параметров обработки (RPM, скорость, время).
"""

import tkinter as tk
from tkinter import ttk, messagebox
import traceback

# Используем относительные импорты
from ... import config # Импортируем константы

class ProcessingFrame(ttk.LabelFrame):
    """
    LabelFrame containing widgets for processing configuration.
    Its appearance changes based on the selected target type.
    LabelFrame, содержащий виджеты для конфигурации обработки.
    Его вид изменяется в зависимости от выбранного типа мишени.
    """
    def __init__(self, master, target_frame_ref, emission_frame_ref, **kwargs):
        """
        Initializes the ProcessingFrame.
        Инициализирует ProcessingFrame.

        Args:
            master: Parent widget. Родительский виджет.
            target_frame_ref: Reference to the TargetFrame instance.
                              Ссылка на экземпляр TargetFrame.
            emission_frame_ref: Reference to the EmissionFrame instance.
                                Ссылка на экземпляр EmissionFrame.
            **kwargs: Additional arguments for ttk.LabelFrame.
                      Дополнительные аргументы для ttk.LabelFrame.
        """
        super().__init__(master, text="Режим обработки", **kwargs)
        self.columnconfigure(1, weight=1) # Allow entry column to expand

        self._target_frame = target_frame_ref
        self._emission_frame = emission_frame_ref

        # Validation commands
        vcmd_float = (self.register(self._validate_float), '%P')
        vcmd_time = (self.register(self._validate_positive_float), '%P') # Time must be positive

        # --- Widgets (created but gridded dynamically) ---
        # RPM 1 (Used for Disk RPM, Planet RPM)
        self.label_rpm1 = ttk.Label(self, text="RPM диска:") # Default text
        self.entry_rpm1 = ttk.Entry(self, validate='key', validatecommand=vcmd_float)
        self.entry_rpm1.insert(0, str(config.DEFAULT_PROCESSING_PARAMS["rpm"]))

        # RPM 2 (Used for Linear Speed, Orbit RPM)
        self.label_rpm2 = ttk.Label(self, text="Скорость (мм/с):") # Default text
        self.entry_rpm2 = ttk.Entry(self, validate='key', validatecommand=vcmd_float)
        self.entry_rpm2.insert(0, str(config.DEFAULT_PROCESSING_PARAMS["speed"]))

        # Time
        self.label_time = ttk.Label(self, text="Время (с):")
        self.entry_time = ttk.Entry(self, validate='key', validatecommand=vcmd_time)
        self.entry_time.insert(0, str(config.DEFAULT_PROCESSING_PARAMS["time"]))

        # Passes (Readonly, calculated for linear)
        self.label_passes = ttk.Label(self, text="Проходов:")
        self.entry_passes_var = tk.StringVar(value="N/A") # Use StringVar for easy update
        self.entry_passes = ttk.Entry(self, state="readonly", textvariable=self.entry_passes_var)

        # --- Bindings (for passes calculation) ---
        self.entry_rpm2.bind("<KeyRelease>", self._update_passes_if_linear)
        self.entry_time.bind("<KeyRelease>", self._update_passes_if_linear)

        # --- Initial State ---
        # Call update_layout initially based on the default target type from the passed reference
        # Вызываем update_layout изначально на основе типа мишени по умолчанию из переданной ссылки
        # This assumes target_frame_ref is valid upon initialization
        # Это предполагает, что target_frame_ref действителен при инициализации
        try:
             self.update_layout(self._target_frame.get_current_target_type())
        except AttributeError:
             print("Warning (ProcessingFrame init): Could not get initial target type from target_frame_ref.")
             # Optionally set a default layout if target_frame is not ready
             # Опционально установить макет по умолчанию, если target_frame не готов
             # self.update_layout(config.TARGET_DISK) # Example default

    def _validate_float(self, P):
        """Validation function: Allow empty string or valid float."""
        if P == "" or P == "-":
            return True
        # Allow '.' for starting decimal entry, but only one
        if P == ".": return True
        if P.count('.') > 1: return False
        try:
            float(P)
            return True
        except ValueError:
            return False

    def _validate_positive_float(self, P):
        """Validation function: Allow empty or positive float."""
        if P == "":
            return True
        # Allow '.' for starting decimal entry, but only one
        if P == ".": return True
        if P.count('.') > 1: return False
        try:
            val = float(P)
            return val >= 0 
        except ValueError:
            return False

    def _update_passes(self):
        """Calculates and updates the 'Passes' entry for linear target."""
        # Ensure target frame reference is valid
        # Убедимся, что ссылка на фрейм мишени действительна
        if not self._target_frame:
             self.entry_passes_var.set("Ошибка (нет ref)")
             return

        target_type = self._target_frame.get_current_target_type()
        if target_type != config.TARGET_LINEAR:
            self.entry_passes_var.set("N/A")
            return

        try:
            width_entry = self._target_frame.get_entry_widget('width')
            if not width_entry:
                 print("Warning: Could not find width entry in TargetFrame.")
                 self.entry_passes_var.set("Ошибка (нет width)")
                 return

            width_str = width_entry.get()
            speed_str = self.entry_rpm2.get()
            time_str = self.entry_time.get()

            if not width_str or not speed_str or not time_str:
                self.entry_passes_var.set("N/A") 
                return

            width = float(width_str)
            speed = float(speed_str)
            duration = float(time_str)

            if width <= 0:
                 self.entry_passes_var.set("N/A (Ширина > 0)")
                 return
            if duration < 0 or speed < 0: 
                self.entry_passes_var.set("N/A")
                return

            passes = (speed * duration) / width if width > 0 else 0
            self.entry_passes_var.set(f"{passes:.2f}")

        except (ValueError, TypeError) as e:
            self.entry_passes_var.set("Ошибка ввода") 
            print(f"Error calculating passes: {e}")
        except Exception:
             self.entry_passes_var.set("Ошибка")
             traceback.print_exc()

    def _update_passes_if_linear(self, event=None):
        """Helper to call _update_passes only if the target is linear."""
        if self._target_frame and self._target_frame.get_current_target_type() == config.TARGET_LINEAR:
            self._update_passes()

    def update_layout(self, target_type: str):
        """
        Updates the layout of the processing frame based on the target type.
        Обновляет макет фрейма обработки в зависимости от типа мишени.
        """
        # Ensure emission frame reference is valid before hiding/showing its parts
        # Убедимся, что ссылка на фрейм эмиссии действительна перед скрытием/показом его частей
        can_access_emission = hasattr(self, '_emission_frame') and self._emission_frame is not None

        # Hide all widgets first
        for widget in (self.label_rpm1, self.entry_rpm1, self.label_rpm2, self.entry_rpm2,
                       self.label_time, self.entry_time, self.label_passes, self.entry_passes):
            widget.grid_remove()
        
        if can_access_emission:
            self._emission_frame.hide_particles_entry()

        current_row = 0
        if target_type in [config.TARGET_DISK, config.TARGET_DOME]:
            self.label_rpm1.config(text="RPM диска:")
            self.label_rpm1.grid(row=current_row, column=0, sticky=tk.W, padx=(0, 5), pady=2)
            self.entry_rpm1.grid(row=current_row, column=1, sticky=tk.EW, pady=2)
            current_row += 1

            self.label_time.grid(row=current_row, column=0, sticky=tk.W, padx=(0, 5), pady=2)
            self.entry_time.grid(row=current_row, column=1, sticky=tk.EW, pady=2)
            current_row += 1

            if can_access_emission:
                self._emission_frame.show_particles_entry(row=current_row)

        elif target_type == config.TARGET_LINEAR:
            self.label_rpm2.config(text="Скорость (мм/с):")
            self.label_rpm2.grid(row=current_row, column=0, sticky=tk.W, padx=(0, 5), pady=2)
            self.entry_rpm2.grid(row=current_row, column=1, sticky=tk.EW, pady=2)
            current_row += 1

            self.label_time.grid(row=current_row, column=0, sticky=tk.W, padx=(0, 5), pady=2)
            self.entry_time.grid(row=current_row, column=1, sticky=tk.EW, pady=2)
            current_row += 1

            self.label_passes.grid(row=current_row, column=0, sticky=tk.W, padx=(0, 5), pady=2)
            self.entry_passes.grid(row=current_row, column=1, sticky=tk.EW, pady=2)
            self._update_passes() 
            current_row += 1

            if can_access_emission:
                self._emission_frame.show_particles_entry(row=current_row)

        elif target_type == config.TARGET_PLANETARY:
            self.label_rpm1.config(text="RPM планеты:")
            self.label_rpm1.grid(row=current_row, column=0, sticky=tk.W, padx=(0, 5), pady=2)
            self.entry_rpm1.grid(row=current_row, column=1, sticky=tk.EW, pady=2)
            # Clear and insert default value for planet RPM
            self.entry_rpm1.delete(0, tk.END)
            self.entry_rpm1.insert(0, str(config.DEFAULT_PROCESSING_PARAMS["rpm_disk"]))
            current_row += 1

            self.label_rpm2.config(text="RPM орбиты:")
            self.label_rpm2.grid(row=current_row, column=0, sticky=tk.W, padx=(0, 5), pady=2)
            self.entry_rpm2.grid(row=current_row, column=1, sticky=tk.EW, pady=2)
            # Clear and insert default value for orbit RPM
            self.entry_rpm2.delete(0, tk.END)
            self.entry_rpm2.insert(0, str(config.DEFAULT_PROCESSING_PARAMS["rpm_orbit"]))
            current_row += 1

            self.label_time.grid(row=current_row, column=0, sticky=tk.W, padx=(0, 5), pady=2)
            self.entry_time.grid(row=current_row, column=1, sticky=tk.EW, pady=2)
            current_row += 1

            if can_access_emission:
                self._emission_frame.show_particles_entry(row=current_row)
        else:
             if can_access_emission:
                 self._emission_frame.hide_particles_entry()

    def get_params(self) -> dict:
        """
        Retrieves the configured processing parameters based on target type.
        Извлекает настроенные параметры обработки в зависимости от типа мишени.
        """
        params = {}
        # Ensure target frame reference is valid
        if not self._target_frame:
             messagebox.showerror("Ошибка (Обработка)", "Не удалось получить ссылку на параметры мишени.", parent=self)
             raise RuntimeError("Missing target frame reference in ProcessingFrame")

        target_type = self._target_frame.get_current_target_type()

        try:
            params['time'] = float(self.entry_time.get())
            if params['time'] < 0: 
                raise ValueError("Время не может быть отрицательным.")

            if target_type in [config.TARGET_DISK, config.TARGET_DOME]:
                params['rpm'] = float(self.entry_rpm1.get())
            elif target_type == config.TARGET_LINEAR:
                params['speed'] = float(self.entry_rpm2.get())
                if params['speed'] < 0:
                     raise ValueError("Скорость не может быть отрицательной.")
            elif target_type == config.TARGET_PLANETARY:
                params['rpm_disk'] = float(self.entry_rpm1.get()) 
                params['rpm_orbit'] = float(self.entry_rpm2.get()) 

        except ValueError as e:
            messagebox.showerror("Ошибка ввода (Обработка)", f"Некорректное значение в параметрах обработки: {e}", parent=self)
            raise ValueError(f"Invalid processing parameter: {e}") from e
        except Exception as e:
             messagebox.showerror("Ошибка (Обработка)", f"Неожиданная ошибка при чтении параметров обработки: {e}", parent=self)
             raise RuntimeError(f"Unexpected error getting processing params: {e}") from e

        return params

# Example usage (requires mock TargetFrame and EmissionFrame)
# --- ИСПРАВЛЕННЫЙ ТЕСТОВЫЙ БЛОК ---
if __name__ == '__main__':
    root = tk.Tk()
    root.title("Тест ProcessingFrame")

    # --- Mock Frames (Simplified) ---
    class MockTargetFrame(ttk.Frame):
        def __init__(self, master):
            super().__init__(master)
            self.combo_tgt = ttk.Combobox(self, values=config.TARGET_TYPES, state="readonly")
            self.combo_tgt.set(config.TARGET_LINEAR) # Start with linear
            self.combo_tgt.pack(pady=5)
            self.entry_width = ttk.Entry(self)
            self.entry_width.insert(0, str(config.DEFAULT_TARGET_PARAMS["width"]))
            self.entry_width.pack(pady=5)
            
            # Store reference to processing frame *after* it's created
            self.processing_frame_ref = None 
            
            self.combo_tgt.bind("<<ComboboxSelected>>", self.notify_processing)
            self.entry_width.bind("<KeyRelease>", self.notify_processing)

        def get_current_target_type(self):
            return self.combo_tgt.get()

        def get_entry_widget(self, name):
             if name == 'width':
                 return self.entry_width
             return None

        def set_processing_frame_ref(self, processing_frame):
             self.processing_frame_ref = processing_frame

        def notify_processing(self, event=None):
             # Check if the reference exists before calling methods
             if self.processing_frame_ref:
                 current_type = self.get_current_target_type()
                 print(f"DEBUG (MockTarget): Notifying processing frame. Type: {current_type}")
                 # Call update_layout directly on the referenced instance
                 self.processing_frame_ref.update_layout(current_type)
                 # Specifically trigger passes update if width changed
                 if event and event.widget == self.entry_width:
                      # Call _update_passes_if_linear directly
                      self.processing_frame_ref._update_passes_if_linear()
             else:
                 print("DEBUG (MockTarget): Processing frame reference not set yet.")


    class MockEmissionFrame(ttk.Frame):
        def __init__(self, master):
            super().__init__(master)
            self.label_parts = ttk.Label(self, text="Частиц (Mock):")
            self.entry_parts = ttk.Entry(self)
            self.entry_parts.insert(0, str(config.DEFAULT_EMISSION_PARAMS["particles"]))
            # Keep track if widgets are gridded to avoid errors on remove
            self._is_particles_gridded = False 

        def show_particles_entry(self, row):
            # Grid in the *actual* parent (ProcessingFrame)
            parent_frame = self.master 
            print(f"DEBUG (MockEmission): Showing particles at row {row} in {parent_frame}")
            self.label_parts.grid(row=row, column=0, sticky=tk.W, padx=(0, 5), pady=2, in_=parent_frame) 
            self.entry_parts.grid(row=row, column=1, sticky=tk.EW, pady=2, in_=parent_frame)
            self._is_particles_gridded = True

        def hide_particles_entry(self):
            # Only remove if gridded
            if self._is_particles_gridded:
                print("DEBUG (MockEmission): Hiding particles")
                self.label_parts.grid_remove()
                self.entry_parts.grid_remove()
                self._is_particles_gridded = False
            else:
                print("DEBUG (MockEmission): Particles already hidden.")
                
    # --- End Mock Frames ---

    mock_target = MockTargetFrame(root)
    # Create mock emission *before* processing frame, but don't pack/grid it yet
    mock_emission = MockEmissionFrame(root) 

    # Create the actual ProcessingFrame, passing references
    processing_frame = ProcessingFrame(root, 
                                       target_frame_ref=mock_target, 
                                       emission_frame_ref=mock_emission, 
                                       padding=10)
    
    # Now set the reference in mock_target
    mock_target.set_processing_frame_ref(processing_frame)
    
    # Set the correct master for mock_emission *after* processing_frame is created
    mock_emission.master = processing_frame 

    # Pack/Grid the main frames
    mock_target.pack(pady=5, fill=tk.X)
    processing_frame.pack(expand=True, fill=tk.BOTH, pady=5)


    def print_params():
        try:
            params = processing_frame.get_params()
            print("Параметры обработки:", params)
        except (ValueError, RuntimeError) as e:
            print("Ошибка получения параметров:", e)

    button = ttk.Button(root, text="Получить параметры обработки", command=print_params)
    button.pack(pady=10)

    # Initial layout update called from ProcessingFrame.__init__ should work now
    # If not, uncomment the line below for explicit initial call
    # mock_target.notify_processing() 

    root.mainloop()
# --- КОНЕЦ ИСПРАВЛЕННОГО ТЕСТОВОГО БЛОКА ---
