# coating_simulator_project/coating_simulator/gui/frames/target.py
"""
Frame for selecting target type and configuring its parameters.
Фрейм для выбора типа мишени и настройки ее параметров.
Added binding for length entry to trigger updates.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import traceback # For detailed error logging

# Используем относительные импорты
from ... import config # Импортируем константы

class TargetFrame(ttk.LabelFrame):
    """
    LabelFrame containing widgets for target configuration.
    LabelFrame, содержащий виджеты для конфигурации мишени.
    """
    def __init__(self, master, target_update_callback=None, **kwargs):
        """
        Initializes the TargetFrame.
        Инициализирует TargetFrame.

        Args:
            master: Parent widget. Родительский виджет.
            target_update_callback: Function to call when target type or relevant dimensions change.
                                    Функция, вызываемая при изменении типа мишени или релевантных размеров.
            **kwargs: Additional arguments for ttk.LabelFrame.
                      Дополнительные аргументы для ttk.LabelFrame.
        """
        super().__init__(master, text="Параметры мишени", **kwargs)
        self.columnconfigure(1, weight=1)

        self._target_update_callback = target_update_callback
        vcmd = (self.register(self._validate_float), '%P')

        ttk.Label(self, text="Тип мишени:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.combo_tgt = ttk.Combobox(self, values=config.TARGET_TYPES, state="readonly", width=20)
        self.combo_tgt.set(config.TARGET_DISK)
        self.combo_tgt.grid(row=0, column=1, sticky=tk.EW)

        self.widgets_map = {}

        self.label_disk_diameter = ttk.Label(self, text="Диаметр (мм):")
        self.entry_disk_diameter = ttk.Entry(self, validate='key', validatecommand=vcmd)
        self.entry_disk_diameter.insert(0, str(config.DEFAULT_TARGET_PARAMS["diameter"]))
        self.widgets_map[config.TARGET_DISK] = [
            (self.label_disk_diameter, self.entry_disk_diameter)
        ]

        self.label_dome_radius = ttk.Label(self, text="Радиус купола (мм):")
        self.entry_dome_radius = ttk.Entry(self, validate='key', validatecommand=vcmd)
        self.entry_dome_radius.insert(0, str(config.DEFAULT_TARGET_PARAMS["dome_radius"]))
        self.widgets_map[config.TARGET_DOME] = [
            (self.label_disk_diameter, self.entry_disk_diameter),
            (self.label_dome_radius, self.entry_dome_radius)
        ]

        self.label_length = ttk.Label(self, text="Длина (мм):")
        self.entry_length = ttk.Entry(self, validate='key', validatecommand=vcmd)
        self.entry_length.insert(0, str(config.DEFAULT_TARGET_PARAMS["length"]))

        self.label_width = ttk.Label(self, text="Ширина (мм):")
        self.entry_width = ttk.Entry(self, validate='key', validatecommand=vcmd)
        self.entry_width.insert(0, str(config.DEFAULT_TARGET_PARAMS["width"]))
        self.widgets_map[config.TARGET_LINEAR] = [
            (self.label_length, self.entry_length),
            (self.label_width, self.entry_width)
        ]
        
        # --- ИЗМЕНЕНИЕ: Привязка для длины и ширины ---
        self.entry_length.bind("<KeyRelease>", self._on_param_change)
        self.entry_width.bind("<KeyRelease>", self._on_param_change)
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---


        self.label_orbit_diameter = ttk.Label(self, text="Диаметр орбиты (мм):")
        self.entry_orbit_diameter = ttk.Entry(self, validate='key', validatecommand=vcmd)
        self.entry_orbit_diameter.insert(0, str(config.DEFAULT_TARGET_PARAMS["orbit_diameter"]))

        self.label_planet_disk_diameter = ttk.Label(self, text="Диаметр диска планеты (мм):")
        self.entry_planet_disk_diameter = ttk.Entry(self, validate='key', validatecommand=vcmd)
        self.entry_planet_disk_diameter.insert(0, str(config.DEFAULT_TARGET_PARAMS["planet_disk_diameter"]))
        self.widgets_map[config.TARGET_PLANETARY] = [
            (self.label_orbit_diameter, self.entry_orbit_diameter),
            (self.label_planet_disk_diameter, self.entry_planet_disk_diameter)
        ]

        self.combo_tgt.bind("<<ComboboxSelected>>", self._update_fields)
        # Привязываем также изменение диаметра, так как он используется для всех нелинейных типов
        self.entry_disk_diameter.bind("<KeyRelease>", self._on_param_change)
        self.entry_dome_radius.bind("<KeyRelease>", self._on_param_change)
        self.entry_orbit_diameter.bind("<KeyRelease>", self._on_param_change)
        self.entry_planet_disk_diameter.bind("<KeyRelease>", self._on_param_change)

        self._update_fields()

    def _validate_float(self, P):
        if P == "": return True
        try: float(P); return True
        except ValueError: return False

    def _on_param_change(self, event=None):
        """Callback when a target parameter entry changes."""
        if self._target_update_callback:
            try:
                # Передаем текущий тип мишени и виджет, который изменился
                self._target_update_callback(self.combo_tgt.get(), event.widget if event else None)
            except Exception:
                 print("Error in target update callback:")
                 traceback.print_exc()

    def _update_fields(self, event=None):
        selected_type = self.combo_tgt.get()
        for target_type_key_iter in self.widgets_map: # Изменено имя переменной цикла
            for label, entry in self.widgets_map[target_type_key_iter]:
                label.grid_remove()
                entry.grid_remove()

        if selected_type in self.widgets_map:
            row_index = 1
            for label, entry in self.widgets_map[selected_type]:
                label.grid(row=row_index, column=0, sticky=tk.W, padx=(0, 5), pady=2)
                entry.grid(row=row_index, column=1, sticky=tk.EW, pady=2)
                row_index += 1
        
        # Уведомляем App, так как тип мог измениться или поля для текущего типа обновились
        # Это важно, чтобы ProcessingFrame мог обновить свой layout
        if self._target_update_callback:
             try:
                 # Передаем combobox как "изменившийся" виджет при смене типа
                 changed_widget = self.combo_tgt if event and event.widget == self.combo_tgt else None
                 self._target_update_callback(selected_type, changed_widget)
             except Exception:
                 print("Error in target update callback during _update_fields:")
                 traceback.print_exc()

    def get_params(self) -> dict:
        params = {}
        target_type = self.combo_tgt.get()
        params['target_type'] = target_type
        try:
            if target_type == config.TARGET_DISK:
                params['diameter'] = float(self.entry_disk_diameter.get())
            elif target_type == config.TARGET_DOME:
                params['diameter'] = float(self.entry_disk_diameter.get())
                params['dome_radius'] = float(self.entry_dome_radius.get())
            elif target_type == config.TARGET_LINEAR:
                length_str = self.entry_length.get()
                width_str = self.entry_width.get()
                if not length_str: raise ValueError("Длина для линейной мишени не может быть пустой.")
                if not width_str: raise ValueError("Ширина для линейной мишени не может быть пустой.")
                params['length'] = float(length_str)
                params['width'] = float(width_str)
            elif target_type == config.TARGET_PLANETARY:
                params['orbit_diameter'] = float(self.entry_orbit_diameter.get())
                params['planet_diameter'] = float(self.entry_planet_disk_diameter.get())
            else:
                raise ValueError(f"Неизвестный тип мишени: {target_type}")

            for key, value in params.items():
                if isinstance(value, (float, int)) and value <= 0 and key != 'target_type':
                     widget_name_map = {
                         'diameter': "Диаметр", 'dome_radius': "Радиус купола",
                         'length': "Длина", 'width': "Ширина",
                         'orbit_diameter': "Диаметр орбиты",
                         'planet_diameter': "Диаметр диска планеты"
                     }
                     widget_name = widget_name_map.get(key, key)
                     raise ValueError(f"Параметр '{widget_name}' ({key}) должен быть положительным числом.")
        except ValueError as e:
            messagebox.showerror("Ошибка ввода (Мишень)", f"Некорректное значение в параметрах мишени: {e}", parent=self)
            raise ValueError(f"Invalid target parameter: {e}") from e
        except Exception as e:
             messagebox.showerror("Ошибка (Мишень)", f"Неожиданная ошибка при чтении параметров мишени: {e}", parent=self)
             raise RuntimeError(f"Unexpected error getting target params: {e}") from e
        return params

    def get_current_target_type(self) -> str:
        return self.combo_tgt.get()

    def get_entry_widget(self, param_name: str) -> ttk.Entry | None:
         if param_name == 'width' and hasattr(self, 'entry_width'): return self.entry_width
         if param_name == 'length' and hasattr(self, 'entry_length'): return self.entry_length # <--- Добавлено
         return None

if __name__ == '__main__':
    root = tk.Tk()
    root.title("Тест TargetFrame")
    def on_target_update(target_type, widget):
        print(f"App notified: Target type/param updated to: {target_type}, triggered by: {widget}")
        if widget: print(f"Widget value: {widget.get() if isinstance(widget, ttk.Entry) else widget.cget('text')}") # type: ignore
        # Пример: доступ к ProcessingFrame (если бы он был здесь)
        # if hasattr(root, 'processing_frame_instance') and root.processing_frame_instance:
        #     root.processing_frame_instance.update_layout(target_type)
        #     if target_type == config.TARGET_LINEAR:
        #         root.processing_frame_instance._calculate_and_update_dependent_linear_param()


    frame = TargetFrame(root, target_update_callback=on_target_update, padding=10)
    frame.pack(expand=True, fill=tk.BOTH)
    def print_params():
        try:
            params = frame.get_params()
            print("Полученные параметры:", params)
        except (ValueError, RuntimeError) as e: print("Ошибка получения параметров:", e)
    button = ttk.Button(root, text="Получить параметры", command=print_params)
    button.pack(pady=10)
    root.mainloop()
