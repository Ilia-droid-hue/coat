# coating_simulator_project/coating_simulator/gui/app.py
"""
Main application class for the Coating Simulator GUI.
Uses a grid layout for parameter frames.
Integrates multiprocessing for simulation.
Progress bar aggregation and text update logic refined.
Corrected call in _handle_target_update.
Updated to import ResultsWindow from the new results_gui subpackage.
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import queue as thread_queue
import multiprocessing
import time
import traceback
from collections import OrderedDict
import numpy as np

# Импорты фреймов параметров
from .frames.target import TargetFrame
from .frames.source import SourceFrame
from .frames.emission import EmissionFrame
from .frames.processing import ProcessingFrame
from .frames.control import ControlFrame

# --- ИЗМЕНЕННЫЙ ИМПОРТ ResultsWindow ---
# Теперь импортируем из подпакета results_gui
from .results_gui.results_window import ResultsWindow
# --- КОНЕЦ ИЗМЕНЕНИЯ ---

from .progress_window import ProgressWindow
from .geometry_view_window import GeometryViewWindow
from ..core import simulation
from ..visualization import export
from .. import config

try:
    import matplotlib
    matplotlib.use('TkAgg') # Установка бэкенда для Tkinter
except ImportError:
    print("Matplotlib не найден. Визуализация будет недоступна.")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Симулятор покрытий v2.9.1 (Рефакторинг Results GUI)") # Обновлена версия для отслеживания
        self.minsize(650, 450) # Минимальный размер окна

        # Менеджер для объектов multiprocessing
        self.mp_manager = multiprocessing.Manager()

        # Атрибуты для управления симуляцией в отдельном процессе
        self.simulation_process_pool = None # Пока не используется, но оставлено для возможного расширения
        self.cancel_event = self.mp_manager.Event() # Событие для отмены симуляции
        self.results_queue = self.mp_manager.Queue() # Очередь для результатов симуляции
        self.progress_updates_queue = self.mp_manager.Queue() # Очередь для обновлений прогресса

        # Хранение последних результатов и инстансов окон
        self.last_results = None
        self.progress_window_instance = None
        self.geometry_window_instance = None
        self.current_simulation_params = None # Параметры текущей запущенной симуляции
        self.simulation_main_thread = None # Поток, который управляет процессом симуляции

        # Для агрегации прогресса от воркеров
        self.last_reported_gui_progress_percent = -1
        self.worker_progress_aggregator = {}

        # Настройка стилей ttk
        style = ttk.Style(self)
        try:
            style.theme_use('clam') # Попытка использовать более современную тему
        except tk.TclError:
            style.theme_use('default') # Фоллбэк на стандартную тему

        # Пользовательский стиль для Progressbar
        self.custom_progress_bar_style_name = "GreenFill.Horizontal.TProgressbar"
        style.configure(self.custom_progress_bar_style_name, background='green', troughcolor='#E0E0E0', borderwidth=1, relief=tk.FLAT)
        # Стиль для Combobox (пример)
        style.map("TCombobox",
                    fieldbackground=[('readonly', 'green'), ('disabled', '#D3D3D3'), ('!readonly', 'white')],
                    foreground=[('readonly', 'white'), ('disabled', '#A0A0A0'), ('!readonly', 'black')],
                    selectbackground=[('!readonly', '#B0E0E6')], # Цвет фона выбранного элемента
                    selectforeground=[('!readonly', 'black')])  # Цвет текста выбранного элемента
        style.configure("TCombobox", arrowcolor='black', arrowsize=12)
        # Стиль для Listbox (пример)
        style.configure("TListbox", background="white", foreground="black", selectbackground="green", selectforeground="white", relief=tk.FLAT, borderwidth=1)


        # Основной фрейм приложения
        main_frame = ttk.Frame(self, padding="5")
        main_frame.pack(expand=True, fill=tk.BOTH)
        # Конфигурация сетки основного фрейма
        main_frame.columnconfigure(0, weight=1) # Левая колонка фреймов параметров
        main_frame.columnconfigure(1, weight=1) # Правая колонка фреймов параметров
        main_frame.rowconfigure(0, weight=1)    # Верхний ряд фреймов
        main_frame.rowconfigure(1, weight=1)    # Нижний ряд фреймов
        main_frame.rowconfigure(2, weight=0)    # Ряд для панели управления (не растягивается)

        # Создание фреймов параметров
        self.target_frame = TargetFrame(main_frame, target_update_callback=self._handle_target_update)
        self.source_frame = SourceFrame(main_frame)
        self.emission_frame = EmissionFrame(main_frame) # processing_frame_update_callback можно добавить, если нужно
        self.processing_frame = ProcessingFrame(main_frame, target_frame_ref=self.target_frame, emission_frame_ref=self.emission_frame)
        self.control_frame = ControlFrame(
            main_frame,
            run_callback=self.start_simulation,
            export_callback=self.export_results,
            show_params_callback=self.show_current_parameters,
            show_geometry_callback=self.show_geometry_preview
        )
        self.progress_var = tk.IntVar() # Переменная для хранения значения прогресс-бара

        # Размещение фреймов на сетке
        frame_pady = (3, 5)
        frame_padx = 5
        sticky_opts = "nsew" # Растягивать во все стороны

        self.target_frame.grid(row=0, column=0, padx=frame_padx, pady=frame_pady, sticky=sticky_opts)
        self.source_frame.grid(row=1, column=0, padx=frame_padx, pady=frame_pady, sticky=sticky_opts)
        self.emission_frame.grid(row=0, column=1, padx=frame_padx, pady=frame_pady, sticky=sticky_opts)
        self.processing_frame.grid(row=1, column=1, padx=frame_padx, pady=frame_pady, sticky=sticky_opts)
        self.control_frame.grid(row=2, column=0, columnspan=2, padx=frame_padx, pady=(10, 5), sticky="ew")

        # Первоначальное обновление макета после инициализации всех виджетов
        self.after(10, self._initial_layout_update)
        # Запуск периодической проверки очередей
        self._check_queues()

    def _initial_layout_update(self):
        """Выполняет первоначальное обновление макета, например, для ProcessingFrame."""
        try:
            if hasattr(self, 'target_frame') and self.target_frame and \
               hasattr(self, 'processing_frame') and self.processing_frame:
                initial_target_type = self.target_frame.get_current_target_type()
                self.processing_frame.update_layout(initial_target_type)
        except Exception as e:
             print(f"Error during initial layout update in App._initial_layout_update: {e}")
             traceback.print_exc()

    def _handle_target_update(self, target_type: str, changed_widget=None):
        """
        Обрабатывает обновление типа мишени или ее релевантных размеров.
        Обновляет макет ProcessingFrame и связанные вычисления.
        """
        # print(f"DEBUG App: _handle_target_update called. Type: {target_type}, Widget: {changed_widget}")
        try:
            if hasattr(self, 'processing_frame') and self.processing_frame:
                self.processing_frame.update_layout(target_type)
                
                if target_type == config.TARGET_LINEAR:
                    if hasattr(self.processing_frame, '_calculate_and_update_dependent_linear_param'):
                        self.processing_frame._calculate_and_update_dependent_linear_param()
                    else:
                        print("ERROR in App._handle_target_update: ProcessingFrame missing _calculate_and_update_dependent_linear_param")
            else:
                 print("Warning: processing_frame not available in _handle_target_update.")
        except AttributeError as e:
             print(f"ERROR in _handle_target_update (AttributeError): {e}")
             traceback.print_exc()
        except Exception as e:
             print(f"ERROR during processing_frame update in _handle_target_update: {e}")
             traceback.print_exc()

    def _gather_parameters(self) -> dict | None:
        """Собирает параметры из всех фреймов конфигурации."""
        try:
            # Проверка наличия всех фреймов
            if not all(hasattr(self, frame_attr) and getattr(self, frame_attr)
                       for frame_attr in ['target_frame', 'source_frame', 'emission_frame', 'processing_frame']):
                messagebox.showerror("Ошибка", "Один или несколько фреймов параметров не инициализированы.", parent=self)
                return None

            params = OrderedDict() # Используем OrderedDict для сохранения порядка добавления
            params.update(self.target_frame.get_params())
            params.update(self.source_frame.get_params())
            params.update(self.emission_frame.get_params())
            params.update(self.processing_frame.get_params())
            return dict(params) # Возвращаем как обычный dict
        except (ValueError, RuntimeError): # Ошибки валидации из get_params() фреймов
            return None # Сообщение об ошибке уже показано во фрейме
        except AttributeError as e: # Если какой-то фрейм не имеет метода get_params
            messagebox.showerror("Ошибка конфигурации", f"Ошибка при доступе к параметрам фрейма: {e}", parent=self)
            traceback.print_exc()
            return None


    def show_current_parameters(self):
        """Отображает окно с текущими параметрами симуляции."""
        params = self._gather_parameters()
        if params:
            # Фильтруем параметры для отображения (только базовые типы)
            params_to_show = {k: v for k, v in params.items() if isinstance(v, (str, int, float, bool))}
            lines = [f"- {key}: {value}" for key, value in params_to_show.items()]
            param_string = "\n".join(lines)

            # Создание окна для отображения параметров
            param_win = tk.Toplevel(self)
            param_win.title("Текущие параметры")
            param_win.geometry("400x450") # Размер окна
            param_win.transient(self) # Делаем окно модальным относительно родительского
            param_win.grab_set()      # Перехватываем фокус

            txt_frame = ttk.Frame(param_win, padding=5)
            txt_frame.pack(expand=True, fill=tk.BOTH)
            txt = scrolledtext.ScrolledText(txt_frame, wrap=tk.WORD, width=50, height=20, relief=tk.FLAT, borderwidth=1)
            txt.insert(tk.INSERT, param_string)
            txt.config(state=tk.DISABLED) # Запрещаем редактирование
            txt.pack(expand=True, fill=tk.BOTH)

            button_frame = ttk.Frame(param_win, padding=(0, 5, 0, 5))
            button_frame.pack(fill=tk.X)
            ttk.Button(button_frame, text="Закрыть", command=param_win.destroy).pack()

            # Центрирование окна параметров
            self.update_idletasks() # Обновляем размеры окна
            win_width = param_win.winfo_reqwidth()
            win_height = param_win.winfo_reqheight()
            parent_x = self.winfo_x()
            parent_y = self.winfo_y()
            parent_width = self.winfo_width()
            parent_height = self.winfo_height()
            center_x = parent_x + (parent_width - win_width) // 2
            center_y = parent_y + (parent_height - win_height) // 2
            param_win.geometry(f'+{center_x}+{center_y}')


    def show_geometry_preview(self):
        """Отображает окно предпросмотра 2D геометрии."""
        if self.geometry_window_instance and self.geometry_window_instance.winfo_exists():
            try:
                self.geometry_window_instance.lift() # Поднять окно наверх
                self.geometry_window_instance.focus_set() # Установить фокус
            except tk.TclError: # Окно могло быть уничтожено неожиданно
                self.geometry_window_instance = None
                self.show_geometry_preview() # Попробовать открыть заново
            return

        params = self._gather_parameters()
        if params is None: # Если параметры не собраны (была ошибка валидации)
            return

        try:
            self.geometry_window_instance = GeometryViewWindow(self, params)
        except Exception as e:
            messagebox.showerror("Ошибка Геометрии", f"Не удалось отобразить геометрию:\n{e}", parent=self)
            traceback.print_exc()
            if self.geometry_window_instance:
                 try:
                     if self.geometry_window_instance.winfo_exists():
                         self.geometry_window_instance.destroy()
                 except tk.TclError: pass # Игнорируем ошибку, если окно уже уничтожено
                 finally: self.geometry_window_instance = None

    def start_simulation(self):
        """Запускает процесс симуляции."""
        if self.simulation_main_thread and self.simulation_main_thread.is_alive():
            messagebox.showwarning("Симуляция", "Симуляция уже выполняется.", parent=self)
            return

        self.current_simulation_params = self._gather_parameters()
        if self.current_simulation_params is None:
            return # Ошибка сбора параметров уже была показана

        self.cancel_event.clear() # Сбрасываем флаг отмены
        self.progress_var.set(0)  # Сбрасываем прогресс-бар
        self.last_reported_gui_progress_percent = -1
        self.worker_progress_aggregator = {} # Очищаем агрегатор прогресса

        self.control_frame.disable_run_button()
        self.control_frame.disable_export_button()
        self.last_results = None # Очищаем предыдущие результаты

        # Закрываем старое окно прогресса, если оно существует
        if self.progress_window_instance and self.progress_window_instance.winfo_exists():
            try: self.progress_window_instance.close_window()
            except tk.TclError: pass
        
        # Создаем новое окно прогресса
        self.progress_window_instance = ProgressWindow(
            self, self.progress_var, self.cancel_event,
            style_name=self.custom_progress_bar_style_name # Применяем кастомный стиль
        )
        self.progress_window_instance.set_progress_text(
            f"Запуск для '{self.current_simulation_params.get('target_type', 'N/A')}'..."
        )
        self.progress_window_instance.lift() # Поднимаем окно прогресса наверх

        # Выбор функции симуляции в зависимости от типа мишени
        target_type = self.current_simulation_params.get('target_type')
        sim_func_mp = None
        if target_type in [config.TARGET_DISK, config.TARGET_DOME]:
            sim_func_mp = simulation.simulate_coating_disk_dome_mp
        elif target_type == config.TARGET_LINEAR:
            sim_func_mp = simulation.simulate_linear_movement_mp
        elif target_type == config.TARGET_PLANETARY:
            sim_func_mp = simulation.simulate_planetary_mp

        if sim_func_mp is None:
            messagebox.showerror("Ошибка", f"Неизвестный тип мишени для MP симуляции: {target_type}", parent=self)
            self.control_frame.enable_run_button()
            if self.progress_window_instance:
                self.progress_window_instance.close_window()
                self.progress_window_instance = None
            return

        # Запуск симуляции в отдельном потоке, который будет управлять пулом процессов
        self.simulation_main_thread = threading.Thread(
            target=self._simulation_runner,
            args=(sim_func_mp, self.current_simulation_params, self.cancel_event,
                  self.results_queue, self.progress_updates_queue),
            daemon=True # Поток завершится при закрытии основного приложения
        )
        self.simulation_main_thread.start()

    def _simulation_runner(self, sim_func_mp, params, cancel_event_mp, results_q_mp, progress_updates_q_mp):
        """
        Обёртка для запуска функции симуляции в отдельном потоке/процессе.
        Обрабатывает результаты и ошибки.
        """
        start_time = time.time()
        results_data = None
        error_info = None
        sim_cancelled_by_event = False

        try:
            # Передаем progress_q и cancel_event в функцию симуляции
            results_data = sim_func_mp(params, progress_q=progress_updates_q_mp, cancel_event=cancel_event_mp)
        except Exception as e:
            error_info = e
            print(f"--- Ошибка в потоке-раннере симуляции (_simulation_runner) ---")
            traceback.print_exc()
            print(f"-------------------------------------------------------------")
        finally:
            if cancel_event_mp.is_set(): # Проверяем, была ли отмена
                sim_cancelled_by_event = True

            end_time = time.time()
            duration = end_time - start_time
            sim_status = "отменена" if sim_cancelled_by_event else \
                         ("завершена с ошибкой" if error_info else "успешно завершена")
            print(f"Симуляция (MP Runner) {sim_status} за {duration:.2f} секунд.")

            # Помещаем результат в очередь для основного потока GUI
            results_q_mp.put({
                'results': results_data if not error_info and not sim_cancelled_by_event else None,
                'error': error_info,
                'params': params, # Возвращаем параметры, с которыми была запущена симуляция
                'cancelled': sim_cancelled_by_event,
                'duration': duration
            })

    def _check_queues(self):
        """Периодически проверяет очереди на наличие обновлений прогресса или результатов."""
        total_particles_overall = self.current_simulation_params.get('particles', 0) if self.current_simulation_params else 0
        new_overall_progress_percent = -1

        # Обработка обновлений прогресса
        try:
            while True: # Вычитываем все сообщения из очереди прогресса
                worker_id, processed_in_worker, total_in_worker = self.progress_updates_queue.get_nowait()
                self.worker_progress_aggregator[worker_id] = (processed_in_worker, total_in_worker)
        except thread_queue.Empty:
            pass # Очередь пуста, это нормально
        except Exception as e:
            # Игнорируем специфичные ошибки, которые могут возникать при закрытии
            if not isinstance(e, (AttributeError, EOFError, BrokenPipeError)):
                print(f"Error processing progress_updates_queue: {e}")

        # Агрегация и обновление GUI прогресс-бара
        if self.worker_progress_aggregator and total_particles_overall > 0:
            total_processed_overall = sum(p[0] for p in self.worker_progress_aggregator.values())
            current_calc_progress_percent = int((total_processed_overall / total_particles_overall) * 100)
            new_overall_progress_percent = max(0, min(100, current_calc_progress_percent))

            self.progress_var.set(new_overall_progress_percent)

            if self.progress_window_instance and self.progress_window_instance.winfo_exists():
                # Обновляем текст в окне прогресса не слишком часто, чтобы не перегружать GUI
                if (self.last_reported_gui_progress_percent == -1 or
                    new_overall_progress_percent == 100 or
                    new_overall_progress_percent >= self.last_reported_gui_progress_percent + 5 or # Обновление каждые 5%
                    new_overall_progress_percent < self.last_reported_gui_progress_percent): # Или если прогресс уменьшился (сброс)
                    new_text = f"Выполнение: {new_overall_progress_percent}%"
                    self.progress_window_instance.set_progress_text(new_text)
                    self.last_reported_gui_progress_percent = new_overall_progress_percent
        elif not self.current_simulation_params and self.progress_var.get() != 0 : # Если симуляция не запущена, сбрасываем прогресс
            self.progress_var.set(0)
            if self.progress_window_instance and self.progress_window_instance.winfo_exists():
                self.progress_window_instance.set_progress_text("Ожидание запуска...")
            self.last_reported_gui_progress_percent = 0


        # Обработка результатов симуляции
        try:
            result_data_from_queue = self.results_queue.get_nowait()
            self._handle_simulation_completion(result_data_from_queue)
        except thread_queue.Empty:
            pass # Очередь пуста
        except Exception as e:
             if not isinstance(e, (AttributeError, EOFError, BrokenPipeError)):
                 print(f"Error processing results_queue: {e}")

        # Повторный вызов через 100 мс
        self.after(100, self._check_queues)

    def _handle_simulation_completion(self, result_data):
        """Обрабатывает завершение симуляции (успешное, с ошибкой или отмененное)."""
        # Закрываем окно прогресса
        if self.progress_window_instance and self.progress_window_instance.winfo_exists():
            self.progress_window_instance.close_window()
        self.progress_window_instance = None
        self.simulation_main_thread = None # Сбрасываем ссылку на поток

        # Устанавливаем прогресс-бар в 100% или 0%
        if not result_data.get('cancelled', False) and result_data.get('error') is None:
            self.progress_var.set(100)
            self.last_reported_gui_progress_percent = 100
        else:
            self.progress_var.set(0)
            self.last_reported_gui_progress_percent = 0

        self.control_frame.enable_run_button() # Включаем кнопку "Запустить"

        error = result_data.get('error')
        results_tuple = result_data.get('results')
        params_completed_with = result_data.get('params')
        cancelled_by_user = result_data.get('cancelled', False)
        duration_seconds = result_data.get('duration', 0)

        self._last_completed_sim_params = params_completed_with # Сохраняем параметры для экспорта

        if cancelled_by_user:
            messagebox.showinfo("Симуляция", f"Симуляция была отменена пользователем.\nВремя выполнения: {duration_seconds:.2f} с.", parent=self)
        elif error:
            messagebox.showerror("Ошибка Симуляции", f"Во время симуляции произошла ошибка:\n{error}\nВремя выполнения: {duration_seconds:.2f} с.", parent=self)
        elif results_tuple is not None:
            # Проверяем корректность формата результатов
            if isinstance(results_tuple, tuple) and len(results_tuple) == 4 and \
               all(isinstance(arr, np.ndarray) for arr in results_tuple):
                self.last_results = results_tuple # Сохраняем результаты
                self.control_frame.enable_export_button() # Включаем кнопку экспорта

                msg_done = f"Симуляция успешно завершена за {duration_seconds:.2f} с."
                # Проверяем, есть ли покрытие
                if np.sum(results_tuple[0]) == 0 : # results_tuple[0] - это coverage_map
                    msg_done += "\n\nВнимание: Карта покрытия пуста (0 частиц попало на мишень)."
                    # Используем type=messagebox.YESNO для showwarning
                    if messagebox.showwarning("Симуляция Завершена", msg_done + "\nПоказать результаты?", parent=self, type=messagebox.YESNO) == messagebox.YES: # type: ignore
                        vis_params_for_results = {'percent': True, 'logscale': False, 'show3d': False} # Параметры по умолчанию для окна результатов
                        try:
                            ResultsWindow(self, *results_tuple, target_params=params_completed_with, vis_params=vis_params_for_results)
                        except Exception as e_res_win:
                            messagebox.showerror("Ошибка Отображения", f"Не удалось открыть окно результатов:\n{e_res_win}", parent=self)
                            traceback.print_exc()
                elif messagebox.askyesno("Симуляция Завершена", msg_done + "\nПоказать результаты?", parent=self):
                    vis_params_for_results = {'percent': True, 'logscale': False, 'show3d': False}
                    try:
                        ResultsWindow(self, *results_tuple, target_params=params_completed_with, vis_params=vis_params_for_results)
                    except Exception as e_res_win:
                        messagebox.showerror("Ошибка Отображения", f"Не удалось открыть окно результатов:\n{e_res_win}", parent=self)
                        traceback.print_exc()
            else:
                messagebox.showwarning("Симуляция", f"Симуляция завершилась, но вернула некорректные результаты.\nВремя выполнения: {duration_seconds:.2f} с.", parent=self)
                self.last_results = None
                self.control_frame.disable_export_button()
        else: # results_tuple is None, но не было отмены или явной ошибки
            if not cancelled_by_user and not error : # Дополнительная проверка
                 messagebox.showwarning("Симуляция", f"Симуляция завершилась, но не вернула результатов (возможно, отменена внутри пула).\nВремя выполнения: {duration_seconds:.2f} с.", parent=self)
            self.last_results = None
            self.control_frame.disable_export_button()

        self.current_simulation_params = None # Сбрасываем параметры текущей симуляции
        # Сброс прогресс-бара через некоторое время, если симуляция не была отменена или не было ошибки
        if not cancelled_by_user and not error:
            self.after(1500, lambda: [self.progress_var.set(0), setattr(self, 'last_reported_gui_progress_percent', -1)])
        else: # Немедленный сброс, если была отмена или ошибка
            self.progress_var.set(0)
            self.last_reported_gui_progress_percent = -1


    def export_results(self):
        """Экспортирует последние результаты симуляции в CSV."""
        if self.last_results is None:
            messagebox.showwarning("Экспорт", "Нет результатов для экспорта. Запустите симуляцию.", parent=self)
            return

        params_for_export = None
        if hasattr(self, '_last_completed_sim_params') and self._last_completed_sim_params:
            params_for_export = self._last_completed_sim_params
        else: # Фоллбэк, если _last_completed_sim_params не установлены
            params_for_export = self._gather_parameters()

        if not params_for_export:
            messagebox.showerror("Ошибка Экспорта", "Не удалось получить параметры для экспорта.", parent=self)
            return

        vis_params_export = {'percent': True} # Параметры для экспорта (например, экспортировать в %)
        try:
            export.export_csv(*self.last_results, target_params=params_for_export, vis_params=vis_params_export)
        except Exception as e:
            messagebox.showerror("Ошибка Экспорта", f"Не удалось экспортировать данные:\n{e}", parent=self)
            traceback.print_exc()

    def on_closing(self):
        """Обработчик закрытия главного окна приложения."""
        if self.simulation_main_thread and self.simulation_main_thread.is_alive():
            if messagebox.askyesno("Выход", "Симуляция еще выполняется. Прервать и выйти?", parent=self):
                 self.cancel_event.set() # Сигнализируем об отмене
                 self.simulation_main_thread.join(timeout=1.0) # Ждем завершения потока

                 # Закрываем дочерние окна
                 if self.progress_window_instance and self.progress_window_instance.winfo_exists():
                     try: self.progress_window_instance.destroy()
                     except tk.TclError: pass
                 if self.geometry_window_instance and self.geometry_window_instance.winfo_exists():
                     try: self.geometry_window_instance.destroy()
                     except tk.TclError: pass
                 self.destroy() # Закрываем главное окно
            else:
                return # Пользователь отменил выход
        else:
            # Закрываем дочерние окна, если они еще открыты
            if self.progress_window_instance and self.progress_window_instance.winfo_exists():
                try: self.progress_window_instance.destroy()
                except tk.TclError: pass
            if self.geometry_window_instance and self.geometry_window_instance.winfo_exists():
                try: self.geometry_window_instance.destroy()
                except tk.TclError: pass
            self.destroy()

if __name__ == '__main__':
    multiprocessing.freeze_support() # Необходимо для Windows при использовании multiprocessing
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_closing) # Обработчик закрытия окна
    app.mainloop()
