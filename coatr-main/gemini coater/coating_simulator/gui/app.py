# coating_simulator_project/coating_simulator/gui/app.py
"""
Main application class for the Coating Simulator GUI.
Uses a grid layout for parameter frames.
Integrates multiprocessing for simulation.
Progress bar aggregation and text update logic refined.
Corrected call in _handle_target_update.
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

from .frames.target import TargetFrame
from .frames.source import SourceFrame
from .frames.emission import EmissionFrame
from .frames.processing import ProcessingFrame
from .frames.control import ControlFrame
from .results import ResultsWindow
from .progress_window import ProgressWindow
from .geometry_view_window import GeometryViewWindow
from ..core import simulation
from ..visualization import export
from .. import config

try:
    import matplotlib
    matplotlib.use('TkAgg')
except ImportError:
    print("Matplotlib не найден. Визуализация будет недоступна.")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Симулятор покрытий v2.9 (Фикс TargetFrame callback)") 
        self.minsize(650, 450)

        self.mp_manager = multiprocessing.Manager()

        self.simulation_process_pool = None
        self.cancel_event = self.mp_manager.Event()
        self.results_queue = self.mp_manager.Queue()
        self.progress_updates_queue = self.mp_manager.Queue()

        self.last_results = None
        self.progress_window_instance = None
        self.geometry_window_instance = None
        self.current_simulation_params = None
        self.simulation_main_thread = None
        
        self.last_reported_gui_progress_percent = -1
        self.worker_progress_aggregator = {} 

        style = ttk.Style(self)
        try: style.theme_use('clam')
        except tk.TclError: style.theme_use('default')
        self.custom_progress_bar_style_name = "GreenFill.Horizontal.TProgressbar"
        style.configure(self.custom_progress_bar_style_name, background='green', troughcolor='#E0E0E0', borderwidth=1, relief=tk.FLAT)
        style.map("TCombobox", fieldbackground=[('readonly', 'green'), ('disabled', '#D3D3D3'), ('!readonly', 'white')], foreground=[('readonly', 'white'), ('disabled', '#A0A0A0'), ('!readonly', 'black')], selectbackground=[('!readonly', '#B0E0E6')], selectforeground=[('!readonly', 'black')])
        style.configure("TCombobox", arrowcolor='black', arrowsize=12)
        style.configure("TListbox", background="white", foreground="black", selectbackground="green", selectforeground="white", relief=tk.FLAT, borderwidth=1)

        main_frame = ttk.Frame(self, padding="5")
        main_frame.pack(expand=True, fill=tk.BOTH)
        main_frame.columnconfigure(0, weight=1); main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1); main_frame.rowconfigure(1, weight=1); main_frame.rowconfigure(2, weight=0)

        self.target_frame = TargetFrame(main_frame, target_update_callback=self._handle_target_update)
        self.source_frame = SourceFrame(main_frame)
        self.emission_frame = EmissionFrame(main_frame)
        self.processing_frame = ProcessingFrame(main_frame, target_frame_ref=self.target_frame, emission_frame_ref=self.emission_frame)
        self.control_frame = ControlFrame(main_frame, run_callback=self.start_simulation, export_callback=self.export_results, show_params_callback=self.show_current_parameters, show_geometry_callback=self.show_geometry_preview)
        self.progress_var = tk.IntVar()

        frame_pady = (3, 5); frame_padx = 5; sticky_opts = "nsew"
        self.target_frame.grid(row=0, column=0, padx=frame_padx, pady=frame_pady, sticky=sticky_opts)
        self.source_frame.grid(row=1, column=0, padx=frame_padx, pady=frame_pady, sticky=sticky_opts)
        self.emission_frame.grid(row=0, column=1, padx=frame_padx, pady=frame_pady, sticky=sticky_opts)
        self.processing_frame.grid(row=1, column=1, padx=frame_padx, pady=frame_pady, sticky=sticky_opts)
        self.control_frame.grid(row=2, column=0, columnspan=2, padx=frame_padx, pady=(10, 5), sticky="ew")

        self.after(10, self._initial_layout_update)
        self._check_queues()

    def _initial_layout_update(self):
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
        Safely updates the processing frame layout and dependent calculations
        when target type or relevant dimensions change.
        """
        # print(f"DEBUG App: _handle_target_update called. Type: {target_type}, Widget: {changed_widget}")
        try:
            if hasattr(self, 'processing_frame') and self.processing_frame:
                self.processing_frame.update_layout(target_type)
                
                if target_type == config.TARGET_LINEAR:
                    # Этот метод должен быть доступен в ProcessingFrame
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
        try:
            if not all(hasattr(self, frame_attr) and getattr(self, frame_attr)
                       for frame_attr in ['target_frame', 'source_frame', 'emission_frame', 'processing_frame']):
                messagebox.showerror("Ошибка", "Один или несколько фреймов параметров не инициализированы.", parent=self)
                return None
            params = OrderedDict()
            params.update(self.target_frame.get_params())
            params.update(self.source_frame.get_params())
            params.update(self.emission_frame.get_params())
            params.update(self.processing_frame.get_params())
            return dict(params)
        except (ValueError, RuntimeError): return None
        except AttributeError: return None

    def show_current_parameters(self):
        params = self._gather_parameters()
        if params:
            params_to_show = {k: v for k, v in params.items() if isinstance(v, (str, int, float, bool))}
            lines = [f"- {key}: {value}" for key, value in params_to_show.items()]
            param_string = "\n".join(lines)
            param_win = tk.Toplevel(self); param_win.title("Текущие параметры"); param_win.geometry("400x450")
            param_win.transient(self); param_win.grab_set()
            txt_frame = ttk.Frame(param_win, padding=5); txt_frame.pack(expand=True, fill=tk.BOTH)
            txt = scrolledtext.ScrolledText(txt_frame, wrap=tk.WORD, width=50, height=20, relief=tk.FLAT, borderwidth=1)
            txt.insert(tk.INSERT, param_string); txt.config(state=tk.DISABLED); txt.pack(expand=True, fill=tk.BOTH)
            button_frame = ttk.Frame(param_win, padding=(0, 5, 0, 5)); button_frame.pack(fill=tk.X)
            ttk.Button(button_frame, text="Закрыть", command=param_win.destroy).pack()
            self.update_idletasks()
            win_width = param_win.winfo_reqwidth(); win_height = param_win.winfo_reqheight()
            parent_x = self.winfo_x(); parent_y = self.winfo_y()
            parent_width = self.winfo_width(); parent_height = self.winfo_height()
            center_x = parent_x + (parent_width - win_width) // 2
            center_y = parent_y + (parent_height - win_height) // 2
            param_win.geometry(f'+{center_x}+{center_y}')

    def show_geometry_preview(self):
        if self.geometry_window_instance and self.geometry_window_instance.winfo_exists():
            try: self.geometry_window_instance.lift(); self.geometry_window_instance.focus_set()
            except tk.TclError: self.geometry_window_instance = None; self.show_geometry_preview()
            return
        params = self._gather_parameters()
        if params is None: return
        try:
            self.geometry_window_instance = GeometryViewWindow(self, params)
        except Exception as e:
            messagebox.showerror("Ошибка Геометрии", f"Не удалось отобразить геометрию:\n{e}", parent=self)
            if self.geometry_window_instance:
                 try:
                     if self.geometry_window_instance.winfo_exists(): self.geometry_window_instance.destroy()
                 except tk.TclError: pass
                 finally: self.geometry_window_instance = None
    
    def start_simulation(self):
        if self.simulation_main_thread and self.simulation_main_thread.is_alive():
            messagebox.showwarning("Симуляция", "Симуляция уже выполняется.", parent=self)
            return
        self.current_simulation_params = self._gather_parameters()
        if self.current_simulation_params is None: return
        self.cancel_event.clear(); self.progress_var.set(0)
        self.last_reported_gui_progress_percent = -1
        self.worker_progress_aggregator = {}
        self.control_frame.disable_run_button(); self.control_frame.disable_export_button()
        self.last_results = None
        if self.progress_window_instance and self.progress_window_instance.winfo_exists():
            try: self.progress_window_instance.close_window()
            except tk.TclError: pass
        self.progress_window_instance = ProgressWindow(self, self.progress_var, self.cancel_event, style_name=self.custom_progress_bar_style_name)
        self.progress_window_instance.set_progress_text(f"Запуск для '{self.current_simulation_params.get('target_type', 'N/A')}'...")
        self.progress_window_instance.lift()
        target_type = self.current_simulation_params.get('target_type')
        sim_func_mp = None
        if target_type in [config.TARGET_DISK, config.TARGET_DOME]: sim_func_mp = simulation.simulate_coating_disk_dome_mp
        elif target_type == config.TARGET_LINEAR: sim_func_mp = simulation.simulate_linear_movement_mp
        elif target_type == config.TARGET_PLANETARY: sim_func_mp = simulation.simulate_planetary_mp
        if sim_func_mp is None:
            messagebox.showerror("Ошибка", f"Неизвестный тип мишени для MP симуляции: {target_type}", parent=self)
            self.control_frame.enable_run_button()
            if self.progress_window_instance: self.progress_window_instance.close_window(); self.progress_window_instance = None
            return
        self.simulation_main_thread = threading.Thread(target=self._simulation_runner, args=(sim_func_mp, self.current_simulation_params, self.cancel_event, self.results_queue, self.progress_updates_queue), daemon=True)
        self.simulation_main_thread.start()

    def _simulation_runner(self, sim_func_mp, params, cancel_event_mp, results_q_mp, progress_updates_q_mp):
        start_time = time.time(); results_data = None; error_info = None; sim_cancelled = False
        try: results_data = sim_func_mp(params, progress_q=progress_updates_q_mp, cancel_event=cancel_event_mp)
        except Exception as e: error_info = e; print(f"--- Ошибка в потоке-раннере симуляции (_simulation_runner) ---"); traceback.print_exc(); print(f"-------------------------------------------------------------")
        finally:
            if cancel_event_mp.is_set(): sim_cancelled = True
            end_time = time.time(); duration = end_time - start_time
            sim_status = "отменена" if sim_cancelled else ("завершена с ошибкой" if error_info else "успешно завершена")
            print(f"Симуляция (MP Runner) {sim_status} за {duration:.2f} секунд.")
            results_q_mp.put({'results': results_data if not error_info and not sim_cancelled else None, 'error': error_info, 'params': params, 'cancelled': sim_cancelled, 'duration': duration})

    def _check_queues(self):
        total_particles_overall = self.current_simulation_params.get('particles', 0) if self.current_simulation_params else 0
        new_overall_progress_percent = -1
        try:
            while True:
                worker_id, processed_in_worker, total_in_worker = self.progress_updates_queue.get_nowait()
                self.worker_progress_aggregator[worker_id] = (processed_in_worker, total_in_worker)
        except thread_queue.Empty: pass
        except Exception as e:
            if not isinstance(e, (AttributeError, EOFError, BrokenPipeError)): print(f"Error processing progress_updates_queue: {e}")
        if self.worker_progress_aggregator and total_particles_overall > 0:
            total_processed_overall = sum(p[0] for p in self.worker_progress_aggregator.values())
            current_calc_progress_percent = int((total_processed_overall / total_particles_overall) * 100)
            new_overall_progress_percent = max(0, min(100, current_calc_progress_percent))
            self.progress_var.set(new_overall_progress_percent)
            if self.progress_window_instance and self.progress_window_instance.winfo_exists():
                if (self.last_reported_gui_progress_percent == -1 or new_overall_progress_percent == 100 or new_overall_progress_percent >= self.last_reported_gui_progress_percent + 5):
                    new_text = f"Выполнение: {new_overall_progress_percent}%"
                    self.progress_window_instance.set_progress_text(new_text)
                    self.last_reported_gui_progress_percent = new_overall_progress_percent
                elif new_overall_progress_percent < self.last_reported_gui_progress_percent :
                    new_text = f"Выполнение: {new_overall_progress_percent}%"
                    self.progress_window_instance.set_progress_text(new_text)
                    self.last_reported_gui_progress_percent = new_overall_progress_percent
        elif not self.current_simulation_params and self.progress_var.get() != 0 :
            self.progress_var.set(0)
            if self.progress_window_instance and self.progress_window_instance.winfo_exists(): self.progress_window_instance.set_progress_text("Ожидание запуска...")
            self.last_reported_gui_progress_percent = 0
        try:
            result_data = self.results_queue.get_nowait()
            self._handle_simulation_completion(result_data)
        except thread_queue.Empty: pass
        except Exception as e:
             if not isinstance(e, (AttributeError, EOFError, BrokenPipeError)): print(f"Error processing results_queue: {e}")
        self.after(100, self._check_queues)

    def _handle_simulation_completion(self, result_data):
        if self.progress_window_instance and self.progress_window_instance.winfo_exists(): self.progress_window_instance.close_window()
        self.progress_window_instance = None
        self.simulation_main_thread = None
        if not result_data.get('cancelled', False) and result_data.get('error') is None:
            self.progress_var.set(100); self.last_reported_gui_progress_percent = 100
        else: self.progress_var.set(0); self.last_reported_gui_progress_percent = 0
        self.control_frame.enable_run_button()
        error = result_data.get('error'); results = result_data.get('results')
        params_completed = result_data.get('params'); cancelled = result_data.get('cancelled', False)
        duration = result_data.get('duration', 0)
        self._last_completed_sim_params = params_completed
        if cancelled: messagebox.showinfo("Симуляция", f"Симуляция была отменена пользователем.\nВремя выполнения: {duration:.2f} с.", parent=self)
        elif error: messagebox.showerror("Ошибка Симуляции", f"Во время симуляции произошла ошибка:\n{error}\nВремя выполнения: {duration:.2f} с.", parent=self)
        elif results is not None:
            if isinstance(results, tuple) and len(results) == 4 and all(isinstance(arr, np.ndarray) for arr in results):
                self.last_results = results; self.control_frame.enable_export_button()
                msg_done = f"Симуляция успешно завершена за {duration:.2f} с."
                if np.sum(results[0]) == 0 :
                    msg_done += "\n\nВнимание: Карта покрытия пуста (0 частиц попало на мишень)."
                    if messagebox.showwarning("Симуляция Завершена", msg_done + "\nПоказать результаты?", parent=self, type=messagebox.YESNO) == messagebox.YES: # type: ignore
                        vis_params = {'percent': True, 'logscale': False, 'show3d': False}
                        try: ResultsWindow(self, *results, target_params=params_completed, vis_params=vis_params)
                        except Exception as e: messagebox.showerror("Ошибка Отображения", f"Не удалось открыть окно результатов:\n{e}", parent=self)
                elif messagebox.askyesno("Симуляция Завершена", msg_done + "\nПоказать результаты?", parent=self):
                    vis_params = {'percent': True, 'logscale': False, 'show3d': False}
                    try: ResultsWindow(self, *results, target_params=params_completed, vis_params=vis_params)
                    except Exception as e: messagebox.showerror("Ошибка Отображения", f"Не удалось открыть окно результатов:\n{e}", parent=self)
            else: messagebox.showwarning("Симуляция", f"Симуляция завершилась, но вернула некорректные результаты.\nВремя выполнения: {duration:.2f} с.", parent=self); self.last_results = None; self.control_frame.disable_export_button()
        else:
            if not cancelled and not error : messagebox.showwarning("Симуляция", f"Симуляция завершилась, но не вернула результатов (возможно, отменена внутри пула).\nВремя выполнения: {duration:.2f} с.", parent=self)
            self.last_results = None; self.control_frame.disable_export_button()
        self.current_simulation_params = None
        if not cancelled and not error: self.after(1500, lambda: [self.progress_var.set(0), setattr(self, 'last_reported_gui_progress_percent', -1)])
        else: self.progress_var.set(0); self.last_reported_gui_progress_percent = -1

    def export_results(self):
        if self.last_results is None: messagebox.showwarning("Экспорт", "Нет результатов для экспорта. Запустите симуляцию.", parent=self); return
        params_for_export = None
        if hasattr(self, '_last_completed_sim_params') and self._last_completed_sim_params: params_for_export = self._last_completed_sim_params
        else: params_for_export = self._gather_parameters()
        if not params_for_export: messagebox.showerror("Ошибка Экспорта", "Не удалось получить параметры для экспорта.", parent=self); return
        vis_params_export = {'percent': True}
        try: export.export_csv(*self.last_results, target_params=params_for_export, vis_params=vis_params_export)
        except Exception as e: messagebox.showerror("Ошибка Экспорта", f"Не удалось экспортировать данные:\n{e}", parent=self)

    def on_closing(self):
        if self.simulation_main_thread and self.simulation_main_thread.is_alive():
            if messagebox.askyesno("Выход", "Симуляция еще выполняется. Прервать и выйти?", parent=self):
                 self.cancel_event.set(); self.simulation_main_thread.join(timeout=1.0)
                 if self.progress_window_instance and self.progress_window_instance.winfo_exists():
                     try: self.progress_window_instance.destroy()
                     except tk.TclError: pass
                 if self.geometry_window_instance and self.geometry_window_instance.winfo_exists():
                     try: self.geometry_window_instance.destroy()
                     except tk.TclError: pass
                 self.destroy()
            else: return
        else:
            if self.progress_window_instance and self.progress_window_instance.winfo_exists():
                try: self.progress_window_instance.destroy()
                except tk.TclError: pass
            if self.geometry_window_instance and self.geometry_window_instance.winfo_exists():
                try: self.geometry_window_instance.destroy()
                except tk.TclError: pass
            self.destroy()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
