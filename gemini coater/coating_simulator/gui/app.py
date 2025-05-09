# coating_simulator_project/coating_simulator/gui/app.py
"""
Main application class for the Coating Simulator GUI.
Uses a grid layout for parameter frames.
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import queue
import time
import traceback
from collections import OrderedDict
import numpy as np 

# Используем относительные импорты
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
    """
    Main application window class.
    Класс главного окна приложения.
    """
    def __init__(self):
        super().__init__()
        self.title("Симулятор покрытий v2.6") 
        self.minsize(650, 450) 

        # --- Simulation State ---
        self.simulation_thread = None
        self.cancel_event = threading.Event()
        self.results_queue = queue.Queue()
        self.progress_queue = queue.Queue()
        self.last_results = None
        self.progress_window_instance = None
        self.geometry_window_instance = None

        # --- Style ---
        style = ttk.Style(self)
        # Try different themes if 'clam' causes issues, e.g., 'default', 'alt', 'vista'
        try:
             style.theme_use('clam') 
        except tk.TclError:
             print("Warning: 'clam' theme not available, using default.")
             style.theme_use('default')


        self.custom_progress_bar_style_name = "GreenFill.Horizontal.TProgressbar"
        style.configure(self.custom_progress_bar_style_name,
                        background='green', troughcolor='#E0E0E0', borderwidth=1, relief=tk.FLAT)
        
        # --- ВОССТАНОВЛЕНИЕ СТИЛЯ COMBOBOX ---
        # Configure Combobox style for readonly state specifically
        style.map("TCombobox",
                  # fieldbackground: Background of the entry part
                  fieldbackground=[('readonly', 'green'), ('disabled', '#D3D3D3'), ('!readonly', 'white')],
                  # foreground: Text color
                  foreground=[('readonly', 'white'), ('disabled', '#A0A0A0'), ('!readonly', 'black')],
                  # background: Background of the dropdown button (might not always work depending on theme/OS)
                  # background=[('readonly', 'green'), ('disabled', '#D3D3D3'), ('!readonly', 'white')],
                  # selectbackground/selectforeground: Colors for the dropdown list items
                  selectbackground=[('!readonly', '#B0E0E6')], # Light blue selection in dropdown
                  selectforeground=[('!readonly', 'black')]  # Black text for selection in dropdown
                  )
        # Configure arrow color (might be theme-dependent)
        # style.configure("TCombobox", arrowcolor='white', arrowsize=12) # White arrow might not show well on green
        style.configure("TCombobox", arrowcolor='black', arrowsize=12) # Try black arrow

        # Configure Listbox style (used by Combobox dropdown)
        style.configure("TListbox", background="white", foreground="black", # White background for dropdown list
                        selectbackground="green", selectforeground="white", # Green selection in dropdown
                        relief=tk.FLAT, borderwidth=1) 
        # --- КОНЕЦ ВОССТАНОВЛЕНИЯ СТИЛЯ ---


        # --- Main Frame ---
        main_frame = ttk.Frame(self, padding="5")
        main_frame.pack(expand=True, fill=tk.BOTH) 

        # --- Grid Setup for main_frame ---
        main_frame.columnconfigure(0, weight=1) 
        main_frame.columnconfigure(1, weight=1) 
        main_frame.rowconfigure(0, weight=1) 
        main_frame.rowconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=0) 

        # --- Create Child Frames ---
        self.target_frame = TargetFrame(main_frame) 
        self.source_frame = SourceFrame(main_frame)
        self.emission_frame = EmissionFrame(main_frame) 
        self.processing_frame = ProcessingFrame(main_frame,
                                                target_frame_ref=self.target_frame,
                                                emission_frame_ref=self.emission_frame)
        self.control_frame = ControlFrame(main_frame,
                                          run_callback=self.start_simulation,
                                          export_callback=self.export_results,
                                          show_params_callback=self.show_current_parameters,
                                          show_geometry_callback=self.show_geometry_preview)
        
        # --- Set Callbacks AFTER All Frames Exist ---
        self.target_frame._target_update_callback = self._handle_target_update
        
        self.progress_var = tk.IntVar()

        # --- Layout Frames using grid ---
        frame_pady = (3, 5) 
        frame_padx = 5    
        sticky_opts = "nsew" 

        self.target_frame.grid(row=0, column=0, padx=frame_padx, pady=frame_pady, sticky=sticky_opts)
        self.source_frame.grid(row=1, column=0, padx=frame_padx, pady=frame_pady, sticky=sticky_opts)
        self.emission_frame.grid(row=0, column=1, padx=frame_padx, pady=frame_pady, sticky=sticky_opts)
        self.processing_frame.grid(row=1, column=1, padx=frame_padx, pady=frame_pady, sticky=sticky_opts)
        self.control_frame.grid(row=2, column=0, columnspan=2, padx=frame_padx, pady=(10, 5), sticky="ew") 
        
        # --- Trigger Initial Layout Update Safely at the END of __init__ ---
        self.after(10, self._initial_layout_update)


        # --- Start GUI Update Loop ---
        self._check_queues()

    def _initial_layout_update(self):
        """ Performs the initial layout update after the main window is stable. """
        try:
            if hasattr(self, 'target_frame') and self.target_frame and \
               hasattr(self, 'processing_frame') and self.processing_frame:
                initial_target_type = self.target_frame.get_current_target_type()
                self.processing_frame.update_layout(initial_target_type)
                if initial_target_type == config.TARGET_LINEAR:
                     self.processing_frame._update_passes_if_linear()
            else:
                 print("Warning: Frames not ready during _initial_layout_update.")
        except Exception as e:
             print(f"Error during initial layout update in App._initial_layout_update: {e}")
             traceback.print_exc()

    # ... (rest of the methods in app.py remain the same as in app_py_fix_v3) ...
    def _handle_target_update(self, target_type: str, changed_widget=None):
        """ Safely updates the processing frame layout when target type changes. """
        try:
            if hasattr(self, 'processing_frame') and self.processing_frame:
                self.processing_frame.update_layout(target_type)
                if hasattr(self, 'target_frame') and self.target_frame:
                    width_entry = self.target_frame.get_entry_widget('width')
                    if changed_widget is not None and changed_widget == width_entry:
                        self.processing_frame._update_passes_if_linear() 
            else:
                 print("Warning: processing_frame not available in _handle_target_update.")
        except AttributeError as e:
             print(f"ERROR in _handle_target_update: Could not access frame attribute. {e}")
             traceback.print_exc()
        except Exception as e:
             print(f"ERROR during processing_frame update: {e}")
             traceback.print_exc()

    def _gather_parameters(self) -> dict | None:
        """ Gathers parameters from all frames. """
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
        except (ValueError, RuntimeError) as e:
            print(f"Ошибка сбора параметров в App: {e}") 
            return None
        except AttributeError as ae:
             messagebox.showerror("Ошибка Атрибута", f"Ошибка доступа к фрейму при сборе параметров: {ae}", parent=self)
             traceback.print_exc()
             return None

    def show_current_parameters(self):
        """ Displays the currently configured parameters. """
        params = self._gather_parameters()
        if params:
            params_to_show = {k: v for k, v in params.items() if isinstance(v, (str, int, float, bool))}
            lines = [f"- {key}: {value}" for key, value in params_to_show.items()]
            param_string = "\n".join(lines)
            
            param_win = tk.Toplevel(self)
            param_win.title("Текущие параметры")
            param_win.geometry("400x450") 
            param_win.transient(self) 
            param_win.grab_set() 

            txt_frame = ttk.Frame(param_win, padding=5)
            txt_frame.pack(expand=True, fill=tk.BOTH)

            txt = scrolledtext.ScrolledText(txt_frame, wrap=tk.WORD, width=50, height=20, relief=tk.FLAT, borderwidth=1) 
            txt.insert(tk.INSERT, param_string)
            txt.config(state=tk.DISABLED) 
            txt.pack(expand=True, fill=tk.BOTH)
            
            button_frame = ttk.Frame(param_win, padding=(0, 5, 0, 5))
            button_frame.pack(fill=tk.X)
            close_button = ttk.Button(button_frame, text="Закрыть", command=param_win.destroy)
            close_button.pack() 

            self.update_idletasks() 
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
        """ Shows the 2D geometry preview window. """
        if self.geometry_window_instance and self.geometry_window_instance.winfo_exists():
            print("Окно геометрии уже открыто.")
            try:
                self.geometry_window_instance.lift()
                self.geometry_window_instance.focus_set()
            except tk.TclError: 
                 self.geometry_window_instance = None
                 self.show_geometry_preview() 
            return

        params = self._gather_parameters()
        if params is None: return

        print("Отображение окна геометрии...")
        try:
            self.geometry_window_instance = GeometryViewWindow(self, params)
        except Exception as e:
            messagebox.showerror("Ошибка Геометрии", f"Не удалось отобразить геометрию:\n{e}", parent=self)
            traceback.print_exc()
            if self.geometry_window_instance:
                 try:
                     if self.geometry_window_instance.winfo_exists():
                         self.geometry_window_instance.destroy()
                 except tk.TclError: pass
                 finally: self.geometry_window_instance = None

    def start_simulation(self):
        """ Starts the simulation in a separate thread. """
        if self.simulation_thread and self.simulation_thread.is_alive():
            messagebox.showwarning("Симуляция", "Симуляция уже запущена.", parent=self)
            return

        params = self._gather_parameters()
        if params is None: return

        self.cancel_event.clear()
        self.progress_var.set(0)
        self.control_frame.disable_run_button()
        self.control_frame.disable_export_button()
        self.last_results = None 

        if self.progress_window_instance and self.progress_window_instance.winfo_exists():
            try: self.progress_window_instance.close_window()
            except tk.TclError: pass
            self.progress_window_instance = None

        self.progress_window_instance = ProgressWindow(
            self, self.progress_var, self.cancel_event, 
            style_name=self.custom_progress_bar_style_name
        )
        self.progress_window_instance.set_progress_text(f"Запуск для '{params.get('target_type', 'N/A')}'...")
        self.progress_window_instance.lift() 

        target_type = params.get('target_type')
        sim_func = None
        if target_type in [config.TARGET_DISK, config.TARGET_DOME]: sim_func = simulation.simulate_coating_disk_dome
        elif target_type == config.TARGET_LINEAR: sim_func = simulation.simulate_linear_movement
        elif target_type == config.TARGET_PLANETARY: sim_func = simulation.simulate_planetary
        
        if sim_func is None:
            messagebox.showerror("Ошибка", f"Неизвестный тип мишени для симуляции: {target_type}", parent=self)
            self.control_frame.enable_run_button()
            if self.progress_window_instance: self.progress_window_instance.close_window(); self.progress_window_instance = None
            return

        print(f"Запуск симуляции для '{target_type}' с {params.get('particles', 'N/A')} частицами...")
        self.simulation_thread = threading.Thread(
            target=self._simulation_worker,
            args=(sim_func, params, self.cancel_event, self.results_queue, self.progress_queue),
            daemon=True 
        )
        self.simulation_thread.start()

    def _simulation_worker(self, sim_func, params, cancel_event, results_q, progress_q):
        """ Worker function executed in the simulation thread. """
        start_time = time.time()
        results = None; error = None
        try:
            def progress_update(value): 
                progress_val = max(0, min(100, int(value)))
                progress_q.put(progress_val)
            results = sim_func(params, progress_callback=progress_update, cancel_event=cancel_event)
        except Exception as e: 
            error = e
            print("--- Ошибка в потоке симуляции ---")
            traceback.print_exc()
            print("---------------------------------")
        finally:
            end_time = time.time(); duration = end_time - start_time
            sim_status = "отменена" if cancel_event.is_set() else ("завершена с ошибкой" if error else "успешно завершена")
            print(f"Симуляция {sim_status} за {duration:.2f} секунд.")
            results_q.put({'results': results, 'error': error, 'params': params, 'cancelled': cancel_event.is_set()})

    def _check_queues(self):
        """ Periodically checks the progress and results queues. """
        try:
            while True: 
                progress = self.progress_queue.get_nowait()
                self.progress_var.set(progress)
                if self.progress_window_instance and self.progress_window_instance.winfo_exists():
                    self.progress_window_instance.set_progress_text(f"Выполнение: {progress}%")
        except queue.Empty: pass 
        except Exception as e: print(f"Error processing progress queue: {e}") 

        try:
            result_data = self.results_queue.get_nowait()
            self._handle_simulation_completion(result_data)
        except queue.Empty: pass 
        except Exception as e:
             print(f"Error processing results queue: {e}") 
             if not (self.simulation_thread and self.simulation_thread.is_alive()):
                 self.control_frame.enable_run_button()
                 if self.last_results is not None:
                     self.control_frame.enable_export_button()

        self.after(100, self._check_queues) 

    def _handle_simulation_completion(self, result_data):
        """ Handles the data received from the simulation thread. """
        if self.progress_window_instance and self.progress_window_instance.winfo_exists():
            self.progress_window_instance.close_window()
        self.progress_window_instance = None

        if not result_data.get('cancelled', False):
            self.progress_var.set(100)
        
        self.control_frame.enable_run_button() 

        error = result_data.get('error')
        results = result_data.get('results')
        params = result_data.get('params') 
        cancelled = result_data.get('cancelled', False)

        if cancelled:
            messagebox.showinfo("Симуляция", "Симуляция была отменена пользователем.", parent=self)
            self.progress_var.set(0) 
            return 

        if error: 
            messagebox.showerror("Ошибка Симуляции", f"Во время симуляции произошла ошибка:\n{error}", parent=self)
            self.progress_var.set(0) 
        elif results is not None:
            if isinstance(results, tuple) and len(results) == 4 and all(isinstance(arr, np.ndarray) for arr in results):
                self.last_results = results
                self.control_frame.enable_export_button() 
                
                if messagebox.askyesno("Симуляция Завершена", "Симуляция успешно завершена.\nПоказать результаты?", parent=self):
                    vis_params = {'percent': True, 'logscale': False, 'show3d': False} 
                    try: 
                        ResultsWindow(self, *results, target_params=params, vis_params=vis_params)
                    except Exception as e: 
                        messagebox.showerror("Ошибка Отображения", f"Не удалось открыть окно результатов:\n{e}", parent=self)
                        traceback.print_exc()
            else:
                 messagebox.showwarning("Симуляция", "Симуляция завершилась, но вернула некорректные результаты.", parent=self)
                 self.last_results = None
                 self.control_frame.disable_export_button()
                 print(f"Unexpected results format: {type(results)}")

        else: 
            messagebox.showwarning("Симуляция", "Симуляция завершилась, но не вернула результатов.", parent=self)
            self.last_results = None
            self.control_frame.disable_export_button()

        if not cancelled:
            self.after(1500, lambda: self.progress_var.set(0))


    def export_results(self):
        """ Exports the last simulation results to CSV. """
        if self.last_results is None: 
            messagebox.showwarning("Экспорт", "Нет результатов для экспорта. Запустите симуляцию.", parent=self)
            return
        
        params_current = self._gather_parameters()
        if not params_current: 
            messagebox.showerror("Ошибка Экспорта", "Не удалось получить текущие параметры для экспорта.", parent=self)
            return 
            
        vis_params_export = {'percent': True} 
        try:
            export.export_csv(*self.last_results, target_params=params_current, vis_params=vis_params_export)
        except Exception as e: 
            messagebox.showerror("Ошибка Экспорта", f"Не удалось экспортировать данные:\n{e}", parent=self)
            traceback.print_exc()

    def on_closing(self):
        """ Handles the window closing event. """
        if self.simulation_thread and self.simulation_thread.is_alive():
            if messagebox.askyesno("Выход", "Симуляция еще выполняется. Прервать и выйти?", parent=self):
                 self.cancel_event.set() 
                 self.simulation_thread.join(timeout=0.2) 
                 
                 if self.progress_window_instance and self.progress_window_instance.winfo_exists():
                     try: self.progress_window_instance.destroy()
                     except tk.TclError: pass
                 if self.geometry_window_instance and self.geometry_window_instance.winfo_exists():
                     try: self.geometry_window_instance.destroy()
                     except tk.TclError: pass
                 
                 self.destroy() 
            else: 
                return 
        else:
            if self.progress_window_instance and self.progress_window_instance.winfo_exists():
                try: self.progress_window_instance.destroy()
                except tk.TclError: pass
            if self.geometry_window_instance and self.geometry_window_instance.winfo_exists():
                try: self.geometry_window_instance.destroy()
                except tk.TclError: pass
            
            self.destroy() 

