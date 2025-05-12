# coating_simulator_project/coating_simulator/gui/progress_window.py
"""
Toplevel window for displaying simulation progress.
Дочернее окно Toplevel для отображения прогресса симуляции.
"""
import tkinter as tk
from tkinter import ttk

class ProgressWindow(tk.Toplevel):
    """
    A Toplevel window that displays a progress bar and an optional cancel button.
    Дочернее окно Toplevel, которое отображает прогресс-бар и опциональную кнопку отмены.
    """
    def __init__(self, parent, progress_variable, cancel_event=None, title="Прогресс симуляции", style_name=None): # <<< Добавлен style_name
        """
        Initializes the ProgressWindow.
        Инициализирует ProgressWindow.

        Args:
            parent: The parent widget (main application window).
                    Родительский виджет (главное окно приложения).
            progress_variable: tk.IntVar to bind to the progress bar.
                               tk.IntVar для привязки к прогресс-бару.
            cancel_event (threading.Event, optional): Event to signal cancellation.
                                                      Событие для сигнализации отмены.
            title (str): The title of the progress window.
                         Заголовок окна прогресса.
            style_name (str, optional): The ttk style to apply to the progress bar.
                                        Стиль ttk для применения к прогресс-бару.
        """
        super().__init__(parent)
        self.title(title)
        self.resizable(False, False)
        self.transient(parent)
        self.protocol("WM_DELETE_WINDOW", self._on_close_button)

        self._cancel_event = cancel_event
        self._parent = parent

        # --- Widgets ---
        main_frame = ttk.Frame(self, padding="10 10 10 10")
        main_frame.pack(expand=True, fill=tk.BOTH)

        self.progress_label = ttk.Label(main_frame, text="Выполнение симуляции...")
        self.progress_label.pack(pady=(0, 5))

        # --- !!! Применение стиля к Progressbar !!! ---
        if style_name:
            self.progress_bar = ttk.Progressbar(main_frame, orient="horizontal",
                                                length=300, mode="determinate",
                                                variable=progress_variable,
                                                style=style_name) # <<< Применяем стиль
        else:
            self.progress_bar = ttk.Progressbar(main_frame, orient="horizontal",
                                                length=300, mode="determinate",
                                                variable=progress_variable)
        # --------------------------------------------
        self.progress_bar.pack(pady=5, fill=tk.X, expand=True)

        if self._cancel_event:
            self.btn_cancel = ttk.Button(main_frame, text="Отмена", command=self._cancel_simulation)
            self.btn_cancel.pack(pady=(5, 0))
        
        self._center_window()

    def _cancel_simulation(self):
        """Signals the simulation to cancel."""
        if self._cancel_event:
            self._cancel_event.set()
        # The main app will handle closing the window upon completion/cancellation.
        # Основное приложение обработает закрытие окна по завершении/отмене.

    def _on_close_button(self):
        """Handles the window's 'X' button being pressed."""
        if self._cancel_event:
            self._cancel_simulation()
        else:
            self.withdraw() 

    def _center_window(self):
        """Centers the window on its parent."""
        self.update_idletasks() 
        parent_x = self._parent.winfo_x()
        parent_y = self._parent.winfo_y()
        parent_width = self._parent.winfo_width()
        parent_height = self._parent.winfo_height()

        window_width = self.winfo_width()
        window_height = self.winfo_height()

        center_x = parent_x + (parent_width - window_width) // 2
        center_y = parent_y + (parent_height - window_height) // 2

        self.geometry(f"+{center_x}+{center_y}")

    def set_progress_text(self, text: str):
        """Updates the text label above the progress bar."""
        self.progress_label.config(text=text)

    def close_window(self):
        """Closes the progress window."""
        try:
            if self.winfo_exists(): # Проверяем, существует ли еще окно
                self.destroy()
        except tk.TclError:
            pass # Окно уже могло быть уничтожено
