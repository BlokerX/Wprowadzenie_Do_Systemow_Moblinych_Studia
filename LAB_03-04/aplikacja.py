import tkinter as tk
from tkinter import ttk
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class BaseStationSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("Symulator Stacji Bazowej (M/M/S/S) - Zgodny z Lab")
        self.root.geometry("1450x850")
        
        self.is_running = False
        self.is_paused = False
        self.current_time = 0
        self.arrivals = []
        self.channels = []
        
        # Statystyki zgodne z PDF
        self.handled_calls = 0
        self.blocked_O = 0  # Bo
        self.blocked_H = 0  # BH
        self.history_time = []
        self.history_Q = []
        self.history_W = []
        self.history_Ro = []

        self.setup_ui()

    def setup_ui(self):
        # --- RAMKA PARAMETRÓW ---
        param_frame = tk.LabelFrame(self.root, text="Parametry Systemu")
        param_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nw")

        # Pola zgodne z Zadaniem 2 [cite: 78-86]
        self.fields = [
            ("Liczba kanałów (S)", "10"),
            ("Kanały rezerwowe (Sr)", "2"),
            ("Długość kolejki", "5"),
            ("Intensywność nowych (lambda_O)", "0.5"),
            ("Intensywność handov. (lambda_H)", "0.2"),
            ("Średnia rozmowa (N) [s]", "30"),
            ("Odchylenie (sigma)", "5"),
            ("Min czas [s]", "10"),
            ("Max czas [s]", "120"),
            ("Czas symulacji [s]", "100")
        ]
        self.entries = {}

        for i, (text, default) in enumerate(self.fields):
            tk.Label(param_frame, text=text).grid(row=i, column=0, sticky="e", padx=5, pady=2)
            entry = tk.Entry(param_frame, width=10)
            entry.insert(0, default)
            entry.grid(row=i, column=1, padx=5, pady=2)
            self.entries[text] = entry

        # Przyciski
        btn_frame = tk.Frame(self.root)
        btn_frame.grid(row=1, column=0, pady=10)
        tk.Button(btn_frame, text="START", command=self.start_simulation, bg="green", fg="white", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Pauza", command=self.toggle_pause).pack(side=tk.LEFT, padx=5)

        # --- WIZUALIZACJA KANAŁÓW ---
        channel_frame = tk.LabelFrame(self.root, text="Stan Kanałów")
        channel_frame.grid(row=0, column=1, rowspan=2, padx=10, pady=10, sticky="ns")

        # Kontener dla canvas i paska przewijania
        canvas_container = tk.Frame(channel_frame)
        canvas_container.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(canvas_container, width=250, bg="#f0f0f0")
        scrollbar = ttk.Scrollbar(canvas_container, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.stats_label = tk.Label(channel_frame, text="", justify="left", font=("Courier", 9))
        self.stats_label.pack(pady=5, side="bottom")

        # --- WYKRESY ---
        plot_frame = tk.Frame(self.root)
        plot_frame.grid(row=0, column=2, rowspan=3, padx=10, pady=10, sticky="nsew")
        self.fig, (self.ax_ro, self.ax_q, self.ax_w) = plt.subplots(3, 1, figsize=(6, 8))
        self.fig.tight_layout(pad=3.0)
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas_plot.get_tk_widget().pack()

    def read_params(self):
        return {
            "S": int(self.entries["Liczba kanałów (S)"].get()),
            "Sr": int(self.entries["Kanały rezerwowe (Sr)"].get()),
            "Q_max": int(self.entries["Długość kolejki"].get()),
            "lO": float(self.entries["Intensywność nowych (lambda_O)"].get()),
            "lH": float(self.entries["Intensywność handov. (lambda_H)"].get()),
            "N": float(self.entries["Średnia rozmowa (N) [s]"].get()),
            "sigma": float(self.entries["Odchylenie (sigma)"].get()),
            "min": float(self.entries["Min czas [s]"].get()),
            "max": float(self.entries["Max czas [s]"].get()),
            "time": int(self.entries["Czas symulacji [s]"].get())
        }

    def generate_arrivals(self, p):
        self.arrivals = []
        # Łączna stopa przybycia lambda = lambda_O + lambda_H [cite: 39]
        total_lambda = p["lO"] + p["lH"]
        t = 0
        while t < p["time"]:
            inter = random.expovariate(total_lambda)
            t += inter
            if t > p["time"]: break
            
            # Typ zgłoszenia na podstawie proporcji intensywności [cite: 17]
            is_handover = random.random() < (p["lH"] / total_lambda)
            
            # Czas trwania - Rozkład Gaussa [cite: 101]
            dur = np.random.normal(p["N"], p["sigma"])
            dur = max(p["min"], min(p["max"], dur))
            
            self.arrivals.append({
                "time": t,
                "duration": dur,
                "rem": dur,
                "type": "H" if is_handover else "O",
                "wait_start": 0
            })

    def start_simulation(self):
        self.p = self.read_params()
        self.generate_arrivals(self.p)
        self.channels = [None] * self.p["S"]
        self.queue = []
        self.current_time = 0
        self.blocked_O = 0
        self.blocked_H = 0
        self.handled_calls = 0
        self.history_time, self.history_Ro, self.history_Q, self.history_W = [], [], [], []
        
        self.is_running = True
        self.is_paused = False
        self.run_loop()

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        if not self.is_paused: self.run_loop()

    def _try_assign_call(self, call):
        """Helper to find a free channel and assign a call."""
        for i in range(self.p["S"]):
            if self.channels[i] is None:
                self.channels[i] = call
                # If the call was in the queue, remove it
                if call in self.queue:
                    self.queue.remove(call)
                return True
        return False

    def run_loop(self):
        if not self.is_running or self.is_paused: return
        
        if self.current_time >= self.p["time"]:
            self.is_running = False
            self.save_results()
            return

        # 1. Obsługa trwających połączeń (zmniejsz czas o 1s) [cite: 102]
        for i in range(self.p["S"]):
            if self.channels[i]:
                self.channels[i]["rem"] -= 1
                if self.channels[i]["rem"] <= 0:
                    self.channels[i] = None
                    self.handled_calls += 1

        # 2. Pobierz zgłoszenia przychodzące w tej sekundzie 
        current_batch = [a for a in self.arrivals if self.current_time <= a["time"] < self.current_time + 1]
        
        Sc = self.p["S"] - self.p["Sr"] # Próg rezerwacji 

        for call in current_batch:
            busy_count = sum(1 for c in self.channels if c is not None)
            
            # Logika rezerwacji: Odrzuć nowe (O) jeśli przekroczono Sc [cite: 55]
            if call["type"] == "O" and busy_count >= Sc:
                # Próba kolejkowania (jeśli dopuszczalne w Zadaniu 2) [cite: 85]
                if len(self.queue) < self.p["Q_max"]:
                    call["wait_start"] = self.current_time
                    self.queue.append(call)
                else:
                    self.blocked_O += 1
            else:
                # Szukaj wolnego kanału
                if not self._try_assign_call(call):
                    if call["type"] == "H": self.blocked_H += 1
                    else: self.blocked_O += 1

        # 3. Próba przesunięcia z kolejki do zwolnionych kanałów
        # Iterate over a copy of the queue as it might be modified
        for q_call in list(self.queue):
            busy_count = sum(1 for c in self.channels if c is not None)
            # Rezerwacja nadal obowiązuje przy wyciąganiu z kolejki
            if q_call["type"] == "O" and busy_count >= Sc:
                continue
            # If assignment is successful, the helper will remove it from the queue
            self._try_assign_call(q_call)

        # 4. Statystyki krokowe [cite: 105]
        ro = sum(1 for c in self.channels if c is not None) / self.p["S"]
        self.history_time.append(self.current_time)
        self.history_Ro.append(ro)
        self.history_Q.append(len(self.queue))
        # Średni czas oczekiwania (uproszczony do bieżącej kolejki)
        avg_w = np.mean([self.current_time - c["wait_start"] for c in self.queue]) if self.queue else 0
        self.history_W.append(avg_w)

        self.update_view()
        self.current_time += 1
        self.root.after(50, self.run_loop)

    def update_view(self):
        self.canvas.delete("all")
        for i in range(self.p["S"]):
            x = 20
            y = 10 + i * 25
            color = "white"
            txt = f"K{i+1}: Wolny"
            if self.channels[i]:
                color = "orange" if self.channels[i]["type"] == "H" else "lightgreen"
                txt = f"K{i+1}: {self.channels[i]['type']} ({int(self.channels[i]['rem'])}s)"
            
            # Wyróżnienie kanałów rezerwowych 
            if i >= (self.p["S"] - self.p["Sr"]):
                self.canvas.create_rectangle(x-5, y, x+200, y+20, outline="red", dash=(2,2))

            self.canvas.create_rectangle(x, y, x+180, y+20, fill=color)
            self.canvas.create_text(x+90, y+10, text=txt)

        # Ustawienie regionu przewijania, aby dopasować do zawartości
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

        self.stats_label.config(text=f"Czas: {self.current_time}/{self.p['time']}s\n"
                                     f"Obsłużone: {self.handled_calls}\n"
                                     f"Blokady Bo (Nowe): {self.blocked_O}\n"
                                     f"Blokady BH (Hand.): {self.blocked_H}\n"
                                     f"W kolejce: {len(self.queue)}")

        # Wykresy [cite: 91]
        for ax, hist, title, col in zip([self.ax_ro, self.ax_q, self.ax_w], 
                                        [self.history_Ro, self.history_Q, self.history_W],
                                        ["Ro - Intensywność", "Q - Długość kolejki", "W - Czas oczekiwania"],
                                        ["g", "r", "b"]):
            ax.clear()
            ax.plot(self.history_time, hist, color=col)
            ax.set_title(title)
        self.canvas_plot.draw()

    def save_results(self):
        # Zapis do pliku zgodnie z pkt 3.c [cite: 105, 92]
        with open("wyniki_lab3.txt", "w") as f:
            f.write(f"PARAMETRY: S={self.p['S']}, Sr={self.p['Sr']}, lO={self.p['lO']}, lH={self.p['lH']}\n")
            f.write("Czas\tRo\tQ\tW\n")
            for t, r, q, w in zip(self.history_time, self.history_Ro, self.history_Q, self.history_W):
                f.write(f"{t}\t{r:.2f}\t{q}\t{w:.2f}\n")
        print("Zapisano do wyniki_lab3.txt")

if __name__ == "__main__":
    root = tk.Tk()
    app = BaseStationSimulator(root)
    root.mainloop()