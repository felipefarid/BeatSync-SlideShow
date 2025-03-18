import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import numpy as np
import threading
import time
import sounddevice as sd
import soundcard as sc
import random
import warnings
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pythoncom
import sys

# Settings to suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
from soundcard import SoundcardRuntimeWarning
warnings.filterwarnings("ignore", category=SoundcardRuntimeWarning)

class TextRedirector:
    """Redirects prints to a Tkinter widget"""
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        self.text_widget.insert(tk.END, message)
        self.text_widget.see(tk.END) # Auto-scroll to the last line

    def flush(self):
        pass  # Required for compatibility with standard stdout

class Player:
    def __init__(self, root, image_paths, random_order):
        self.root = root
        self.image_paths = image_paths
        self.random_order = random_order
        self.image_index = 0
        self.running = True

        # Create a new window for the player
        self.viewer_window = tk.Toplevel(self.root)
        self.viewer_window.title("Player")
        self.viewer_window.geometry("800x600")

        self.canvas = tk.Canvas(self.viewer_window, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Configure the window close event
        self.viewer_window.protocol("WM_DELETE_WINDOW", self.on_close)

        # Display the first image
        self.show_image()

    def show_image(self, attempt=1):
        """Exibe a imagem atual na tela com tratamento de erros reforçado"""
        if not self.image_paths or attempt > 5:
            print("Error: Can't load images")
            return

        try:
            img = Image.open(self.image_paths[self.image_index])
            
            # Wait until the canvas has valid dimensions (up to 1 second)
            timeout = time.time() + 1.0  # 1-second timeout
            while True:
                self.viewer_window.update()  # Force a complete window update
                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()
                if canvas_width > 0 and canvas_height > 0:
                    break
                if time.time() > timeout:
                    print("Error: canvas")
                    return
                time.sleep(0.05)
            
            # Resize the image using the obtained dimensions
            img = self.resize_image(img, canvas_width, canvas_height)
            self.tk_image = ImageTk.PhotoImage(img)
            
            self.canvas.delete("all")
            x = (canvas_width - img.width) // 2
            y = (canvas_height - img.height) // 2
            self.canvas.create_image(x, y, anchor=tk.NW, image=self.tk_image)
        except Exception as e:
            print(f"Tentativa {attempt}: Error: load img - {e}")
            if self.running:  # Só tenta novamente se o player ainda estiver ativo
                self.root.after(100, lambda: self.show_image(attempt + 1))

    def resize_image(self, img, canvas_width, canvas_height):
        """Redimensiona a imagem para caber na tela utilizando as dimensões do canvas"""
        # If the canvas dimensions are invalid, use fallback values
        canvas_width = canvas_width if canvas_width > 0 else 800
        canvas_height = canvas_height if canvas_height > 0 else 600

        img_ratio = img.width / img.height
        canvas_ratio = canvas_width / canvas_height

        if img_ratio > canvas_ratio:
            new_width = canvas_width
            new_height = max(int(canvas_width / img_ratio), 1)
        else:
            new_height = canvas_height
            new_width = max(int(canvas_height * img_ratio), 1)

        return img.resize((new_width, new_height), Image.LANCZOS)

    def next_image(self):
        """Avança para a próxima imagem"""
        if not self.image_paths:
            print("Error: No img on folder.")
            return

        # Increment the image index
        self.image_index = (self.image_index + 1) % len(self.image_paths)
        self.show_image()

    def on_close(self):
        """Fecha o player e marca como inativo"""
        self.running = False
        self.viewer_window.destroy()

class ImageViewer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Control Panel")
        self.root.geometry("400x600")
        self.root.protocol("WM_DELETE_WINDOW", self.exit_viewer)

        # Main settings
        self.players = []  # List of active players
        self.include_subfolders = tk.BooleanVar(value=True)
        self.random_order = tk.BooleanVar(value=True)
        self.image_formats = ["jpg", "jpeg", "png", "bmp", "gif"]

        # Audio settings
        self.samplerate = 1024
        self.blocksize = 32
        self.volume_window = []
        self.window_size = 25
        self.volume_threshold_factor = 1.20
        self.valley_threshold_factor = 0.90
        self.recent_frequencies = []  
        self.hysteresis = 0.2
        self.hysteresis_multiplier = 1.5
        self.waiting_for_valley = False

        # Graphical interface
        self.setup_initial_interface()
        self.audio_thread = threading.Thread(target=self.listen_to_audio, daemon=True)
        self.audio_thread.start()
        self.root.mainloop()

    def setup_initial_interface(self):
        """Configura a interface inicial"""
        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack(fill=tk.BOTH, expand=True)

        # Button to select folder
        self.folder_button = tk.Button(self.control_frame, text="Select Folder", command=self.select_folder)
        self.folder_button.pack(pady=10)

        # Input for image formats
        self.format_label = tk.Label(self.control_frame, text="Image Formats (separated by commas)")
        self.format_label.pack(pady=5)
        self.format_entry = tk.Entry(self.control_frame)
        self.format_entry.insert(0, ",".join(self.image_formats))
        self.format_entry.pack(pady=5)

        # Checkbox to include subfolders
        self.subfolder_check = tk.Checkbutton(self.control_frame, text="Include Subfolders", variable=self.include_subfolders)
        self.subfolder_check.pack(pady=5)

        # Checkbox for random order
        self.random_check = tk.Checkbutton(self.control_frame, text="Shuffled Order", variable=self.random_order)
        self.random_check.pack(pady=5)

        # Control sliders
        self.setup_interface()

        # Waveform visualization
        self.setup_waveform_view()

    def setup_interface(self):
        """Configura os sliders de controle"""
        config_frame = tk.Frame(self.control_frame, bg="gray")
        config_frame.pack(pady=10)

        # Slider for peak factor
        self.threshold_slider = tk.Scale(
            config_frame, from_=1.10, to_=2.5, orient="horizontal",
            resolution=0.01, label="Peak Factor", command=lambda v: self.update_threshold_factor(v)
        )
        self.threshold_slider.set(self.volume_threshold_factor)
        self.threshold_slider.pack()

        # Slider for valley factor
        self.valley_slider = tk.Scale(
            config_frame, from_=0.5, to_=1.0, orient="horizontal",
            resolution=0.01, label="Valley Factor", command=lambda v: self.update_valley_factor(v)
        )
        self.valley_slider.set(self.valley_threshold_factor)
        self.valley_slider.pack()

    def setup_waveform_view(self):
        """Configura a visualização da forma de onda e a caixa de logs"""
        frame = tk.Frame(self.control_frame)
        frame.pack(pady=10, fill=tk.BOTH, expand=True)

        # Waveform visualization
        self.fig, self.ax = plt.subplots(figsize=(4, 2))
        self.fig.patch.set_alpha(0)
        self.ax.set_facecolor('none')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        self.ax.set_ylim(-1, 1)
        self.ax.set_xlim(0, self.blocksize)
        self.line, = self.ax.plot([], [], lw=2, color='red')
        self.canvas_graph = FigureCanvasTkAgg(self.fig, master=frame)  
        self.canvas_graph.draw()
        self.canvas_graph.get_tk_widget().config(bg='black')
        self.canvas_graph.get_tk_widget().pack(pady=10)

        # Log box below the graph
        self.log_text = tk.Text(frame, height=10, width=40, bg="white", fg="black")
        self.log_text.pack(pady=5, expand=True, fill=tk.BOTH)  # Removido 'side=tk.RIGHT', agora está abaixo

        # Redirect prints to the log box
        sys.stdout = TextRedirector(self.log_text)
        
    def select_folder(self):
        """Seleciona a pasta com as imagens"""
        folder_path = filedialog.askdirectory(title="Selecione a pasta com as imagens")
        if not folder_path:
            return

        # Updates the image formats
        self.image_formats = [fmt.strip().lower() for fmt in self.format_entry.get().split(",")]

        # Loads the images
        image_paths = self.load_images(folder_path)

        if image_paths:
            # Creates a new player with the loaded images
            player = Player(self.root, image_paths, self.random_order.get())
            self.players.append(player)

    def load_images(self, folder_path):
        """Carrega as imagens da pasta selecionada"""
        image_paths = []
        if self.include_subfolders.get():
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in self.image_formats):
                        image_paths.append(os.path.join(root, file))
        else:
            for file in os.listdir(folder_path):
                if any(file.lower().endswith(ext) for ext in self.image_formats):
                    image_paths.append(os.path.join(folder_path, file))

        if not image_paths:
            messagebox.showerror("Erro", "Nenhuma imagem válida encontrada!")
            return []

        if self.random_order.get():
            random.shuffle(image_paths)

        return image_paths

    def listen_to_audio(self):
        """Thread principal de captura e análise de áudio"""
        pythoncom.CoInitialize()
        
        try:
            speaker = sc.default_speaker()
            mic = sc.get_microphone(speaker.name, include_loopback=True)
            
            with mic.recorder(samplerate=self.samplerate) as mic_stream:
                print("Audio monitoring started...")
                
                while True:
                    data = mic_stream.record(numframes=self.blocksize)
                    self.process_audio_data(data)
                    
        except Exception as e:
            print(f"Error: on audio {e}")
            self.exit_viewer()
        finally:
            pythoncom.CoUninitialize()

    def process_audio_data(self, data):
        """Processa os dados de áudio capturados"""
        raw_volume = np.sqrt(np.mean(data.astype(float)**2))
        self.volume_window.append(raw_volume)

        if len(self.volume_window) > self.window_size:
            self.volume_window.pop(0)

        dominant_freq = self.analyze_frequencies(data)
        self.recent_frequencies.append(dominant_freq)
        if len(self.recent_frequencies) > self.window_size:
            self.recent_frequencies.pop(0)

        self.update_visualization(data.astype(float).flatten())

        if len(self.volume_window) == self.window_size:
            self.dynamic_detection()

    def analyze_frequencies(self, audio_data):
        """Aplica FFT no áudio e retorna a frequência dominante"""
        fft_result = np.fft.fft(audio_data.flatten())
        freqs = np.fft.fftfreq(len(fft_result), d=1/self.samplerate)
        magnitude = np.abs(fft_result)
        dominant_freq = freqs[np.argmax(magnitude)]
        return abs(dominant_freq)
        
    def dynamic_detection(self):
        """Modifica a lógica de detecção de som com base na média das últimas frequências"""
        if len(self.recent_frequencies) == 0:
            return

        baseline = np.percentile(self.volume_window, 70)
        valley_base = np.percentile(self.volume_window, 30)
        peak_threshold = baseline * self.volume_threshold_factor
        valley_threshold = valley_base * self.valley_threshold_factor
        current_volume = self.volume_window[-1]

        avg_frequency = np.mean(self.recent_frequencies)

        if self.waiting_for_valley and (time.time() - self.hysteresis_start_time) < self.hysteresis:
            peak_threshold *= self.hysteresis_multiplier

        if current_volume > peak_threshold:
            if not self.waiting_for_valley:
                print(f"🔄 Frequência: {avg_frequency:.2f} Hz")
                for player in self.players:
                    if player.running:  
                        player.next_image()
                self.waiting_for_valley = True
                self.hysteresis_start_time = time.time()

        elif current_volume < valley_threshold:
            self.waiting_for_valley = False
                
    def update_visualization(self, data):
        """Atualiza a visualização da forma de onda"""
        self.line.set_ydata(data)
        self.line.set_xdata(np.arange(len(data)))
        self.canvas_graph.draw()        

    def update_threshold_factor(self, value):
        self.volume_threshold_factor = float(value)

    def update_valley_factor(self, value):
        self.valley_threshold_factor = float(value)

    def exit_viewer(self, event=None):
        """Fecha todos os players, encerra as threads e finaliza o programa de forma segura."""
        print("Exiting...")

        # Closes all players
        for player in self.players:
            player.on_close()

        # Waits for the audio thread to finish before closing
        if self.audio_thread.is_alive():
            self.audio_thread.join(timeout=1)  

        # Closes the graphical interface
        self.root.quit()  
        self.root.destroy()  


if __name__ == "__main__":
    ImageViewer()