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
        self.samplerate = 44100  # Standard audio sample rate
        self.blocksize = 1024    # Better block size for processing
        self.volume_window = []
        self.window_size = 25
        self.volume_threshold_factor = 1.20
        self.valley_threshold_factor = 1.00
        self.recent_frequencies = []  
        self.hysteresis = 0.2
        self.hysteresis_multiplier = 1.5
        self.waiting_for_valley = False
        self.hysteresis_start_time = 0
        
        # Bass level tracking for peak validation
        self.bass_levels = []  # Stores bass levels when valid peaks are detected
        self.bass_window_size = 10  # Number of bass levels to keep for average
        
        # Peak timing tracking
        self.peak_timestamps = []  # Stores timestamps of valid peaks
        self.peak_interval_window_size = 5  # Number of intervals to average
        self.early_peak_threshold = 1.25  # 15% above normal for early peaks (configurable)
        self.timing_buffer = 0.100  # 100ms buffer before expected peak (configurable)

        # Variáveis para controle de limpeza do histórico
        self.last_audio_time = time.time()  # Timestamp da última recepção de áudio significativo
        self.silence_threshold = 0.5  # 0.5 segundos de silêncio
        self.bass_cleaned = False  # Flag para evitar limpeza repetida
        self.silence_volume_threshold = 0.01  # Threshold mínimo para considerar como "áudio real"

        # Audio monitoring control
        self.audio_running = True

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
        self.line, = self.ax.plot(np.arange(self.blocksize), np.zeros(self.blocksize), lw=2, color='lime')
        self.canvas_graph = FigureCanvasTkAgg(self.fig, master=frame)  
        self.canvas_graph.draw()
        self.canvas_graph.get_tk_widget().config(bg='black')
        self.canvas_graph.get_tk_widget().pack(pady=10)

        # Log box below the graph
        self.log_text = tk.Text(frame, height=10, width=40, bg="white", fg="black")
        self.log_text.pack(pady=5, expand=True, fill=tk.BOTH)

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
        
        # Inicia o monitor de silêncio na thread principal
        self.root.after(100, self.check_audio_silence)
        
        try:
            # Usa soundcard para capturar áudio do sistema (loopback)
            speaker = sc.default_speaker()
            
            # Usa uma versão mais robusta do sounddevice para loopback
            # Lista dispositivos disponíveis para encontrar o loopback
            devices = sd.query_devices()
            loopback_device = None
            
            for i, device in enumerate(devices):
                if 'loopback' in device['name'].lower() or 'stereo mix' in device['name'].lower():
                    loopback_device = i
                    break
            
            if loopback_device is not None:
                print(f"Usando dispositivo loopback: {devices[loopback_device]['name']}")
                
                def audio_callback(indata, frames, time, status):
                    if status:
                        print(f"Audio status: {status}")
                    self.process_audio_data(indata)
                
                with sd.InputStream(callback=audio_callback, 
                                  device=loopback_device,
                                  channels=1, 
                                  samplerate=self.samplerate, 
                                  blocksize=self.blocksize):
                    print("Audio monitoring started...")
                    while self.audio_running:
                        time.sleep(0.1)
            else:
                # Fallback para soundcard se não encontrar dispositivo loopback
                print("Dispositivo loopback não encontrado, usando soundcard...")
                self.fallback_soundcard_audio()
                    
        except Exception as e:
            print(f"Error: on audio {e}")
            # Tenta fallback para soundcard
            self.fallback_soundcard_audio()
        finally:
            pythoncom.CoUninitialize()

    def fallback_soundcard_audio(self):
        """Fallback usando soundcard para capturar áudio do sistema"""
        try:
            # Patch temporário para numpy compatibility
            import soundcard.mediafoundation
            original_fromstring = np.fromstring
            
            def patched_fromstring(string, dtype, **kwargs):
                if hasattr(string, 'raw'):
                    return np.frombuffer(string.raw, dtype=dtype, **kwargs)
                else:
                    return np.frombuffer(string, dtype=dtype, **kwargs)
            
            np.fromstring = patched_fromstring
            
            speaker = sc.default_speaker()
            mic = sc.get_microphone(speaker.name, include_loopback=True)
            
            with mic.recorder(samplerate=self.samplerate) as mic_stream:
                print("Audio monitoring started with soundcard...")
                
                while self.audio_running:
                    try:
                        data = mic_stream.record(numframes=self.blocksize)
                        self.process_audio_data(data)
                    except Exception as e:
                        print(f"Erro na captura: {e}")
                        time.sleep(0.1)
                        
        except Exception as e:
            print(f"Erro no soundcard fallback: {e}")
        finally:
            # Restaura o método original
            np.fromstring = original_fromstring

    def process_audio_data(self, data):
        """Processa os dados de áudio capturados"""
        try:
            # Fix for numpy compatibility
            data = np.asarray(data, dtype=np.float32)
            if data.ndim > 1:
                data = data.flatten()
            
            raw_volume = np.sqrt(np.mean(data**2))
            
            # Só atualiza last_audio_time se o volume for significativo (não é silêncio)
            current_time = time.time()
            if raw_volume > self.silence_volume_threshold:
                self.last_audio_time = current_time
                
                # Se estava em silêncio e agora recebeu áudio significativo, resetar flag de limpeza
                if self.bass_cleaned:
                    self.bass_cleaned = False
                    print("🔊 Áudio significativo detectado novamente - Flag de limpeza resetada")
            
            self.volume_window.append(raw_volume)

            if len(self.volume_window) > self.window_size:
                self.volume_window.pop(0)

            dominant_freq = self.analyze_frequencies(data)
            self.recent_frequencies.append(dominant_freq)
            if len(self.recent_frequencies) > self.window_size:
                self.recent_frequencies.pop(0)

            # Analyze bass level for this audio chunk
            bass_level = self.analyze_bass_level(data)

            # Schedule visualization update on main thread
            self.root.after_idle(lambda: self.update_visualization(data))

            if len(self.volume_window) == self.window_size:
                self.dynamic_detection(bass_level)
                
        except Exception as e:
            print(f"Erro no processamento de áudio: {e}")

    def check_audio_silence(self):
        """Verifica se houve silêncio por mais de 0.5s e limpa histórico se necessário"""
        try:
            current_time = time.time()
            time_since_last_audio = current_time - self.last_audio_time
            
            # Se passou do threshold de silêncio E ainda não limpou
            if time_since_last_audio > self.silence_threshold and not self.bass_cleaned:
                if len(self.bass_levels) > 0:
                    print(f"🧹 Silêncio detectado por {time_since_last_audio:.2f}s - Limpando histórico de {len(self.bass_levels)} níveis de grave")
                    self.bass_levels.clear()
                    # Também limpa os timestamps para reset completo
                    self.peak_timestamps.clear()
                    self.bass_cleaned = True
                    print("🔄 Histórico de timing também limpo - Sistema resetado")
                else:
                    # Marca como limpo mesmo se não havia dados
                    self.bass_cleaned = True
                    
        except Exception as e:
            print(f"Erro na verificação de silêncio: {e}")
        
        # Agenda próxima verificação em 100ms
        if self.audio_running:
            self.root.after(100, self.check_audio_silence)

    def analyze_frequencies(self, audio_data):
        """Aplica FFT no áudio e retorna a frequência dominante"""
        try:
            if len(audio_data) < 2:
                return 0.0
                
            fft_result = np.fft.fft(audio_data)
            freqs = np.fft.fftfreq(len(fft_result), d=1/self.samplerate)
            magnitude = np.abs(fft_result)
            
            # Ignora frequência DC (0 Hz)
            magnitude[0] = 0
            
            if len(magnitude) == 0:
                return 0.0
                
            dominant_freq = freqs[np.argmax(magnitude)]
            return abs(dominant_freq)
        except Exception as e:
            print(f"Erro na análise de frequência: {e}")
            return 0.0

    def analyze_bass_level(self, audio_data):
        """Analisa o nível de graves no áudio (frequências baixas 20-250 Hz)"""
        try:
            if len(audio_data) < 2:
                return 0.0
                
            fft_result = np.fft.fft(audio_data)
            freqs = np.fft.fftfreq(len(fft_result), d=1/self.samplerate)
            magnitude = np.abs(fft_result)
            
            # Define range de frequências graves (20-250 Hz)
            bass_mask = (np.abs(freqs) >= 20) & (np.abs(freqs) <= 250)
            
            if np.any(bass_mask):
                bass_magnitude = magnitude[bass_mask]
                bass_level = np.mean(bass_magnitude)
                return bass_level
            else:
                return 0.0
                
        except Exception as e:
            print(f"Erro na análise de graves: {e}")
            return 0.0
        
    def dynamic_detection(self, current_bass_level):
        """Modifica a lógica de detecção de som com validação de graves e timing"""
        if len(self.recent_frequencies) == 0:
            return

        try:
            baseline = np.percentile(self.volume_window, 70)
            valley_base = np.percentile(self.volume_window, 30)
            peak_threshold = baseline * self.volume_threshold_factor
            valley_threshold = valley_base * self.valley_threshold_factor
            current_volume = self.volume_window[-1]
            current_time = time.time()

            avg_frequency = np.mean(self.recent_frequencies)

            if hasattr(self, 'waiting_for_valley') and self.waiting_for_valley and hasattr(self, 'hysteresis_start_time') and (current_time - self.hysteresis_start_time) < self.hysteresis:
                peak_threshold *= self.hysteresis_multiplier

            # Check if current volume exceeds threshold
            if current_volume > peak_threshold:
                if not hasattr(self, 'waiting_for_valley') or not self.waiting_for_valley:
                    
                    # Validate peak using bass level average
                    is_valid_peak = True
                    
                    if len(self.bass_levels) > 0:
                        # Calculate average bass level from previous valid peaks
                        avg_bass_level = np.mean(self.bass_levels)
                        
                        # Peak is valid if bass level is sufficient
                        if current_bass_level < (avg_bass_level * 0.5):  # Bass too low
                            is_valid_peak = False
                            print(f"❌ Falso pico detectado - Grave muito baixo: {current_bass_level:.4f} (média: {avg_bass_level:.4f})")
                    
                    # Check peak timing if we have previous peaks
                    if is_valid_peak and len(self.peak_timestamps) > 0:
                        time_since_last_peak = current_time - self.peak_timestamps[-1]
                        
                        # Calculate average interval between peaks
                        if len(self.peak_timestamps) >= 2:
                            intervals = []
                            for i in range(1, min(len(self.peak_timestamps), self.peak_interval_window_size + 1)):
                                interval = self.peak_timestamps[-i] - self.peak_timestamps[-i-1]
                                intervals.append(interval)
                            
                            avg_interval = np.mean(intervals)
                            # Apply timing buffer - expect peak 100ms earlier than average
                            expected_time = avg_interval - self.timing_buffer
                            
                            # If peak is coming too early (before expected time), require higher threshold
                            if time_since_last_peak < expected_time:
                                required_threshold = peak_threshold * self.early_peak_threshold
                                if current_volume < required_threshold:
                                    is_valid_peak = False
                                    print(f"❌ Pico muito cedo - Requer {self.early_peak_threshold:.0%} acima do normal.")
                                    print(f"   Intervalo atual: {time_since_last_peak:.3f}s, Esperado: {expected_time:.3f}s (média: {avg_interval:.3f}s - buffer: {self.timing_buffer:.3f}s)")
                                    print(f"   Volume atual: {current_volume:.4f}, Necessário: {required_threshold:.4f}")
                                else:
                                    print(f"✅ Pico cedo mas forte o suficiente: {current_volume:.4f} ≥ {required_threshold:.4f}")
                            else:
                                print(f"⏰ Pico no timing esperado: {time_since_last_peak:.3f}s ≥ {expected_time:.3f}s")
                    
                    if is_valid_peak:
                        # Record bass level and timestamp for this valid peak
                        self.bass_levels.append(current_bass_level)
                        if len(self.bass_levels) > self.bass_window_size:
                            self.bass_levels.pop(0)
                        
                        self.peak_timestamps.append(current_time)
                        if len(self.peak_timestamps) > self.peak_interval_window_size:
                            self.peak_timestamps.pop(0)
                        
                        avg_bass = np.mean(self.bass_levels) if len(self.bass_levels) > 0 else current_bass_level
                        
                        if len(self.peak_timestamps) >= 2:
                            last_interval = self.peak_timestamps[-1] - self.peak_timestamps[-2]
                            # Calculate current average for display
                            intervals = []
                            for i in range(1, min(len(self.peak_timestamps), self.peak_interval_window_size + 1)):
                                interval = self.peak_timestamps[-i] - self.peak_timestamps[-i-1]
                                intervals.append(interval)
                            current_avg_interval = np.mean(intervals)
                            
                            print(f"🔄 Pico válido - Freq: {avg_frequency:.2f} Hz | Volume: {current_volume:.4f} | Grave: {current_bass_level:.4f} (média: {avg_bass:.4f})")
                            print(f"   Intervalo: {last_interval:.3f}s | Média de intervalos: {current_avg_interval:.3f}s")
                        else:
                            print(f"🔄 Pico válido - Freq: {avg_frequency:.2f} Hz | Volume: {current_volume:.4f} | Grave: {current_bass_level:.4f} (média: {avg_bass:.4f})")
                        
                        for player in self.players:
                            if player.running:  
                                player.next_image()
                                
                        self.waiting_for_valley = True
                        self.hysteresis_start_time = current_time

            elif current_volume < valley_threshold:
                self.waiting_for_valley = False
        except Exception as e:
            print(f"Erro na detecção: {e}")
                
    def update_visualization(self, data):
        """Atualiza a visualização da forma de onda"""
        try:
            if hasattr(self, 'line') and len(data) > 0:
                # Normaliza os dados para melhor visualização
                normalized_data = data / (np.max(np.abs(data)) + 1e-10)
                
                # Limita o tamanho dos dados para a visualização
                if len(normalized_data) > self.blocksize:
                    normalized_data = normalized_data[:self.blocksize]
                elif len(normalized_data) < self.blocksize:
                    # Pad with zeros if needed
                    padded_data = np.zeros(self.blocksize)
                    padded_data[:len(normalized_data)] = normalized_data
                    normalized_data = padded_data
                
                self.line.set_ydata(normalized_data)
                self.line.set_xdata(np.arange(len(normalized_data)))
                
                # Force canvas update
                self.canvas_graph.draw_idle()
                
        except Exception as e:
            print(f"Erro na visualização: {e}")

    def update_threshold_factor(self, value):
        self.volume_threshold_factor = float(value)

    def update_valley_factor(self, value):
        self.valley_threshold_factor = float(value)

    def exit_viewer(self, event=None):
        """Fecha todos os players, encerra as threads e finaliza o programa de forma segura."""
        print("Exiting...")

        # Para o monitoramento de áudio
        self.audio_running = False

        # Closes all players
        for player in self.players:
            player.on_close()

        # Closes the graphical interface
        self.root.quit()  
        self.root.destroy()  


if __name__ == "__main__":
    ImageViewer()
