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
        self.current_image = None  # Armazena a imagem atual para enquadramentos
        self.current_frame_type = "normal"  # Tipo de enquadramento atual

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
            # Carrega a imagem original
            self.current_image = Image.open(self.image_paths[self.image_index])
            
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
            
            # Aplica o enquadramento atual (normal por padrão)
            display_img = self.apply_framing(self.current_image, canvas_width, canvas_height)
            self.tk_image = ImageTk.PhotoImage(display_img)
            
            self.canvas.delete("all")
            x = (canvas_width - display_img.width) // 2
            y = (canvas_height - display_img.height) // 2
            self.canvas.create_image(x, y, anchor=tk.NW, image=self.tk_image)
            
        except Exception as e:
            print(f"Tentativa {attempt}: Error: load img - {e}")
            if self.running:  # Só tenta novamente se o player ainda estiver ativo
                self.root.after(100, lambda: self.show_image(attempt + 1))

    def apply_framing(self, img, canvas_width, canvas_height, frame_type="normal"):
        """Aplica diferentes tipos de enquadramento na imagem"""
        self.current_frame_type = frame_type
        
        if frame_type == "normal":
            # Enquadramento normal - redimensiona para caber na tela
            return self.resize_image(img, canvas_width, canvas_height)
        else:
            # Enquadramentos em tamanho 100% - crop de diferentes regiões
            return self.crop_image_100(img, canvas_width, canvas_height, frame_type)

    def crop_image_100(self, img, canvas_width, canvas_height, crop_type):
        """Faz crop da imagem mantendo tamanho 100% e retorna região específica"""
        # Dimensões da área de visualização
        view_width = canvas_width
        view_height = canvas_height
        
        # Dimensões da imagem original
        img_width = img.width
        img_height = img.height
        
        # Se a imagem for menor que a área de visualização, centraliza
        if img_width <= view_width and img_height <= view_height:
            return img
        
        # Define as coordenadas de crop baseado no tipo
        if crop_type == "center":
            # Centro da imagem
            left = max(0, (img_width - view_width) // 2)
            top = max(0, (img_height - view_height) // 2)
            
        elif crop_type == "top_left":
            # Canto superior esquerdo
            left = 0
            top = 0
            
        elif crop_type == "top_right":
            # Canto superior direito
            left = max(0, img_width - view_width)
            top = 0
            
        elif crop_type == "bottom_left":
            # Canto inferior esquerdo
            left = 0
            top = max(0, img_height - view_height)
            
        elif crop_type == "bottom_right":
            # Canto inferior direito
            left = max(0, img_width - view_width)
            top = max(0, img_height - view_height)
            
        elif crop_type == "random":
            # Posição aleatória
            max_left = max(0, img_width - view_width)
            max_top = max(0, img_height - view_height)
            left = random.randint(0, max_left)
            top = random.randint(0, max_top)
            
        else:
            # Fallback para centro
            left = max(0, (img_width - view_width) // 2)
            top = max(0, (img_height - view_height) // 2)
        
        # Calcula as coordenadas finais do crop
        right = min(img_width, left + view_width)
        bottom = min(img_height, top + view_height)
        
        # Faz o crop da imagem
        cropped_img = img.crop((left, top, right, bottom))
        
        return cropped_img

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

    def apply_frame_to_current_image(self, frame_type):
        """Aplica um novo enquadramento à imagem atual sem trocar de imagem"""
        if not self.current_image or not self.running:
            return
            
        try:
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width <= 0 or canvas_height <= 0:
                return
                
            # Aplica o novo enquadramento
            display_img = self.apply_framing(self.current_image, canvas_width, canvas_height, frame_type)
            self.tk_image = ImageTk.PhotoImage(display_img)
            
            self.canvas.delete("all")
            x = (canvas_width - display_img.width) // 2
            y = (canvas_height - display_img.height) // 2
            self.canvas.create_image(x, y, anchor=tk.NW, image=self.tk_image)
            
            print(f"🖼️ Enquadramento aplicado: {frame_type}")
            
        except Exception as e:
            print(f"Erro ao aplicar enquadramento: {e}")

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
        self.root.geometry("400x650")  # Aumentei um pouco para comportar o novo checkbox
        self.root.protocol("WM_DELETE_WINDOW", self.exit_viewer)

        # Main settings
        self.players = []  # List of active players
        self.include_subfolders = tk.BooleanVar(value=True)
        self.random_order = tk.BooleanVar(value=True)
        self.dynamic_framing_enabled = tk.BooleanVar(value=False)  # Nova opção
        self.image_formats = ["jpg", "jpeg", "png", "bmp", "gif"]

        # Audio settings
        self.samplerate = 44100  # Standard audio sample rate
        self.blocksize = 1024    # Better block size for processing
        self.volume_window = []
        self.window_size = 10
        self.volume_threshold_factor = 1.10
        self.valley_threshold_factor = 1.00
        self.recent_frequencies = []  
        self.hysteresis = 0
        self.hysteresis_multiplier = 1
        self.waiting_for_valley = False
        self.hysteresis_start_time = 0
        
        # Configurações dos níveis de sensibilidade (expandido para 5 níveis)
        self.sensitivity_level = 1  # Nível atual (1, 2, 3)
        self.change_request_counter = 0  # Contador para controlar trocas de foto
        
        # Bass level tracking for peak validation
        self.bass_levels = []  # Stores bass levels when valid peaks are detected
        self.bass_window_size = 10  # Number of bass levels to keep for average
        
        # Peak timing tracking - SISTEMA PRINCIPAL (10 valores)
        self.peak_timestamps = []  # Stores timestamps of valid peaks (window_size=10)
        self.peak_interval_window_size = 10  # Number of intervals to average
        self.timing_buffer = 0.130  # x ms buffer before expected peak (configurable)
        self.default_interval = 0.2  # Intervalo padrão de x segundo para inicialização

        # *** NOVO SISTEMA DE MÚLTIPLAS LISTAS ***
        self.main_timing_list = []  # Lista principal (até 10 valores)
        self.secondary_timing_lists = {
            "sec_1": [],  # Lista secundária 1 (até 5 valores)
            "sec_2": [],  # Lista secundária 2 (até 5 valores) 
            "sec_3": [],  # Lista secundária 3 (até 5 valores)
            "sec_4": [],  # Lista secundária 4 (até 5 valores)
            "sec_5": [],  # Lista secundária 5 (até 5 valores)
            "sec_6": [],  # Lista secundária 6 (até 5 valores)
        }
        self.current_list_type = "main"  # Tipo de lista atual
        self.max_secondary_lists = 6 
        self.list_change_threshold = 0.3  # 30% de diferença para considerar mudança
        self.list_match_threshold = 0.15  # 15% de similaridade para trocar lista
        self.secondary_list_min_size = 3  # Mínimo de valores para considerar lista secundária válida

        # Sistema de confirmação rápida
        self.pattern_confirmation_count = 0
        self.pending_switch_list = None
        self.confirmation_threshold = 2  # Troca após 2 confirmações

        # Variáveis para controle de limpeza do histórico
        self.last_audio_time = time.time()  # Timestamp da última recepção de áudio significativo
        self.silence_threshold = 0.5  # 0.5 segundos de silêncio
        self.bass_cleaned = False  # Flag para evitar limpeza repetida
        self.silence_volume_threshold = 0.01  # Threshold mínimo para considerar como "áudio real"
        
        # Controle de visualização do gráfico (decimação)
        self.visualization_counter = 0  # Contador para controlar atualizações do gráfico
        self.visualization_decimation = 1  # Atualiza gráfico apenas 1 a cada 10 amostras

        # Audio monitoring control
        self.audio_running = True

        # Inicializa histórico com timestamps padrão de 1 segundo
        self.initialize_default_timing()

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

    def on_dynamic_framing_toggle(self):
        """Callback quando o checkbox de enquadramento dinâmico é alterado"""
        if self.dynamic_framing_enabled.get():
            print("🎯 Enquadramento Dinâmico ATIVADO")
        else:
            print("🎯 Enquadramento Dinâmico DESATIVADO")

    def setup_interface(self):
        """Configura os sliders de controle"""
        config_frame = tk.Frame(self.control_frame)
        config_frame.pack(pady=10)

        # Slider para níveis de sensibilidade (agora com 5 níveis)
        self.sensitivity_slider = tk.Scale(
            config_frame, from_=1, to_=3, orient="horizontal",
            resolution=1, label="Nível de Sensibilidade", command=lambda v: self.update_sensitivity_level(v)
        )
        self.sensitivity_slider.set(self.sensitivity_level)
        self.sensitivity_slider.pack()
        
        # Label para mostrar descrição do nível atual
        self.sensitivity_description = tk.Label(config_frame, text="Nível 2: Ignora 1 de cada 2 chamadas")
        self.sensitivity_description.pack(pady=5)

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

    # *** NOVAS FUNÇÕES PARA SISTEMA DE MÚLTIPLAS LISTAS ***

    def calculate_list_avg(self, timing_list):
        """Calcula a média de intervalos de uma lista de timing"""
        if len(timing_list) < 2:
            return self.default_interval
            
        intervals = []
        for i in range(1, len(timing_list)):
            interval = timing_list[i] - timing_list[i-1]
            intervals.append(interval)
            
        return np.mean(intervals) if intervals else self.default_interval

    def should_switch_lists(self, current_interval):
        """Troca após 2 confirmações do novo padrão"""
        if self.current_list_type != "main":
            return False
            
        # Verifica se há padrão emergente
        emerging_list = self.detect_emerging_pattern(current_interval)
        
        if emerging_list:
            # Primeira confirmação ou mesma lista?
            if self.pending_switch_list == emerging_list:
                self.pattern_confirmation_count += 1
                print(f"🔄 {self.pattern_confirmation_count}ª confirmação para {emerging_list.upper()}")
            else:
                # Novo padrão detectado - reinicia contador
                self.pattern_confirmation_count = 1
                self.pending_switch_list = emerging_list
                print(f"🔄 1ª confirmação - Padrão {emerging_list.upper()} detectado")
            
            # Troca após 2 confirmações
            if self.pattern_confirmation_count >= self.confirmation_threshold:
                print(f"✅ {self.confirmation_threshold} confirmações - Trocando para {emerging_list.upper()}!")
                return True
        else:
            # Reset contador se padrão não se confirmou
            if self.pattern_confirmation_count > 0:
                print(f"🔄 Reset contador - Padrão não confirmado")
                self.pattern_confirmation_count = 0
                self.pending_switch_list = None
                
        return False

    def find_best_matching_list(self, current_interval):
        """Encontra a lista secundária que melhor corresponde ao intervalo atual entre 6 listas"""
        best_list = None
        best_diff = float('inf')
        
        # Verifica todas as 6 listas secundárias
        for list_name, timing_list in self.secondary_timing_lists.items():
            if len(timing_list) >= self.secondary_list_min_size:
                list_avg = self.calculate_list_avg(timing_list)
                diff = abs(current_interval - list_avg) / list_avg
                if diff <= self.list_match_threshold and diff < best_diff:
                    best_list = list_name
                    best_diff = diff
        
        return best_list

    def switch_to_list(self, list_type):
        """Troca para uma lista específica com logs detalhados"""
        if list_type == self.current_list_type:
            return
            
        old_list = self.current_list_type
        self.current_list_type = list_type
        
        emojis = {
            "main": "🏠", 
            "sec_1": "📁¹", "sec_2": "📁²", "sec_3": "📁³",
            "sec_4": "📁⁴", "sec_5": "📁⁵", "sec_6": "📁⁶"
        }
        
        if list_type == "main":
            print(f"🔄 VOLTANDO para lista PRINCIPAL {emojis['main']}")
        else:
            print(f"🔄 TROCA para lista {list_type.upper()} {emojis.get(list_type, '📁')}")
            
        # Log de estatísticas
        current_list = self.get_current_timing_list()
        if len(current_list) >= 2:
            current_avg = self.calculate_list_avg(current_list)
            print(f"   📊 Nova média: {current_avg:.3f}s ({len(current_list)} valores)")

    def update_timing_lists(self, current_time):
        """Atualiza listas com sistema de confirmação rápida"""
        if len(self.peak_timestamps) < 2:
            return
            
        current_interval = current_time - self.peak_timestamps[-2]
        
        # Atualiza lista principal (mantém últimos 10)
        self.main_timing_list = self.peak_timestamps[-10:] if len(self.peak_timestamps) >= 10 else self.peak_timestamps.copy()
        
        # VERIFICA TROCA ANTES de atualizar secundárias
        if self.should_switch_lists(current_interval):
            self.switch_to_list(self.pending_switch_list)
            # Limpa contador após troca bem-sucedida
            self.pattern_confirmation_count = 0
            self.pending_switch_list = None
        
        # Atualiza lista secundária (sempre)
        self.update_secondary_lists(current_time)


    def update_secondary_lists(self, current_time):
        """Atualiza as 6 listas secundárias com o novo timestamp"""
        if len(self.peak_timestamps) < 2:
            return
            
        current_interval = current_time - self.peak_timestamps[-2]
        
        # Encontra lista com menos elementos ou que melhor se encaixa
        target_list_name = None
        min_size = float('inf')
        
        for list_name, timing_list in self.secondary_timing_lists.items():
            list_size = len(timing_list)
            
            # Prefere listas não cheias
            if list_size < 5:
                if list_size < min_size:
                    min_size = list_size
                    target_list_name = list_name
            else:
                # Se todas estão cheias, encontra a com média mais diferente
                list_avg = self.calculate_list_avg(timing_list)
                current_diff = abs(current_interval - list_avg)
                if target_list_name is None or current_diff > best_diff:
                    best_diff = current_diff
                    target_list_name = list_name
        
        # Adiciona à lista escolhida
        if target_list_name:
            target_list = self.secondary_timing_lists[target_list_name]
            if len(target_list) >= 5:
                target_list.pop(0)  # Remove o mais antigo se cheia
            target_list.append(current_time)
            
            print(f"📊 Adicionado à lista {target_list_name.upper()} - Intervalo: {current_interval:.3f}s")
            self.print_list_status()

    def print_list_status(self):
        """Mostra status de todas as 7 listas"""
        status = []
        main_size = len(self.main_timing_list)
        main_avg = self.calculate_list_avg(self.main_timing_list) if main_size >= 2 else 0
        status.append(f"🏠:{main_size}vals({main_avg:.3f}s)")
        
        for i in range(1, 7):
            list_name = f"sec_{i}"
            timing_list = self.secondary_timing_lists[list_name]
            list_size = len(timing_list)
            list_avg = self.calculate_list_avg(timing_list) if list_size >= 2 else 0
            status.append(f"📁{i}:{list_size}vals({list_avg:.3f}s)")
        
        print("   " + " | ".join(status))

    def detect_emerging_pattern(self, current_interval):
        """Detecta rapidamente quando um novo padrão está surgindo"""
        if self.current_list_type != "main":
            return None
            
        main_avg = self.calculate_list_avg(self.main_timing_list)
        
        # Se não tem dados suficientes, ignora
        if main_avg <= 0 or len(self.main_timing_list) < 3:
            return None
            
        main_diff = abs(current_interval - main_avg) / main_avg
        
        # Se muito diferente da lista atual (>50%)
        if main_diff > 0.5:
            # Procura lista secundária compatível
            best_match = self.find_best_matching_list(current_interval)
            if best_match:
                target_list = self.secondary_timing_lists[best_match]
                if len(target_list) >= 2:  # Pelo menos 2 valores para ter média
                    list_avg = self.calculate_list_avg(target_list)
                    list_diff = abs(current_interval - list_avg) / list_avg
                    
                    # Se está dentro de 20% da lista secundária
                    if list_diff <= 0.20:
                        print(f"🎯 Padrão compatível: {current_interval:.3f}s ~ {list_avg:.3f}s (diff: {list_diff:.2%})")
                        return best_match
                        
        return None

    def get_current_timing_list(self):
        """Retorna a lista de timing atualmente ativa entre as 7 opções"""
        if self.current_list_type == "main":
            return self.main_timing_list
        else:
            return self.secondary_timing_lists.get(self.current_list_type, self.main_timing_list)
        def update_sensitivity_level(self, value):
            """Atualiza o nível de sensibilidade e reseta o contador"""
            self.sensitivity_level = int(value)
            self.change_request_counter = 0  # Reseta contador ao mudar nível
            
            # Atualiza descrição do nível (expandido para 5 níveis)
            descriptions = {
                1: "Nível 1: Máxima responsividade (todas as chamadas)",
                2: "Nível 2: Ignora 1 de cada 2 chamadas (50% redução)",
                3: "Nível 3: Ignora 2 de cada 3 chamadas (66% redução)",
            }
            self.sensitivity_description.config(text=descriptions[self.sensitivity_level])
            print(f"📊 Nível de sensibilidade alterado para: {self.sensitivity_level} - {descriptions[self.sensitivity_level]}")

    def initialize_default_timing(self):
        """Inicializa o histórico de timing com intervalos padrão de 1 segundo"""
        current_time = time.time()
        # Cria 3 timestamps históricos com intervalo de 1 segundo
        for i in range(3):
            timestamp = current_time - (3 - i) * self.default_interval
            self.peak_timestamps.append(timestamp)
            self.main_timing_list.append(timestamp)
        
    def reset_timing_history(self):
        """Reseta e reinicializa o histórico de timing das 7 listas"""
        self.bass_levels.clear()
        self.peak_timestamps.clear()
        self.main_timing_list.clear()
        
        # Limpa todas as 6 listas secundárias
        for list_name in self.secondary_timing_lists:
            self.secondary_timing_lists[list_name].clear()
        
        self.current_list_type = "main"
        self.change_request_counter = 0
        self.initialize_default_timing()
        print("🔄 Histórico completamente resetado - 7 listas reinicializadas")
        
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
                    print("📊 Áudio significativo detectado novamente - Flag de limpeza resetada")
            
            self.volume_window.append(raw_volume)

            if len(self.volume_window) > self.window_size:
                self.volume_window.pop(0)

            dominant_freq = self.analyze_frequencies(data)
            self.recent_frequencies.append(dominant_freq)
            if len(self.recent_frequencies) > self.window_size:
                self.recent_frequencies.pop(0)

            # Analyze bass level for this audio chunk
            bass_level = self.analyze_bass_level(data)

            # Atualiza visualização apenas a cada X amostras (decimação)
            self.visualization_counter += 1
            if self.visualization_counter >= self.visualization_decimation:
                self.visualization_counter = 0  # Reset counter
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
                    # Reset completo e reinicialização
                    self.reset_timing_history()
                    self.bass_cleaned = True
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
            
            # Define range de frequências graves (20-110 Hz)
            bass_mask = (np.abs(freqs) >= 10) & (np.abs(freqs) <= 110)
            
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
                    
                    # Se já temos algum nível de grave registrado, valida contra a média
                    if len(self.bass_levels) > 0:
                        # Calculate average bass level from previous valid peaks
                        avg_bass_level = np.mean(self.bass_levels)
                        
                        # Peak is valid if bass level is sufficient
                        if current_bass_level < (avg_bass_level * 0.5):  # Bass too low
                            is_valid_peak = False
                            print(f"❌ Falso pico detectado - Grave muito baixo: {current_bass_level:.4f} (média: {avg_bass_level:.4f})")
                    else:
                        # Se não temos histórico de graves, aceita qualquer grave > threshold mínimo
                        min_bass_threshold = 0.001  # Threshold mínimo para considerar grave válido
                        if current_bass_level < min_bass_threshold:
                            is_valid_peak = False
                            print(f"❌ Primeiro pico rejeitado - Grave insuficiente: {current_bass_level:.4f} < {min_bass_threshold:.4f}")
                        else:
                            print(f"✅ Primeiro pico aceito - Estabelecendo baseline de grave: {current_bass_level:.4f}")
                    
                    # *** MODIFICAÇÃO: Usa lista atual para cálculos de timing ***
                    current_timing_list = self.get_current_timing_list()
                    
                    # Check peak timing if we have previous peaks
                    if is_valid_peak and len(current_timing_list) > 0:
                        time_since_last_peak = current_time - current_timing_list[-1]
                        
                        # Calculate average interval between peaks da lista atual
                        if len(current_timing_list) >= 2:
                            intervals = []
                            for i in range(1, min(len(current_timing_list), self.peak_interval_window_size + 1)):
                                interval = current_timing_list[-i] - current_timing_list[-i-1]
                                intervals.append(interval)
                            
                            avg_interval = np.mean(intervals)
                            
                            # Ignora médias acima de 1 segundo
                            if avg_interval > 1.0:
                                print(f"⚠️ Média de ({avg_interval:.3f}s),")
                            else:
                                # Apply timing buffer - expect peak 100ms earlier than average
                                expected_time = avg_interval - self.timing_buffer
                                
                                # If peak is coming too early (before expected time), require higher threshold
                                if time_since_last_peak < expected_time:
                                    # Calcula threshold dinâmico que diminui conforme se aproxima do tempo esperado
                                    # Começa em 1.50 no tempo 0 e vai diminuindo até 1.00 no tempo esperado
                                    
                                    # Proporção de quanto já passou do tempo esperado (0.0 = início, 1.0 = tempo esperado)
                                    time_ratio = time_since_last_peak / expected_time if expected_time > 0 else 0
                                    
                                    # Interpola entre 1.50 (início) e 1.00 (fim)
                                    dynamic_threshold = 1.50 - (0.50 * time_ratio)
                                    
                                    required_threshold = peak_threshold * dynamic_threshold
                                    
                                    if current_volume < required_threshold:
                                        is_valid_peak = False
                                        print(f"❌ Pico muito cedo - Requer {dynamic_threshold:.2f}x ({dynamic_threshold:.0%}) acima do normal.")
                                        print(f"   Intervalo atual: {time_since_last_peak:.3f}s ({time_ratio:.1%} do esperado)")
                                        print(f"   Tempo esperado: {expected_time:.3f}s (média: {avg_interval:.3f}s - buffer: {self.timing_buffer:.3f}s)")
                                        print(f"   Volume atual: {current_volume:.4f}, Necessário: {required_threshold:.4f}")
                                    else:
                                        print(f"✅ Pico cedo mas forte o suficiente: {current_volume:.4f} ≥ {required_threshold:.4f} (threshold: {dynamic_threshold:.2f}x)")
                                else:
                                    print(f"⏰ Pico no timing esperado: {time_since_last_peak:.3f}s ≥ {expected_time:.3f}s")
                    
                    if is_valid_peak:
                        # Record bass level and timestamp for this valid peak
                        self.bass_levels.append(current_bass_level)
                        if len(self.bass_levels) > self.bass_window_size:
                            self.bass_levels.pop(0)
                        
                        # *** MODIFICAÇÃO: Atualiza sistema principal E gerencia listas ***
                        self.peak_timestamps.append(current_time)
                        if len(self.peak_timestamps) > self.peak_interval_window_size:
                            self.peak_timestamps.pop(0)
                        
                        # *** NOVO: Atualiza sistema de múltiplas listas ***
                        self.update_timing_lists(current_time)
                        
                        # Calcula média atual de graves (sempre disponível após o primeiro pico)
                        avg_bass = np.mean(self.bass_levels)
                        
                        # Verifica se deve trocar imagem ou aplicar enquadramento baseado no nível de sensibilidade
                        action_result = self.should_change_image()
                        
                        # *** MODIFICAÇÃO: Usa lista atual para display ***
                        current_list = self.get_current_timing_list()
                        if len(current_list) >= 2:
                            last_interval = current_list[-1] - current_list[-2]
                            # Calculate current average for display
                            intervals = []
                            for i in range(1, min(len(current_list), self.peak_interval_window_size + 1)):
                                interval = current_list[-i] - current_list[-i-1]
                                intervals.append(interval)
                            current_avg_interval = np.mean(intervals)
                            
                            # Define emoji baseado na ação
                            if action_result["action"] == "change_image":
                                status_emoji = "🔄"
                            elif action_result["action"] == "apply_frame":
                                status_emoji = "🖼️"
                            else:  # ignore
                                status_emoji = "🔇"
                            
                            list_emoji = "🏠" if self.current_list_type == "main" else "📁"
                            print(f"{status_emoji} {list_emoji} Pico válido - Freq: {avg_frequency:.2f} Hz | Volume: {current_volume:.4f} | Grave: {current_bass_level:.4f} (média: {avg_bass:.4f})")
                            print(f"   Intervalo: {last_interval:.3f}s | Média: {current_avg_interval:.3f}s | Lista: {self.current_list_type} | Nível: {self.sensitivity_level}")
                        else:
                            # Define emoji baseado na ação
                            if action_result["action"] == "change_image":
                                status_emoji = "🔄"
                            elif action_result["action"] == "apply_frame":
                                status_emoji = "🖼️"
                            else:  # ignore
                                status_emoji = "🔇"
                            
                            list_emoji = "🏠" if self.current_list_type == "main" else "📁"
                            print(f"{status_emoji} {list_emoji} Pico válido - Freq: {avg_frequency:.2f} Hz | Volume: {current_volume:.4f} | Grave: {current_bass_level:.4f} (média: {avg_bass:.4f}) | Lista: {self.current_list_type} | Nível: {self.sensitivity_level}")
                        
                        # Executa a ação determinada
                        if action_result["action"] == "change_image":
                            for player in self.players:
                                if player.running:  
                                    player.next_image()
                        elif action_result["action"] == "apply_frame":
                            for player in self.players:
                                if player.running:
                                    player.apply_frame_to_current_image(action_result["frame_type"])
                        # Se action for "ignore", não faz nada
                                
                        self.waiting_for_valley = True
                        self.hysteresis_start_time = current_time

            elif current_volume < valley_threshold:
                self.waiting_for_valley = False
        except Exception as e:
            print(f"Erro na detecção: {e}")

    def should_change_image(self):
        """Determina se deve trocar a imagem ou aplicar enquadramento baseado no nível de sensibilidade"""
        self.change_request_counter += 1
        
        # Se enquadramento dinâmico não estiver ativado, usa lógica original
        if not self.dynamic_framing_enabled.get():
            return self._original_should_change_logic()
        
        # Nova lógica com enquadramento dinâmico
        if self.sensitivity_level == 1:
            # Nível 1: Sempre troca (sem enquadramentos)
            return {"action": "change_image", "frame_type": None}
        
        elif self.sensitivity_level == 2:
            # Nível 2: 1º normal, 2º enquadramento, 3º troca
            position_in_cycle = (self.change_request_counter - 1) % 2 + 1
            
            if position_in_cycle == 1:
                # Primeira chamada do ciclo - mostra normal (mas só troca se for realmente a primeira imagem)
                if self.change_request_counter == 1:
                    return {"action": "change_image", "frame_type": None}
                else:
                    return {"action": "change_image", "frame_type": None}
            else:
                # Segunda chamada - aplica enquadramento
                return {"action": "apply_frame", "frame_type": "center"}
                
        elif self.sensitivity_level == 3:
            # Nível 3: 1º normal, 2º enquadramento, 3º enquadramento, 4º troca
            position_in_cycle = (self.change_request_counter - 1) % 3 + 1
            
            if position_in_cycle == 1:
                return {"action": "change_image", "frame_type": None}
            elif position_in_cycle == 2:
                return {"action": "apply_frame", "frame_type": "center"}
            else:  # position_in_cycle == 3
                return {"action": "apply_frame", "frame_type": "random"}
                
        elif self.sensitivity_level == 4:
            # Nível 4: 1º normal, 2º-4º enquadramentos, 5º troca
            position_in_cycle = (self.change_request_counter - 1) % 4 + 1
            
            if position_in_cycle == 1:
                return {"action": "change_image", "frame_type": None}
            elif position_in_cycle == 2:
                return {"action": "apply_frame", "frame_type": "top_left"}
            elif position_in_cycle == 3:
                return {"action": "apply_frame", "frame_type": "center"}
            else:  # position_in_cycle == 4
                return {"action": "apply_frame", "frame_type": "bottom_right"}
                
        elif self.sensitivity_level == 5:
            # Nível 5: 1º normal, 2º-5º enquadramentos diferentes, 6º troca
            position_in_cycle = (self.change_request_counter - 1) % 5 + 1
            
            if position_in_cycle == 1:
                return {"action": "change_image", "frame_type": None}
            elif position_in_cycle == 2:
                return {"action": "apply_frame", "frame_type": "top_left"}
            elif position_in_cycle == 3:
                return {"action": "apply_frame", "frame_type": "center"}
            elif position_in_cycle == 4:
                return {"action": "apply_frame", "frame_type": "bottom_right"}
            else:  # position_in_cycle == 5
                return {"action": "apply_frame", "frame_type": "random"}
        
        return {"action": "change_image", "frame_type": None}  # Fallback

    def _original_should_change_logic(self):
        """Lógica original de troca de imagem (para quando enquadramento dinâmico está desabilitado)"""
        if self.sensitivity_level == 1:
            return {"action": "change_image", "frame_type": None}
        elif self.sensitivity_level == 2:
            if self.change_request_counter % 2 == 1:
                return {"action": "change_image", "frame_type": None}
            else:
                print("🔇 Troca ignorada (Nível 2)")
                return {"action": "ignore", "frame_type": None}
        elif self.sensitivity_level == 3:
            if self.change_request_counter % 3 == 1:
                return {"action": "change_image", "frame_type": None}
            else:
                print("🔇 Troca ignorada (Nível 3)")
                return {"action": "ignore", "frame_type": None}
        elif self.sensitivity_level == 4:
            if self.change_request_counter % 4 == 1:
                return {"action": "change_image", "frame_type": None}
            else:
                print("🔇 Troca ignorada (Nível 4)")
                return {"action": "ignore", "frame_type": None}
        elif self.sensitivity_level == 5:
            if self.change_request_counter % 5 == 1:
                return {"action": "change_image", "frame_type": None}
            else:
                print("🔇 Troca ignorada (Nível 5)")
                return {"action": "ignore", "frame_type": None}
        
        return {"action": "change_image", "frame_type": None}  # Fallback
                
    def update_visualization(self, data):
        """Atualiza a visualização da forma de onda"""
        try:
            if hasattr(self, 'line') and len(data) > 0:
                # Calcula fator de decimação baseado na diferença de samplerate
                # Original: 44100, Desejado para gráfico: 10040
                decimation_factor = int(44100 / 30000)  # ≈ 4.4, arredonda para 4
                
                # Aplica decimação nos dados (pega 1 a cada 4 amostras)
                decimated_data = data[::decimation_factor]
                
                # Normaliza os dados decimados para melhor visualização
                normalized_data = decimated_data / (np.max(np.abs(decimated_data)) + 1e-10)
                
                # Ajusta o tamanho para caber no gráfico
                target_size = self.blocksize // decimation_factor  # ≈ 256 pontos
                
                if len(normalized_data) > target_size:
                    normalized_data = normalized_data[:target_size]
                elif len(normalized_data) < target_size:
                    # Pad with zeros if needed
                    padded_data = np.zeros(target_size)
                    padded_data[:len(normalized_data)] = normalized_data
                    normalized_data = padded_data
                
                # Atualiza o gráfico com dados decimados
                self.line.set_ydata(normalized_data)
                self.line.set_xdata(np.arange(len(normalized_data)))
                
                # Ajusta escala X do gráfico para os novos dados
                self.ax.set_xlim(0, len(normalized_data))
                
                # Force canvas update
                self.canvas_graph.draw_idle()
                
        except Exception as e:
            print(f"Erro na visualização: {e}")

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
