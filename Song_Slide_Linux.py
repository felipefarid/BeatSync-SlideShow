#!/usr/bin/env python3
# Song_Slide - Linux Adaptado (sem pythoncom, com suporte a ALSA/PulseAudio)
# FIX CRÍTICO: Uso de update_idletasks() para garantir dimensões do canvas antes do carregamento da primeira imagem.

import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import os
import numpy as np
import threading
import time
import soundcard as sc
import random
import warnings
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys
import queue
import subprocess

# -------------------------------------------------------------------
# Configurações de Áudio e UI
# -------------------------------------------------------------------
SAMPLERATE = 48000
BLOCKSIZE = 1536
AUDIO_QUEUE_MAX = 8          
RESULT_QUEUE_MAX = 64         
PROCESSOR_SLEEP_ON_EMPTY = 0.005 
CAPTURE_REOPEN_BACKOFF = 0.5  
MAX_CAPTURE_REOPEN = 5        
VIZ_UPDATE_MS = 25 

# TAMANHO DO BUFFER DE PRÉ-CARREGAMENTO
PRELOAD_BUFFER_SIZE = 5 

# Faixas reais de processamento de áudio (em Hz)
FREQ_BANDS = [
    (20, 100), (100, 150), (150, 250), (250, 500), (500, 1000),
    (1000, 2000), (2002, 4000), (4000, 8000), (8000, 12000), (12000, 20000)
]

# -------------------------
# Thread-safe queues/state
# -------------------------
audio_queue = queue.Queue(maxsize=AUDIO_QUEUE_MAX)
beat_detection_queue = queue.Queue(maxsize=RESULT_QUEUE_MAX)
shared_lock = threading.Lock()
shared_state = {
    "bar_heights": [0.0] * len(FREQ_BANDS),
    "peak_heights": [0.0] * len(FREQ_BANDS),
    "vu_updated": False,
    "last_fft_time": None,
}

warnings.filterwarnings("ignore", category=UserWarning)
try:
    from soundcard import SoundcardRuntimeWarning 
    warnings.filterwarnings("ignore", category=SoundcardRuntimeWarning) 
except Exception:
    pass

# -------------------------
# Utilities
# -------------------------
def safe_put(q: queue.Queue, item):
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            _ = q.get_nowait()
            q.put_nowait(item)
        except Exception:
            pass

def clear_queue(q: queue.Queue):
    """Esvazia a fila de forma segura."""
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            break

class TextRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget
    def write(self, message):
        try:
            self.text_widget.after(0, self._insert_text, str(message))
        except Exception:
            pass
    def _insert_text(self, message):
        try:
            self.text_widget.insert(tk.END, message)
            self.text_widget.see(tk.END)
            lines = int(self.text_widget.index('end-1c').split('.')[0])
            if lines > 1000:
                self.text_widget.delete('1.0', '500.0')
        except Exception:
            pass
    def flush(self):
        pass

# -------------------------
# ThreadCapture - LINUX VERSION
# -------------------------
class ThreadCapture(threading.Thread):
    def __init__(self, samplerate=SAMPLERATE, blocksize=BLOCKSIZE, stop_event=None):
        super().__init__(daemon=True)
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.stop_event = stop_event or threading.Event()

    def run(self):
        print("[Soundcard] Capturando SOMENTE áudio do sistema (output/playback)")
        
        try:
            # IMPORTANTE: include_loopback=True é para pegar áudio de SAÍDA
            # No Linux, loopback = áudio que está saindo pelas caixas
            all_speakers = sc.all_speakers()
            print(f"[Soundcard] Saídas de áudio disponíveis ({len(all_speakers)}):")
            for i, spk in enumerate(all_speakers):
                print(f"  [{i}] {spk.name}")
            
            if not all_speakers:
                print("[Soundcard] ❌ Nenhuma saída de áudio encontrada!")
                return
            
            # Usar o speaker padrão (o que está tocando música)
            speaker = sc.default_speaker()
            print(f"[Soundcard] ✅ Usando saída: {speaker.name}")
            
            # AGORA O IMPORTANTE: Pegar o MICROFONE que corresponde a esta saída (loopback)
            # Isso captura o áudio que ESTÁ SENDO TOCADO, não o que está entrando pelo mic
            
            reopen_attempts = 0
            mic = None
            
            while not self.stop_event.is_set() and mic is None:
                try:
                    # Método 1: Tentar pelo nome do speaker
                    try:
                        mic = sc.get_microphone(speaker.name, include_loopback=True)
                        print(f"[Soundcard] ✅ Encontrado loopback para: {speaker.name}")
                    except:
                        pass
                    
                    # Método 2: Procurar manualmente
                    if mic is None:
                        all_mics = sc.all_microphones(include_loopback=True)
                        for m in all_mics:
                            if speaker.name in m.name or "loopback" in m.name.lower() or "monitor" in m.name.lower():
                                mic = m
                                print(f"[Soundcard] ✅ Loopback encontrado: {m.name}")
                                break
                    
                    # Método 3: Usar qualquer loopback
                    if mic is None:
                        all_mics = sc.all_microphones(include_loopback=True)
                        if all_mics:
                            mic = all_mics[0]
                            print(f"[Soundcard] ⚠️ Usando qualquer loopback: {mic.name}")
                    
                    if mic is None:
                        print("[Soundcard] ❌ Não encontrou loopback. O áudio NÃO será capturado!")
                        print("[Soundcard] ℹ️ Configure um dispositivo de loopback no seu sistema")
                        time.sleep(5)
                        continue
                        
                except Exception as e:
                    reopen_attempts += 1
                    print(f"[Soundcard] Erro: {e}")
                    if reopen_attempts > 3:
                        print("[Soundcard] ❌ Falha crítica. Verifique configurações de áudio.")
                        return
                    time.sleep(2)
            
            # AGORA CAPTURA
            while not self.stop_event.is_set():
                try:
                    with mic.recorder(samplerate=self.samplerate, blocksize=self.blocksize) as rec:
                        print(f"[Soundcard] 🎵 CAPTURANDO ÁUDIO DO SISTEMA (sem microfone)")
                        
                        while not self.stop_event.is_set():
                            try:
                                data = rec.record(numframes=self.blocksize)
                                if data is not None and len(data) > 0:
                                    if data.ndim > 1:
                                        data = data.mean(axis=1)
                                    safe_put(audio_queue, np.array(data, copy=True))
                            except Exception as e:
                                print(f"[Soundcard] Erro na captura: {e}")
                                break
                                
                except Exception as e:
                    print(f"[Soundcard] Erro no recorder: {e}")
                    time.sleep(1)
                    
        except Exception as e:
            print(f"[Soundcard] ❌ ERRO FATAL: {e}")
            import traceback
            traceback.print_exc()

# -------------------------
# ThreadProcessor - MESMA LÓGICA (sem alterações)
# -------------------------
class ThreadProcessor(threading.Thread):
    def __init__(self, samplerate=SAMPLERATE, blocksize=BLOCKSIZE, freq_bands=FREQ_BANDS, stop_event=None):
        super().__init__(daemon=True)
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.freq_bands = freq_bands
        self.stop_event = stop_event or threading.Event()
        
        # Configurações VU Meter
        self.decay_rate = 0.85 
        self.peak_decay_rate = 0.995 
        self.peak_hold_frames = 20 
        self.bar_heights = [0.0] * len(self.freq_bands)
        self.peak_heights = [0.0] * len(self.freq_bands)
        self.peak_hold_time = [0] * len(self.freq_bands)
        
        # --- LÓGICA AVANÇADA DE DETECÇÃO ---
        self.volume_window = [] 
        self.window_size = 10 
        self.volume_threshold_factor = 1.00 
        self.valley_threshold_factor = 0.95 
        self.recent_frequencies = []  
        
        # Histerese
        self.hysteresis = 0.2        
        self.hysteresis_multiplier = 1.5 
        self.waiting_for_valley = True
        self.hysteresis_start_time = 0
        
        self.bass_levels = []
        self.bass_window_size = 10
        
        self.peak_timestamps = [] 
        self.peak_interval_window_size = 5
        self.timing_buffer = 0.130
        
        self.default_interval = 0.5 
        self.dynamic_threshold_max = 1.90 
        self.non_bass_override_factor = 1.60 

        self.main_timing_list = [] 

        self.last_audio_time = time.time()
        self.silence_threshold = 0.5
        self.bass_cleaned = False
        self.silence_volume_threshold = 0.01

    def run(self):
        print("[Processor] Thread de processamento iniciada (Linux Adaptado)") 
        print("⏳ Aguardando 5 segundos para Player estabilizar...")
        time.sleep(5.0) 
        print("✅ Atraso inicial de 5 segundos concluído. Iniciando processamento de áudio.")
        frame_count = 0
        
        self.initialize_default_timing()
        self.reset_timing_history()
        
        while not self.stop_event.is_set():
            try:
                try:
                    raw = audio_queue.get(timeout=0.2) 
                    frame_count += 1
                except queue.Empty:
                    time.sleep(PROCESSOR_SLEEP_ON_EMPTY)
                    continue

                data = np.asarray(raw, dtype=np.float32) 
                if data.ndim > 1:
                    data = data.mean(axis=1)

                if data.size == 0:
                    continue

                # VU Meter (Visual)
                fft_result = np.fft.rfft(data) 
                freqs = np.fft.rfftfreq(len(data), d=1.0 / self.samplerate) 
                magnitude = np.abs(fft_result) 

                band_energies = []
                for (lo, hi) in self.freq_bands:
                    mask = (freqs >= lo) & (freqs < hi) 
                    if np.any(mask):
                        e = float(np.mean(magnitude[mask])) 
                    else:
                        e = 0.0 
                    band_energies.append(e)

                max_energy = max(band_energies) if band_energies else 1.0
                if max_energy <= 0: max_energy = 1.0
                new_heights = [min(1.0, e / max_energy) for e in band_energies] 

                for i in range(len(self.bar_heights)):
                    self.bar_heights[i] = max(new_heights[i], self.bar_heights[i] * self.decay_rate) 
                    if self.bar_heights[i] > self.peak_heights[i]:
                        self.peak_heights[i] = self.bar_heights[i] 
                        self.peak_hold_time[i] = self.peak_hold_frames 
                    else:
                        if self.peak_hold_time[i] > 0:
                            self.peak_hold_time[i] -= 1 
                        else:
                            self.peak_heights[i] *= self.peak_decay_rate 

                with shared_lock:
                    shared_state["bar_heights"] = list(self.bar_heights) 
                    shared_state["peak_heights"] = list(self.peak_heights) 
                    shared_state["vu_updated"] = True 
                    shared_state["last_fft_time"] = time.time() 

                # Processamento de Lógica
                self.process_beat_detection(data)
           
                # Verificação de silêncio
                current_time = time.time()
                if (current_time - self.last_audio_time) > self.silence_threshold and not self.bass_cleaned:
                    if len(self.bass_levels) > 0:
                        print(f"🧹 Silêncio detectado - Limpando buffer e histórico") 
                        
                        clear_queue(audio_queue) 
                        
                        self.reset_timing_history()
                        self.bass_cleaned = True

            except Exception as e:
                print(f"[Processor] error: {e}") 

    def process_beat_detection(self, audio_data):
        try:
            data = np.asarray(audio_data, dtype=np.float32)
            if data.ndim > 1: data = data.flatten()

            raw_volume = np.sqrt(np.mean(data**2))
            current_time = time.time()
         
            if raw_volume > self.silence_volume_threshold: 
                self.last_audio_time = current_time
                if self.bass_cleaned:
                    self.bass_cleaned = False
            
            self.volume_window.append(raw_volume) 
 
            if len(self.volume_window) > self.window_size: 
                self.volume_window.pop(0)

            bass_level = self.analyze_bass_level(data)
            self.recent_frequencies.append(self.analyze_frequencies(data))
            if len(self.recent_frequencies) > self.window_size:
                self.recent_frequencies.pop(0)

            if len(self.volume_window) == self.window_size and raw_volume > self.silence_volume_threshold: 
                self.dynamic_detection(bass_level, raw_volume, current_time)
            
        except Exception as e:
            print(f"Erro process_beat: {e}")

    def analyze_frequencies(self, audio_data):
        try:
            if len(audio_data) < 2: return 0.0
        
            fft_result = np.fft.fft(audio_data) 
            freqs = np.fft.fftfreq(len(fft_result), d=1/self.samplerate) 
            magnitude = np.abs(fft_result) 
            magnitude[0] = 0
            if len(magnitude) == 0: return 0.0
            dominant_freq = freqs[np.argmax(magnitude)] 
            return abs(dominant_freq)
       
        except: return 0.0 

    def analyze_bass_level(self, audio_data):
        return self._analyze_frequency_band(audio_data, 10, 110)

    def _analyze_frequency_band(self, audio_data, min_freq, max_freq):
        try:
            if len(audio_data) < 2: return 0.0
            
            fft_result = np.fft.fft(audio_data)
            freqs = np.fft.fftfreq(len(fft_result), d=1/self.samplerate)
      
            magnitude = np.abs(fft_result) 
            freq_mask = (np.abs(freqs) >= min_freq) & (np.abs(freqs) <= max_freq) 
            
            if np.any(freq_mask):
                band_energy = np.sum(magnitude[freq_mask]**2)
                num_bins = np.sum(freq_mask)
           
                if num_bins > 0: return band_energy / num_bins 
                return band_energy
            return 0.0
        except: return 0.0

    def calculate_list_avg(self, timing_list):
        if len(timing_list) < 2: return self.default_interval
        intervals = [timing_list[i] - timing_list[i-1] for i in range(1, len(timing_list))] 
        return np.mean(intervals) if intervals else self.default_interval

    def dynamic_detection(self, current_bass_level, current_volume, current_time):
        try:
            baseline = np.percentile(self.volume_window, 70)
            valley_base = np.percentile(self.volume_window, 30)
  
            peak_threshold = baseline * self.volume_threshold_factor 
            valley_threshold = valley_base * self.valley_threshold_factor 
            
            if self.waiting_for_valley and (current_time - self.hysteresis_start_time) < self.hysteresis:
                peak_threshold *= self.hysteresis_multiplier 

            if current_volume > peak_threshold:
                if not self.waiting_for_valley:
                    is_valid_peak = True
                    
                    if len(self.bass_levels) > 0: 
                        avg_bass_level = np.mean(self.bass_levels) 
                  
                        if current_bass_level < (avg_bass_level * 0.5): 
                            override_threshold = baseline * self.non_bass_override_factor 

                            if current_volume < override_threshold:
                                is_valid_peak = False 
                            
                    else:
                        if current_bass_level < 0.001:
                            is_valid_peak = False 

                    current_timing_list = self.main_timing_list 

                    if is_valid_peak and len(current_timing_list) > 0: 
                        time_since_last_peak = current_time - current_timing_list[-1]

                        if len(current_timing_list) >= 2:
                            intervals = [current_timing_list[i] - current_timing_list[i-1] 
                                         for i in range(1, len(current_timing_list))] 
                            avg_interval = np.mean(intervals)

                            if avg_interval <= 1.0:
                                expected_time = avg_interval - self.timing_buffer 
                                if time_since_last_peak < expected_time:
                                    time_ratio = time_since_last_peak / expected_time if expected_time > 0 else 0 
                                    time_ratio = np.clip(time_ratio, 0.0, 1.0) 
                                    
                                    dynamic_threshold = self.dynamic_threshold_max - (0.50 * time_ratio) 
                                    required_threshold = peak_threshold * dynamic_threshold

                                    if current_volume < required_threshold:
                                        is_valid_peak = False 

                    if is_valid_peak:
                        self.bass_levels.append(current_bass_level) 
                        if len(self.bass_levels) > self.bass_window_size: self.bass_levels.pop(0)

                        self.peak_timestamps.append(current_time) 
                        if len(self.peak_timestamps) > 20: self.peak_timestamps.pop(0)

                        self.update_timing_lists(current_time) 

                        avg_time_ms = self.calculate_list_avg(self.main_timing_list) * 1000
                        
                        # Dispara ação
                        safe_put(beat_detection_queue, {"action": "change_image"}) 
                        
                        avg_freq = np.mean(self.recent_frequencies) if self.recent_frequencies else 0
                        print(f"🎵 Batida! | Freq: {avg_freq:.0f}Hz | Média Intervalo: {avg_time_ms:.2f}ms")

                        self.waiting_for_valley = True
                        self.hysteresis_start_time = current_time

            elif current_volume < valley_threshold:
                self.waiting_for_valley = False
        except Exception as e: 
            print(f"Erro dynamic_detection: {e}")

    def update_timing_lists(self, current_time):
        if len(self.peak_timestamps) < 2: return
        self.main_timing_list = self.peak_timestamps[-10:] if len(self.peak_timestamps) >= 10 else self.peak_timestamps.copy() 
    
    def initialize_default_timing(self):
        current_time = time.time() 
        self.peak_timestamps = []
        self.main_timing_list = []
        for i in range(10):
            ts = current_time - (3 - i) * self.default_interval
            self.peak_timestamps.append(ts)
            self.main_timing_list.append(ts)
        
    def reset_timing_history(self):
        self.bass_levels.clear() 
        self.peak_timestamps.clear() 
        self.main_timing_list.clear() 
        self.initialize_default_timing() 


# -------------------------
# Player class - SEM ALTERAÇÕES
# -------------------------
class Player:
    def __init__(self, root, image_paths, random_order=True):
        self.root = root
        self.image_paths = image_paths
        self.random_order = random_order
        self.image_index = 0
        self.running = True
        
        # ESSENCIAL: Armazena a referência da imagem *atualmente* exibida.
        self.tk_image = None
        
        # Estado e Lock para gerenciamento do buffer de pré-carregamento
        self.image_load_lock = threading.Lock()
        # {índice: (tk_image, display_width, display_height)}
        self.preloaded_images = {} 
        # Índices que estão atualmente sendo carregados por threads
        self.loading_indices = set() 

        self.viewer_window = tk.Toplevel(self.root)
        self.viewer_window.title("Player")
        self.viewer_window.geometry("800x600")

        self.canvas = tk.Canvas(self.viewer_window, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.viewer_window.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Inicia o primeiro carregamento após a UI estar pronta
        self.root.after_idle(self.initial_load)

    def _preload_image_in_background(self, index, path, cw, ch):
        """Função pesada: Carrega e redimensiona a imagem PIL (executada em thread)."""
        try:
            # 1. Carregamento e redimensionamento (Thread-safe)
            current_image = Image.open(path)
            display_img = self.resize_image(current_image, cw, ch)
            
            # 2. Agenda a atualização do buffer na thread principal
            self.root.after_idle(lambda: self._store_preloaded_pil_data(display_img, display_img.width, display_img.height, index))
        except Exception as e:
            print(f"Error pre-loading image {index}: {e}")
            with self.image_load_lock:
                if index in self.loading_indices:
                    self.loading_indices.remove(index)
            self.root.after_idle(self.refill_preload_buffer)

    def _store_preloaded_pil_data(self, pil_image, dw, dh, index_loaded):
        """Armazena o objeto PIL Image (executado na thread principal), convertendo para PhotoImage."""
        try:
            # CRIAÇÃO DO ImageTk.PhotoImage na thread principal (segurança máxima)
            tk_image = ImageTk.PhotoImage(pil_image) 
            
            with self.image_load_lock:
                # Armazena o objeto pronto para exibição (tk_image)
                self.preloaded_images[index_loaded] = (tk_image, dw, dh)
                if index_loaded in self.loading_indices:
                    self.loading_indices.remove(index_loaded)
            
        except Exception as e:
            print(f"Erro ao criar PhotoImage para índice {index_loaded}: {e}")
            with self.image_load_lock:
                if index_loaded in self.loading_indices:
                    self.loading_indices.remove(index_loaded)
        
        self.root.after_idle(self.refill_preload_buffer) 
        
    def _update_canvas_display(self, tk_image, dw, dh, cw, ch):
        """Atualiza o canvas com uma imagem TK já pronta (SÓ NA THREAD PRINCIPAL)."""
        try:
            # CRÍTICO: Armazena a referência no atributo de instância, garantindo a persistência.
            self.tk_image = tk_image 
            
            self.canvas.delete("all")
            x = (cw - dw) // 2
            y = (ch - dh) // 2
            self.canvas.create_image(x, y, anchor=tk.NW, image=self.tk_image)
            
            # Não chama refill_preload_buffer aqui se for a primeira imagem, pois _display_initial_pil_image fará isso.
            # Chama apenas se for uma troca via next_image (buffer).
            if self.image_index != 0 or self.image_index in self.preloaded_images:
                 self.refill_preload_buffer() 

        except Exception as e:
            print(f"Error updating canvas: {e}")
            
    def _display_initial_pil_image(self, pil_image, dw, dh, cw, ch):
        """Exibe a primeira imagem (PIL) na thread principal e inicia o pre-load."""
        try:
            # 1. CRÍTICO: Cria o PhotoImage e armazena a referência *imediatamente*.
            self.tk_image = ImageTk.PhotoImage(pil_image)

            # 2. Desenha no canvas
            self.canvas.delete("all")
            x = (cw - dw) // 2
            y = (ch - dh) // 2
            self.canvas.create_image(x, y, anchor=tk.NW, image=self.tk_image)
            
        except Exception as e:
            print(f"Error displaying initial image: {e}")
        finally:
            # 3. Limpa o flag de carregamento e inicia o buffer
            with self.image_load_lock:
                if self.image_index in self.loading_indices:
                    self.loading_indices.remove(self.image_index) 
            self.refill_preload_buffer() 

    def initial_load(self, attempt=1): 
        """Carrega a primeira imagem (índice 0) de forma assíncrona."""
        if not self.image_paths or attempt > 20:
            print("Error: Can't load images or canvas dimensions not available.")
            return
        
        # 🚨 CORREÇÃO CRÍTICA: Força o Tkinter a calcular a geometria da janela Toplevel
        self.canvas.update_idletasks() 
        
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
            
        if cw <= 1 or ch <= 1: # Verifica se a dimensão é mínima (1x1)
            if self.running:
                # Polling: Tenta novamente em 100ms
                self.root.after(100, lambda: self.initial_load(attempt + 1))
            return

        path = self.image_paths[self.image_index]
        
        with self.image_load_lock:
            if self.image_index in self.loading_indices: return
            self.loading_indices.add(self.image_index) 
        
        def initial_load_in_background():
            try:
                # HEAVY LIFTING (Background Thread)
                current_image = Image.open(path)
                display_img = self.resize_image(current_image, cw, ch)
                
                # SCHEDULE UPDATE TO MAIN THREAD, passing the thread-safe PIL Image object
                self.root.after_idle(lambda: self._display_initial_pil_image(display_img, display_img.width, display_img.height, cw, ch))
            except Exception as e:
                print(f"Error loading initial image in thread: {e}")
            
        threading.Thread(target=initial_load_in_background, daemon=True).start()


    def refill_preload_buffer(self):
        """Preenche o buffer de pré-carregamento com as próximas N imagens."""
        if not self.image_paths: return

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw <= 1 or ch <= 1: return

        with self.image_load_lock:
            current_index = self.image_index
            
            for i in range(1, PRELOAD_BUFFER_SIZE + 1):
                target_index = (current_index + i) % len(self.image_paths)
                
                if target_index not in self.preloaded_images and target_index not in self.loading_indices:
                    self.loading_indices.add(target_index)
                    path = self.image_paths[target_index]
                    
                    threading.Thread(
                        target=self._preload_image_in_background, 
                        args=(target_index, path, cw, ch), 
                        daemon=True
                    ).start()

    def resize_image(self, img, canvas_w, canvas_h):
        canvas_w = canvas_w if canvas_w > 1 else 800
        canvas_h = canvas_h if canvas_h > 1 else 600
        img_ratio = img.width / img.height
        canvas_ratio = canvas_w / canvas_h
  
        if img_ratio > canvas_ratio: 
            nw = canvas_w
            nh = max(int(canvas_w / img_ratio), 1)
        else:
            nh = canvas_h
            nw = max(int(canvas_h * img_ratio), 1)
        return img.resize((nw, nh), Image.LANCZOS)

    def next_image(self):
        """Troca a imagem usando a versão pré-carregada, se disponível."""
        if not self.image_paths: 
            print("Error: No img on folder.")
            return

        with self.image_load_lock:
            next_index = (self.image_index + 1) % len(self.image_paths)
            
            if next_index in self.preloaded_images:
                # Pega a referência persistente do buffer (já é um ImageTk.PhotoImage)
                tk_image, dw, dh = self.preloaded_images.pop(next_index)
                
                self.image_index = next_index 
                
                cw = self.canvas.winfo_width()
                ch = self.canvas.winfo_height()

                # Usa a função de atualização segura
                self.root.after_idle(lambda: self._update_canvas_display(tk_image, dw, dh, cw, ch))
                
            else:
                # Fallback
                print("⚠️ Fallback: Próxima imagem não estava no buffer. Carregamento normal iniciado.")
                self.image_index = next_index
                self.initial_load() 

    def on_close(self):
            print("🧹 Limpeza de Player iniciada.")
            self.running = False
            
            # 🟢 ALTERAÇÃO 3A: Limpa o buffer de pré-carregamento
            with self.image_load_lock:
                self.preloaded_images.clear() 
                self.loading_indices.clear()
                
            try:
                self.viewer_window.destroy()
            except Exception:
                pass 
            print("✅ Player encerrado e buffer limpo.")

# -------------------------
# Main UI class - COM DETECÇÃO DE SISTEMA
# -------------------------
class ImageViewer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Control Panel - Linux")
        self.root.geometry("420x680")
        self.root.protocol("WM_DELETE_WINDOW", self.exit_viewer)

        # UI / App state
        self.players = []
        self.include_subfolders = tk.BooleanVar(value=True)
        self.random_order = tk.BooleanVar(value=True) 
        self.image_formats = ["jpg", "jpeg", "png", "bmp", "gif"]
        self.selected_folder = None

        # audio thread control
        self.stop_event = threading.Event()
        
        # Build UI FIRST
        self.setup_initial_interface()
        
        sys.stdout = TextRedirector(self.log_text_widget)
        
        print(f"=== Song Slide - Linux Adaptado ===")
        print(f"Buffer de Pré-carregamento: {PRELOAD_BUFFER_SIZE}")
        print("Sistema: Linux")
        print("Inicializando...")

        # Verificar configurações de áudio
        self.check_audio_config()

        # Inicia threads
        self.capture_thread = ThreadCapture(samplerate=SAMPLERATE, blocksize=BLOCKSIZE, stop_event=self.stop_event)
        self.processor_thread = ThreadProcessor(samplerate=SAMPLERATE, blocksize=BLOCKSIZE, freq_bands=FREQ_BANDS, stop_event=self.stop_event)
        
        self.capture_thread.start()
        self.processor_thread.start()
        print("Threads de áudio e processamento iniciadas")

        # schedule visualization updater
        self.after_id = self.root.after(VIZ_UPDATE_MS, self.update_visualization_from_shared)
        
        # schedule beat detection handler
        self.beat_after_id = self.root.after(50, self.handle_beat_detection)

        self.root.mainloop()

    def check_audio_config(self):
        """Verifica e configura áudio para Linux"""
        try:
            import soundcard as sc
            mics = sc.all_microphones(include_loopback=True)
            print(f"Dispositivos de áudio encontrados: {len(mics)}")
            for i, mic in enumerate(mics):
                print(f"  {i}: {mic.name}")
            
            # Verificar se tem pulseaudio
            try:
                result = subprocess.run(["which", "pulseaudio"], capture_output=True, text=True)
                if result.returncode == 0:
                    print("✅ PulseAudio está instalado")
                else:
                    print("⚠️ PulseAudio não encontrado, usando ALSA")
            except:
                pass
                
        except Exception as e:
            print(f"⚠️ Aviso de configuração de áudio: {e}")
            print("Instale: sudo apt install pulseaudio alsa-utils")

    def setup_initial_interface(self):
        frame = tk.Frame(self.root)
        frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
   
        # Título
        title_label = tk.Label(frame, text="Song Slide - Linux", font=("Arial", 14, "bold"))
        title_label.pack(pady=5)
        
        # Botão para selecionar pasta
        btn_frame = tk.Frame(frame)
        btn_frame.pack(pady=6, fill=tk.X)
        
        self.select_folder_btn = tk.Button(
            btn_frame, 
            text="Selecionar Pasta de Imagens", 
            command=self.select_folder_and_load,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 10, "bold"),
            height=2
        )
        self.select_folder_btn.pack(fill=tk.X, pady=3)
        
        self.folder_label = tk.Label(frame, text="Nenhuma pasta selecionada", fg="gray", wraplength=380)
        self.folder_label.pack(pady=3)
        
        tk.Label(frame, text="Formatos de Imagem (separados por vírgula)").pack(pady=3)
        self.format_entry = tk.Entry(frame)
        self.format_entry.insert(0, ",".join(self.image_formats))
        self.format_entry.pack(pady=3, fill=tk.X)
        tk.Checkbutton(frame, text="Incluir Subpastas", variable=self.include_subfolders).pack(pady=3)
        tk.Checkbutton(frame, text="Ordem Aleatória", variable=self.random_order).pack(pady=3)
        
        self.setup_waveform_view(frame)

    def select_folder_and_load(self):
        self.select_folder_btn.config(state=tk.DISABLED, text="Abrindo diálogo...")
        print("🔍 Abrindo seletor de pasta...")
        threading.Thread(target=self._open_folder_dialog_thread, daemon=True).start()
    
    def _open_folder_dialog_thread(self):
        try:
            temp_root = tk.Tk()
            temp_root.withdraw()
            temp_root.attributes('-topmost', True)
            
            folder_path = filedialog.askdirectory(
                parent=temp_root,
                title="Selecione a pasta com as imagens"
            )
            
            temp_root.destroy()
            self.root.after(0, lambda: self._process_selected_folder(folder_path))
            
        except Exception as e:
            print(f"❌ Erro ao abrir diálogo: {e}")
            self.root.after(0, lambda: self.select_folder_btn.config(
                state=tk.NORMAL, 
                text="Selecionar Pasta de Imagens"
            ))
    
    def _process_selected_folder(self, folder_path):
        self.select_folder_btn.config(state=tk.NORMAL, text="Selecionar Pasta de Imagens")
        
        if not folder_path:
            print("❌ Nenhuma pasta selecionada.")
            return
        
        self.selected_folder = folder_path
        self.folder_label.config(text=f"Pasta: {folder_path}", fg="black")
        print(f"📂 Pasta selecionada: {folder_path}")
        
        image_formats = [fmt.strip().lower() for fmt in self.format_entry.get().split(",")]
        include_sub = self.include_subfolders.get()
        random_ord = self.random_order.get()
        
        print(f"⏳ Iniciando carregamento de imagens (Buffer: {PRELOAD_BUFFER_SIZE})...") 
        
        threading.Thread(
            target=self._load_files_in_background, 
            args=(folder_path, image_formats, include_sub, random_ord), 
            daemon=True
        ).start()

    def _load_files_in_background(self, folder_path, image_formats, include_subfolders, random_order): 
        try:
            image_paths = self.load_images(folder_path, image_formats, include_subfolders, random_order)
            if image_paths:
                self.root.after_idle(lambda: self._start_player_in_main_thread(image_paths, random_order))
        except Exception as e:
            self.root.after_idle(lambda: messagebox.showerror("Erro de Carregamento", f"Erro ao processar arquivos:\n{e}"))
            print(f"❌ Erro na thread de carregamento de arquivos: {e}") 

    def _start_player_in_main_thread(self, image_paths, random_order):
        try:
            player = Player(self.root, image_paths, random_order)
            self.players.append(player)
            print(f"✅ Player criado com {len(image_paths)} imagens")
        except Exception as e:
            print(f"❌ Erro ao iniciar Player na thread principal: {e}")
          
    def load_images(self, folder_path, image_formats, include_subfolders, random_order):
        image_paths = []
        try:
            if include_subfolders:
                for root, _, files in os.walk(folder_path):
                    for file in files:
                        if any(file.lower().endswith(ext) for ext in image_formats): 
                            image_paths.append(os.path.join(root, file))
            else:
                for file in os.listdir(folder_path):
                    if any(file.lower().endswith(ext) for ext in image_formats): 
                        image_paths.append(os.path.join(folder_path, file))

            if not image_paths:
                self.root.after_idle(lambda: messagebox.showerror("Erro", "Nenhuma imagem válida encontrada!"))
                return []

            if random_order:
                random.shuffle(image_paths) 

            print(f"📸 {len(image_paths)} imagens carregadas")
            return image_paths

        except Exception as e:
            print(f"❌ Erro ao carregar imagens: {e}")
            self.root.after_idle(lambda: messagebox.showerror("Erro de Carga", f"Erro ao carregar imagens:\n{e}"))
            return [] 

    def handle_beat_detection(self):
        try:
            while True:
                command = beat_detection_queue.get_nowait()
                if command["action"] == "change_image":
                    for player in self.players:
                        player.next_image() 
        except queue.Empty:
            pass
        
        if not self.stop_event.is_set():
            self.beat_after_id = self.root.after(50, self.handle_beat_detection)

    def setup_waveform_view(self, parent_frame):
        frame = tk.Frame(parent_frame)
        frame.pack(fill=tk.BOTH, expand=True)

        self.fig, self.ax = plt.subplots(figsize=(5, 2.2)) 
        self.fig.patch.set_facecolor('#1a1a1a')
        self.ax.set_facecolor('#0a0a0a')

        labels = ['20-100','100-150','150-250','250-500','500-1k','1k-2k','2k-4k','4k-8k','8k-12k','12k-20k']
        self.ax.set_xticks(np.arange(len(FREQ_BANDS)))
        self.ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7, color='#888888')
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        self.ax.set_ylim(0, 1.0)
        self.ax.set_xlim(-0.5, len(FREQ_BANDS)-0.5)
   
        self.ax.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.5) 

        init_heights = [0.0] * len(FREQ_BANDS)
        self.bar_rects = self.ax.bar(np.arange(len(FREQ_BANDS)), init_heights, width=0.7, color='lime', edgecolor='none')
        self.peak_lines = []
        for i in range(len(FREQ_BANDS)):
            line = self.ax.plot([i-0.35, i+0.35], [0, 0], color='red', linewidth=2, alpha=0.8)[0]
            self.peak_lines.append(line)

        self.canvas_graph = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas_graph.draw() 
        self.canvas_graph.get_tk_widget().pack(pady=6, fill=tk.BOTH, expand=True)

        self.log_text_widget = tk.Text(frame, height=10, width=40, bg="white", fg="black", font=("Comic Sans MS", 11))
        self.log_text_widget.pack(pady=3, fill=tk.BOTH, expand=True)

    def update_visualization_from_shared(self):
        try:
            updated = False
            with shared_lock:
                if shared_state.get("vu_updated"):
                    bh = shared_state.get("bar_heights", []) 
                    ph = shared_state.get("peak_heights", []) 
                    shared_state["vu_updated"] = False
                    updated = True
                else: 
                    bh = None
                    ph = None

            if updated and bh is not None:
                for i, rect in enumerate(self.bar_rects):
                    if i < len(bh): 
                        h = bh[i]
                        rect.set_height(h)
                        if h < 0.3: 
                            # Nível 1: Azul Gelo (muito baixo)
                            rect.set_color('#33B8A5') 
                        elif h < 0.5:
                            # Nível 2: Ciano Elétrico (baixo)
                            rect.set_color('#33A7D8') 
                        elif h < 0.7:
                            # Nível 3: Verde Ácido (médio)
                            rect.set_color('#3276B5') 
                        elif h < 0.9:
                            # Nível 4: Magenta (alto)
                            rect.set_color('#8869AD')
                        else:
                            # Nível 5: Vermelho Brilhante (pico)
                            rect.set_color('#B868AD')
                for i, pline in enumerate(self.peak_lines):
                    if i < len(ph):
                        phv = ph[i]
                        pline.set_ydata([phv, phv]) 
                self.canvas_graph.draw_idle()
        except Exception as e:
            print("update_visualization error:", e)
        if not self.stop_event.is_set():
            self.after_id = self.root.after(VIZ_UPDATE_MS, self.update_visualization_from_shared)

    def exit_viewer(self):
        print("🛑 Encerrando aplicação...")
        self.stop_event.set() 
        try:
            if hasattr(self, 'capture_thread') and self.capture_thread.is_alive():
                self.capture_thread.join(timeout=0.5)
            if hasattr(self, 'processor_thread') and self.processor_thread.is_alive():
                self.processor_thread.join(timeout=0.5)
        except Exception as e:
            print(f"Erro ao parar threads: {e}") 
        
        try:
            if hasattr(self, "after_id") and self.after_id is not None:
                self.root.after_cancel(self.after_id)
            if hasattr(self, "beat_after_id") and self.beat_after_id is not None:
                self.root.after_cancel(self.beat_after_id)
        except Exception:
            pass 
        
        try:
            self.root.quit()
            self.root.destroy()
        except Exception:
            pass

# -------------------------
# Entrypoint
# -------------------------
if __name__ == "__main__":
    try:
        ImageViewer()
    except Exception as e:
        print("Fatal error:", e)
