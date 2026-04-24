import torch

# Сохраняем оригинальную функцию
_orig_to = torch.Tensor.to

def _patched_to(self, *args, **kwargs):
    # Если в аргументах (позиционных или именованных) затесался torch.device там, где его не ждут
    new_args = list(args)
    for i, arg in enumerate(new_args):
        if isinstance(arg, torch.device):
            # Переносим из позиционного аргумента в именованный 'device'
            kwargs['device'] = new_args.pop(i)
            break
            
    if 'dtype' in kwargs and isinstance(kwargs['dtype'], (torch.device, str)):
        # Если пришла строка "cpu" или объект device в поле dtype
        val = kwargs.pop('dtype')
        if val == "cpu" or val == "cuda" or isinstance(val, torch.device):
            kwargs['device'] = val
            
    return _orig_to(self, *new_args, **kwargs)

# Заменяем метод во всей системе
torch.Tensor.to = _patched_to

import os
import warnings

# Пытаемся заставить torchaudio использовать старый надежный бэкэнд
#os.environ["TORCHAUDIO_USE_BACKEND_DISPATCHER"] = "1"

# Если у вас установлен ffmpeg в конкретную папку, можно указать путь прямо в коде:
os.add_dll_directory(r"D:\ffmpeg-8.1-full_build-shared\bin")

import os
import torch
import subprocess
import numpy as np
import librosa
from mt3_infer import transcribe
from music21 import stream, note, chord, instrument, tempo, metadata, duration

# --- БЛОК 1: РАЗДЕЛЕНИЕ ДОРОЖЕК (DEMUCS) ---
def separate_tracks(input_file, output_dir):
    print(f"--- Разделение дорожек для {input_file} ---")
    # Запуск demucs через системный вызов (самый надежный способ)
    subprocess.run(["demucs", "-o", output_dir, input_file], check=True)
    # По умолчанию Demucs создает папку с именем модели (например, htdemucs)
    # и внутри папку с именем файла.
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    return os.path.join(output_dir, "htdemucs", base_name)

# --- БЛОК 2: АНАЛИЗ ТЕМПА (LIBROSA) ---
def get_audio_info(audio_path:str)->float:
    y, sr = librosa.load(audio_path)
    bpm, _ = librosa.beat.beat_track(y=y, sr=sr)
    return float(bpm[0])

def convert_midi_to_list(midi_obj):
    notes_list = []
    # Если midi_obj — это PrettyMIDI
    for instrument in midi_obj.instruments:
        for note in instrument.notes:
            notes_list.append({
                "pitch": note.pitch,
                "start": note.start,
                "end": note.end,
                "velocity": note.velocity,
                "instr": instrument.program # MIDI ID инструмента
            })
    return notes_list

# --- БЛОК 3: ТРАНСКРИПЦИЯ (MT3 MOCK / LOGIC) ---
def transcribe_audio_to_notes(audio_path):
    print(f"--- Транскрипция {audio_path} ---")
    
    # ИСПРАВЛЕНИЕ: Загружаем аудио сразу с частотой 16000 Гц
    # MT3 критически важна именно эта частота
    audio, sr = librosa.load(audio_path, sr=16000) 
    
    print(f"Аудио загружено: частота {sr} Гц, форма {audio.shape}")

    try:
        # Теперь передаем аудио с правильным sr
        midi = transcribe(
            audio=audio, 
            sr=sr, 
            model="yourmt3", 
            device='cuda'  # Мы договорились использовать CPU для стабильности
        )
        
        # Дальнейшая логика обработки midi...
        # (Обычно midi объект содержит список нот или NoteSequence)
        return convert_midi_to_list(midi) 
        
    except Exception as e:
        print(f"Ошибка при транскрипции: {e}")
        return []

# --- БЛОК 4: КВАНТОВАНИЕ И СБОРКА (MUSIC21) ---
class Quantizer:
    def __init__(self, bpm, grid_step=0.25): # 0.25 = 1/16 нота
        self.bpm = bpm
        self.quarter_duration = 60.0 / bpm
        self.grid_step = grid_step

    def to_quarters(self, seconds):
        """Переводит секунды в длительности music21 (четверти)"""
        raw_quarters = seconds / self.quarter_duration
        # Квантование: притягиваем к сетке
        return round(raw_quarters / self.grid_step) * self.grid_step

def build_score(transcribed_data, bpm):
    score = stream.Score()
    score.insert(0, metadata.Metadata(title="Transcribed Score"))
    score.insert(0, tempo.MetronomeMark(number=bpm))
    
    quantizer = Quantizer(bpm, grid_step=0.25) # Сетка 1/16
    
    # Словарь для разделения по инструментам
    parts = {}

    for n_data in transcribed_data:
        instr_id = n_data['instr']
        if instr_id not in parts:
            new_part = stream.Part()
            # Назначаем инструмент (упрощенно)
            if instr_id >= 32 and instr_id <= 39:
                new_part.insert(0, instrument.BassGuitar())
            else:
                new_part.insert(0, instrument.Guitar())
            parts[instr_id] = new_part

        # Квантуем время начала и длительность
        q_start = quantizer.to_quarters(n_data['start'])
        q_end = quantizer.to_quarters(n_data['end'])
        q_len = max(q_end - q_start, 0.25) # Минимальная длина - 1/16

        m21_note = note.Note(n_data['pitch'])
        m21_note.duration.quarterLength = q_len
        m21_note.volume.velocity = n_data['velocity']

        # Вставляем в партию
        parts[instr_id].insert(q_start, m21_note)

    # Добавляем все партии в партитуру и чистим ритм
    for p in parts.values():
        p.makeMeasures(inPlace=True)
        p.makeRests(inPlace=True)
        score.insert(0, p)
        
    return score

# --- БЛОК 5: ГЛАВНЫЙ ПРОЦЕСС ---
def main(input_audio):
    output_dir = "output_data"
    
    # 1. Определяем общий темп
    bpm = get_audio_info(input_audio)
    print(f"Определен темп: {bpm} BPM")

    # 2. Разделяем на дорожки
    stems_folder = separate_tracks(input_audio, output_dir)
    
    all_notes = []
    # 3. Обрабатываем каждую дорожку (например, 'other.wav' и 'bass.wav')
    target_stems = ['other.wav', 'bass.wav', 'drums.wav']
    
    for stem in target_stems:
        stem_path = os.path.join(stems_folder, stem)
        if os.path.exists(stem_path):
            notes = transcribe_audio_to_notes(stem_path)
            all_notes.extend(notes)

    # 4. Собираем MusicXML
    final_score = build_score(all_notes, bpm)
    
    # 5. Сохраняем и открываем
    xml_path = "final_transcription.musicxml"
    final_score.write('musicxml', fp=xml_path)
    print(f"Готово! Файл сохранен как {xml_path}")
    
    # Открываем в MuseScore
    final_score.show()

if __name__ == "__main__":
    main("./media_data_set/Spring-Flowers(chosic.com).mp3")