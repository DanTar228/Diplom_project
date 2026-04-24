import os
import sys
import subprocess
import librosa
import numpy as np
from basic_pitch.inference import predict
from music21 import stream, note, chord, instrument, tempo, metadata

# --- НАСТРОЙКИ ---
QUANTIZATION_GRID = 0.5  # 0.25 = 1/16 нота. 0.5 = 1/8 нота.

def separate_tracks(input_file, output_dir):
    """Разделение аудио на дорожки с помощью Demucs"""
    print(f"--- Шаг 1: Разделение дорожек (Demucs) ---")
    # Используем вызов модуля через текущий интерпретатор
    cmd = [sys.executable, "-m", "demucs.separate", "-o", output_dir, input_file]
    subprocess.run(cmd, check=True)
    
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    return os.path.join(output_dir, "htdemucs", base_name)

def transcribe_track(audio_path):
    """Транскрипция аудио в список нот с помощью Basic Pitch"""
    print(f"--- Шаг 2: Транскрипция {os.path.basename(audio_path)} ---")
    

    # Настройки для фильтрации "мусора"
    onset_threshold = 0.6  # Выше = меньше лишних "вскриков" в начале ноты
    frame_threshold = 0.8  # Выше = модель должна быть более уверена, чтобы держать ноту
    minimum_duration = 0.1 # Игнорировать ноты короче 100мс (убирает артефакты)
    # Извлекаем данные. note_events — это список кортежей
    model_output, midi_data, note_events = predict(
    audio_path,
    onset_threshold=onset_threshold,
    frame_threshold=frame_threshold,
    minimum_note_length=minimum_duration,
    minimum_frequency=30, # Для баса можно оставить низко, для гитары поднять до 80
    )
    print("-"*40+f'\n{model_output}\n{midi_data}\n'+"-"*40)
    extracted_notes = []
    for event in note_events:
        # Распаковываем кортеж (в нем 5 элементов)
        # start_time, end_time, pitch, amplitude, pitch_bends
        start_time = event[0]
        end_time = event[1]
        pitch = event[2]
        amplitude = event[3]

        extracted_notes.append({
            'pitch': int(pitch),           # Номер MIDI ноты
            'start': float(start_time),    # Время в секундах
            'end': float(end_time),        # Время в секундах
            'velocity': int(amplitude * 127) # Громкость (0-127)
        })
    
    print(f"Извлечено нот: {len(extracted_notes)}")
    return extracted_notes

class ScoreBuilder:
    def __init__(self, bpm):
        self.bpm = bpm
        self.quarter_duration = 60.0 / bpm
        self.score = stream.Score()
        self.score.insert(0, metadata.Metadata(title="Automated Transcription"))
        self.score.insert(0, tempo.MetronomeMark(number=bpm))

    def quantize(self, seconds):
        """Привязка секунд к сетке долей (quarterLength)"""
        raw_quarters = seconds / self.quarter_duration
        return round(raw_quarters / QUANTIZATION_GRID) * QUANTIZATION_GRID

    def add_part(self, note_data, instrument_obj, name):
        """Создание дорожки в партитуре"""
        part = stream.Part()
        part.id = name
        part.insert(0, instrument_obj)
        
        # Сортируем ноты по времени начала
        note_data.sort(key=lambda x: x['start'])
        
        # Группировка в аккорды (если ноты начинаются одновременно)
        chord_buckets = {}
        for n in note_data:
            q_start = self.quantize(n['start'])
            if q_start not in chord_buckets:
                chord_buckets[q_start] = []
            chord_buckets[q_start].append(n)

        for q_start, notes in chord_buckets.items():
            if len(notes) > 1:
                # Создаем аккорд
                pitches = [nt['pitch'] for nt in notes]
                m21_obj = chord.Chord(pitches)
            else:
                # Одиночная нота
                m21_obj = note.Note(notes[0]['pitch'])
            
            # Длительность (берем по первой ноте в группе)
            q_end = self.quantize(notes[0]['end'])
            duration_val = max(q_end - q_start, QUANTIZATION_GRID)
            m21_obj.duration.quarterLength = duration_val
            m21_obj.volume.velocity = notes[0]['velocity']
            
            part.insert(q_start, m21_obj)

        # "Причесываем" нотную запись
        part.makeMeasures(inPlace=True)
        part.makeRests(inPlace=True)
        self.score.insert(0, part)

def main(input_audio):
    output_dir = "output_data"
    
    # 1. Анализ темпа
    print(f"--- Анализ темпа ---")
    y, sr = librosa.load(input_audio)
    bpm, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(bpm)
    print(f"Определен темп: {bpm:.2f} BPM")

    # 2. Разделение на стэмы
    stems_folder = separate_tracks(input_audio, output_dir)
    
    # 3. Инициализация сборщика нот
    builder = ScoreBuilder(bpm)

    # 4. Обработка дорожек
    # Basic Pitch лучше всего работает с гитарой (other.wav) и басом (bass.wav)
    mapping = [
        ('bass.wav', instrument.ElectricBass(), "Bass"),
        ('other.wav', instrument.Guitar(), "Guitar")
    ]

    for filename, instr, name in mapping:
        path = os.path.join(stems_folder, filename)
        if os.path.exists(path):
            note_data = transcribe_track(path)
            if note_data:
                builder.add_part(note_data, instr, name)

    # 5. Экспорт
    output_xml = "transcription.musicxml"
    builder.score.write('musicxml', fp=output_xml)
    print(f"--- Успешно! Файл сохранен: {output_xml} ---")
    
    # 6. Открытие в MuseScore
    builder.score.show()

if __name__ == "__main__":
    # Укажите путь к вашему файлу
    FILE_PATH = "media_data_set/Собачий вальс — На пианино (www.lightaudio.ru).mp3"
    
    if os.path.exists(FILE_PATH):
        main(FILE_PATH)
    else:
        print(f"Файл {FILE_PATH} не найден!")