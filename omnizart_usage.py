import os
import sys
import subprocess
import librosa
import pretty_midi
from omnizart.music import app as music_app
from omnizart.drum import app as drum_app
from music21 import stream, note, chord, instrument, tempo, metadata, converter

# --- НАСТРОЙКИ ---
QUANTIZATION_GRID = 0.5  # Привязка к 1/16 ноте

def separate_tracks(input_file, output_dir):
    """Шаг 1: Разделение аудио на дорожки с помощью Demucs"""
    print(f"--- Шаг 1: Разделение дорожек (Demucs) ---")
    # Используем текущий интерпретатор для вызова demucs
    cmd = [sys.executable, "-m", "demucs.separate", "-o", output_dir, input_file]
    subprocess.run(cmd, check=True)
    
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    return os.path.join(output_dir, "htdemucs", base_name)

def transcribe_with_omnizart(audio_path, mode="music"):
    """Шаг 2: Транскрипция аудио в MIDI с помощью Omnizart"""
    print(f"--- Шаг 2: Транскрипция {os.path.basename(audio_path)} ({mode}) ---")
    
    # Путь для временного MIDI файла
    output_midi = audio_path.replace(".wav", ".mid")
    
    if mode == "music":
        # Используем ансамблевую модель (подходит для гитары, баса, клавиш)
        music_app.transcribe(audio_path, output=output_midi)
    elif mode == "drum":
        try:
            # Специализированная модель для ударных
            drum_app.transcribe(audio_path, output=output_midi)
        except:
            pass

    return output_midi

class ScoreBuilder:
    def __init__(self, bpm):
        self.bpm = bpm
        self.quarter_duration = 60.0 / bpm
        self.score = stream.Score()
        self.score.insert(0, metadata.Metadata(title="Omnizart Transcription"))
        self.score.insert(0, tempo.MetronomeMark(number=bpm))

    def midi_to_m21_part(self, midi_path, instrument_obj, part_name):
        """Конвертация MIDI от Omnizart в дорожку music21 с квантованием"""
        print(f"--- Шаг 3: Конвертация {part_name} в ноты ---")
        
        # Загружаем MIDI
        pm = pretty_midi.PrettyMIDI(midi_path)
        part = stream.Part()
        part.id = part_name
        part.insert(0, instrument_obj)

        for pm_instr in pm.instruments:
            # Группируем ноты по времени начала для создания аккордов
            chord_dict = {}
            for pm_note in pm_instr.notes:
                # Квантуем время начала (переводим секунды в четверти)
                start_quarter = round((pm_note.start / self.quarter_duration) / QUANTIZATION_GRID) * QUANTIZATION_GRID
                end_quarter = round((pm_note.end / self.quarter_duration) / QUANTIZATION_GRID) * QUANTIZATION_GRID
                duration_quarters = max(end_quarter - start_quarter, QUANTIZATION_GRID)

                if start_quarter not in chord_dict:
                    chord_dict[start_quarter] = []
                chord_dict[start_quarter].append((pm_note.pitch, duration_quarters, pm_note.velocity))

            # Добавляем ноты/аккорды в поток
            for q_start in sorted(chord_dict.keys()):
                notes_data = chord_dict[q_start]
                if len(notes_data) > 1:
                    # Создаем аккорд
                    pitches = [n[0] for n in notes_data]
                    m21_obj = chord.Chord(pitches)
                else:
                    # Одиночная нота
                    m21_obj = note.Note(notes_data[0][0])
                
                m21_obj.duration.quarterLength = notes_data[0][1]
                m21_obj.volume.velocity = notes_data[0][2]
                part.insert(q_start, m21_obj)

        part.makeMeasures(inPlace=True)
        part.makeRests(inPlace=True)
        return part

def main(input_audio):
    output_dir = "output_data"
    
    # 0. Анализ BPM оригинального файла
    y, sr = librosa.load(input_audio)
    bpm, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(bpm)
    print(f"Обнаружен темп: {bpm:.2f} BPM")

    # 1. Разделение Demucs
    stems_folder = separate_tracks(input_audio, output_dir)
    
    builder = ScoreBuilder(bpm)

    # 2. Список дорожек для обработки
    # (Файл стэма, Инструмент music21, Имя, Режим Omnizart)
    tasks = [
        ('bass.wav', instrument.ElectricBass(), "Bass", "music"),
        ('other.wav', instrument.Guitar(), "Guitar", "music"),
        ('drums.wav', instrument.Percussion(), "Drums", "drum")
    ]

    for stem_name, instr, p_name, mode in tasks:
        stem_path = os.path.join(stems_folder, stem_name)
        if os.path.exists(stem_path):
            try:
                # Транскрипция в MIDI
                midi_res = transcribe_with_omnizart(stem_path, mode=mode)
                # MIDI в Music21
                m21_part = builder.midi_to_m21_part(midi_res, instr, p_name)
                builder.score.insert(0, m21_part)
            except:
                pass

    # 3. Сохранение и показ
    output_xml = "omnizart_output.musicxml"
    builder.score.write('musicxml', fp=output_xml)
    print(f"\n--- ГОТОВО! ---")
    print(f"Результат сохранен в: {output_xml}")
    
    # Открываем в MuseScore
    builder.score.show()

if __name__ == "__main__":
    # Укажите ваш файл
    PATH_TO_AUDIO = "media_data_set/Собачий вальс — На пианино (www.lightaudio.ru).mp3"
    
    if os.path.exists(PATH_TO_AUDIO):
        main(PATH_TO_AUDIO)
    else:
        print("Файл не найден.")