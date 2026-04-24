import music21 as m21
import os
import numpy as np
import subprocess
import sys
from collections import defaultdict
from typing import List, Dict, Union, Optional

def createXMLNoteFile(
    notes: List[Dict[str, Union[int, float, str]]],
    tempo: float,
    filename: str = "output.musicxml",
    beats_per_bar: int = 4,
    open_in_musescore: bool = True
) -> None:
    """
    Создаёт MusicXML файл на основе распознанных нот и темпа.
    """
    # Создаём партитуру и партию
    score = m21.stream.Score()
    score.insert(0, m21.tempo.MetronomeMark(number=tempo))
    part = m21.stream.Part()
    part.partName = "Melody"
    part.partAbbreviation = "Mel."

    # Добавляем размер такта
    time_signature = m21.meter.TimeSignature(f"{beats_per_bar}/4")
    part.append(time_signature)

    # Добавляем темп (метроном) в начало партии
    # Убедимся, что tempo — число
    if isinstance(tempo, (list, np.ndarray)):
        tempo = float(tempo[0])
    else:
        tempo = float(tempo)
    tempo_mark = m21.tempo.MetronomeMark(number=tempo, referent=m21.note.Note(type='quarter'))
    # part.append(tempo_mark)

    # Опционально: тональность (по умолчанию C major)
    key = m21.key.Key('F#')
    part.append(key)

    # Определяем диапазон тактов
    if not notes:
        print("Нет нот для записи.")
        return
    min_bar = min(note['bar'] for note in notes)
    max_bar = max(note['bar'] for note in notes)

    # Создаём все такты (даже пустые) для непрерывности
    for bar_num in range(min_bar, max_bar + 1):
        measure = m21.stream.Measure(number=bar_num)
        # Добавляем ноты, начинающиеся в этом такте
        notes_in_bar = [n for n in notes if n['bar'] == bar_num]

        # Группируем по позиции (в 1/32 долях)
        notes_by_pos = defaultdict(list)
        for n in notes_in_bar:
            pos = n['position']
            notes_by_pos[pos].append(n)

        # Сортируем позиции и вставляем ноты/аккорды
        for pos in sorted(notes_by_pos.keys()):
            offset_quarter = (pos - 1) / 8.0  # перевод 1/32 → четверти
            chord_notes = []
            for n in notes_by_pos[pos]:
                # Длительность в четвертях
                dur_quarter = n['duration'] / 8.0

                # Извлекаем MIDI (может быть set)
                midi_val = n['midi']
                if isinstance(midi_val, set):
                    midi_val = next(iter(midi_val)) if midi_val else 60

                nt = m21.note.Note(midi=midi_val)
                nt.quarterLength = dur_quarter
                chord_notes.append(nt)

            if len(chord_notes) == 1:
                measure.insert(offset_quarter, chord_notes[0])
            elif len(chord_notes) > 1:
                chord = m21.chord.Chord(chord_notes)
                measure.insert(offset_quarter, chord)

        part.append(measure)

    # Добавляем партию в партитуру
    score.append(part)

    # Запись в файл
    score.show()
    print(f"MusicXML файл сохранён как {filename}")

    # Открыть в MuseScore, если требуется
    if open_in_musescore:
        _open_with_musescore(filename)


def _open_with_musescore(filepath: str) -> None:
    """Пытается открыть файл в MuseScore в зависимости от ОС."""
    try:
        if sys.platform.startswith('win'):
            # Windows
            os.startfile(filepath)
        elif sys.platform.startswith('darwin'):
            # macOS
            subprocess.run(['open', filepath])
        elif sys.platform.startswith('linux'):
            # Linux (предполагаем, что MuseScore установлен и доступен как mscore)
            subprocess.run(['mscore', filepath])
        else:
            print("Не удалось определить команду для открытия MuseScore.")
    except Exception as e:
        print(f"Не удалось открыть файл в MuseScore: {e}")