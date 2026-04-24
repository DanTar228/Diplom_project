import librosa
import numpy as np
from scipy.signal import find_peaks, medfilt
from typing import List, Dict, Union, Optional

def midi_note_name(midi_number: int) -> str:
    """Преобразует MIDI номер в название ноты (например, 60 -> C4)."""
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = midi_number // 12 - 1
    note = notes[midi_number % 12]
    return f"{note}{octave}"

def freq_to_midi(freq: float) -> float:
    """Преобразует частоту в MIDI номер (не округлённый)."""
    return 12 * np.log2(freq / 440.0) + 69

def midi_to_freq(midi: float) -> float:
    """Преобразует MIDI номер в частоту."""
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)

def estimate_note_from_peak(freq: float, tolerance_semitones: float = 0.8) -> Optional[int]:
    """
    Определяет MIDI номер ноты, если частота близка к стандартной частоте.
    tolerance_semitones — допустимое отклонение в полутонах.
    """
    midi_est = freq_to_midi(freq)
    midi_round = int(round(midi_est))
    freq_exact = midi_to_freq(midi_round)
    if abs(12 * np.log2(freq / freq_exact)) <= tolerance_semitones:
        return midi_round
    return None

def harmonic_check(freq: float, magnitudes, freqs, sr, window_length, num_harmonics=3, threshold_ratio=0.1):
    """
    Проверяет наличие гармоник у частоты freq.
    Возвращает True, если сумма магнитуд гармоник превышает порог относительно основной.
    """
    # Находим индекс основной частоты
    idx = np.argmin(np.abs(freqs - freq))
    if idx >= len(magnitudes):
        return False
    fundamental_mag = magnitudes[idx]
    if fundamental_mag == 0:
        return False
    harmonic_sum = 0
    for h in range(2, num_harmonics + 1):
        hfreq = freq * h
        if hfreq > sr / 2:
            break
        hidx = np.argmin(np.abs(freqs - hfreq))
        if hidx < len(magnitudes):
            harmonic_sum += magnitudes[hidx]
    return (harmonic_sum / fundamental_mag) > threshold_ratio

def analyze_notes(
    audio_file: str,
    threshold: float = 0.1,
    tolerance_semitones: float = 0.8,
    beats_per_bar: int = 4,
    verbose: bool = False,
    use_harmonic_check: bool = False,
    magnitude_percentile: float = 50.0,
    max_gap_segments: int = 1
) -> tuple:
    """
    Анализирует аудиофайл и возвращает список нотных событий с их длительностью,
    выраженной в количестве 1/32 долей (ритмических единиц).

    Параметры:
        audio_file (str): путь к аудиофайлу.
        threshold (float): порог обнаружения пика (доля от максимума в окне).
        tolerance_semitones (float): допуск при сопоставлении частоты с нотой (в полутонах).
        beats_per_bar (int): количество четвертей в такте (определяет размер такта).
        verbose (bool): если True, выводит отладочную информацию.
        use_harmonic_check (bool): использовать ли проверку гармоник для подтверждения ноты.
        magnitude_percentile (float): нижний перцентиль магнитуд событий для фильтрации.
        max_gap_segments (int): максимальный разрыв в 1/32 долях для объединения нот одной высоты.

    Возвращает:
        tuple: (events, tempo) где events — список словарей с ключами:
            'bar', 'position', 'midi', 'note_name', 'start_time', 'duration_32', 'max_magnitude'.
    """
    # 1. Загрузка аудио (моно)
    y, sr = librosa.load(audio_file, sr=None, mono=True)
    if verbose:
        print(f"Аудио загружено: {len(y)} сэмплов, частота дискретизации {sr} Гц")

    # 2. Определение темпа (BPM) с несколькими попытками
    # Сначала используем стандартный метод
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units='time')
    if isinstance(tempo, np.ndarray):
        tempo = tempo[0]
    # Проверяем разумность темпа, если нет — используем альтернативный метод
    if tempo < 40 or tempo > 200:
        # Альтернатива: автокорреляция onset strength
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, aggregate=np.median)[0]
    if verbose:
        print(f"Определён темп: {tempo:.2f} BPM")

    # 3. Длительность 1/32 ноты
    beat_duration = 60.0 / tempo
    thirty_second_duration = beat_duration / 8.0
    window_length = int(thirty_second_duration * sr)

    # 4. Определение смещения начала сетки по первому onset (если есть)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time', backtrack=True)
    if len(onsets) > 0:
        first_onset = onsets[0]
        # Смещение в сэмплах, чтобы окно начиналось с первого onset
        first_onset_sample = int(first_onset * sr)
        # Но окна должны идти с шагом window_length, поэтому подбираем начало так,
        # чтобы first_onset_sample был близок к началу какого-то окна.
        # Просто сдвинем начало на first_onset_sample, но тогда первое окно будет неполным,
        # если first_onset_sample не кратен window_length. Лучше начать с сэмпла,
        # кратного window_length, но ближайшего к first_onset_sample.
        # Чтобы не усложнять, начнём с first_onset_sample, но обрежем аудио до этого момента.
        # Однако это приведёт к потере начала, если первая нота не самая первая.
        # Вместо этого создадим равномерную сетку, начиная с первого onsets, но сегменты будут
        # смещены относительно начала файла. Это нормально, так как нас интересуют относительные позиции.
        start_sample = first_onset_sample
    else:
        start_sample = 0

    # Количество полных окон
    num_segments = (len(y) - start_sample) // window_length
    thirty_seconds_per_bar = beats_per_bar * 8

    # Подготовка окна (Ханна)
    window = np.hanning(window_length)

    segments_data = []  # каждый элемент: {'time': float, 'bar': int, 'position': int, 'notes': list}

    # Проходим по окнам
    for i in range(num_segments):
        seg_start = start_sample + i * window_length
        seg_end = seg_start + window_length
        if seg_end > len(y):
            break
        segment = y[seg_start:seg_end] * window

        # БПФ
        spectrum = np.fft.rfft(segment)
        magnitudes = np.abs(spectrum)
        freqs = np.fft.rfftfreq(window_length, d=1/sr)

        # Адаптивный порог: если максимум мал, используем threshold * max_mag, иначе threshold * max_mag
        # Можно также учитывать средний уровень шума
        max_mag = np.max(magnitudes) if len(magnitudes) > 0 else 0
        if max_mag == 0:
            continue
        # Используем фиксированный порог от максимума (можно модифицировать)
        peak_threshold = threshold * max_mag

        # Поиск пиков
        peaks, properties = find_peaks(magnitudes, distance=5, prominence=0.5*max_mag, height=peak_threshold)

        notes_in_segment = []
        for p in peaks:
            # Параболическая интерполяция для уточнения частоты
            if 1 <= p <= len(magnitudes) - 2:
                y1, y2, y3 = np.log(magnitudes[p-1:p+2])
                denom = 2 * (2*y2 - y1 - y3)
                if denom != 0:
                    shift = (y3 - y1) / denom
                    peak_bin = p + shift
                else:
                    peak_bin = p
            else:
                peak_bin = p

            peak_freq = peak_bin * sr / window_length
            # Проверка гармоник, если включено
            if use_harmonic_check and not harmonic_check(peak_freq, magnitudes, freqs, sr, window_length):
                continue

            midi_note = estimate_note_from_peak(peak_freq, tolerance_semitones)
            if midi_note is not None:
                notes_in_segment.append({
                    'midi': midi_note,
                    'magnitude': magnitudes[p]
                })

        # Удаляем дубликаты нот в одном сегменте (оставляем с максимальной магнитудой)
        unique_notes = {}
        for note in notes_in_segment:
            m = note['midi']
            if m not in unique_notes or note['magnitude'] > unique_notes[m]['magnitude']:
                unique_notes[m] = note
        notes_in_segment = list(unique_notes.values())

        # Сохраняем данные сегмента
        segments_data.append({
            'time': seg_start / sr,
            'bar': i // thirty_seconds_per_bar + 1,
            'position': i % thirty_seconds_per_bar + 1,
            'notes': notes_in_segment
        })

    if verbose:
        print(f"Обработано сегментов: {num_segments}")

    # 5. Группировка в нотные события с учётом стабильности
    events = []
    active_notes = {}  # key: midi, value: {'start_idx', 'max_mag', 'start_time', 'start_bar', 'start_pos', 'last_mag', 'stable'}

    # Сглаживание магнитуды: будем хранить последние несколько значений для каждой ноты
    # Для простоты применим медианный фильтр к магнитудам при обнаружении спада.
    # Вместо этого будем использовать порог на падение: если текущая магнитуда < 0.3 * max_mag, завершаем.
    # Но это может быть слишком грубо. Лучше использовать порог на основе отношения к максимуму за время ноты.

    # Для каждой ноты будем отслеживать максимум и если магнитуда падает ниже alpha * max_mag, завершаем.
    alpha = 0.1 # порог завершения ноты (30% от максимума)

    for idx, seg in enumerate(segments_data):
        current_notes = {note['midi']: note['magnitude'] for note in seg['notes']}

        # Проверяем активные ноты: если нет в текущем сегменте или магнитуда слишком мала, завершаем
        finished = []
        for midi, info in list(active_notes.items()):
            if midi not in current_notes:
                # Нота исчезла
                finished.append((midi, info))
            else:
                # Нота есть, проверяем падение
                mag = current_notes[midi]
                # # Обновляем максимум
                # if mag > info['max_mag']:
                #     info['max_mag'] = mag
                # Если магнитуда упала ниже alpha * max_mag, завершаем (кроме случая, когда это начало)
                if mag < alpha * info['max_mag'] and idx > info['start_idx']:
                    finished.append((midi, info))
                else:
                    # Продолжаем, обновляем last_mag
                    info['last_mag'] = mag

        # Завершаем отмеченные ноты
        for midi, info in finished:
            active_notes.pop(midi, None)
            duration_32 = idx - info['start_idx']
            # Добавляем событие, если длительность > 0
            if duration_32 > 0:
                events.append({
                    'bar': info['start_bar'],
                    'position': info['start_pos'],
                    'midi': midi,
                    'note_name': midi_note_name(midi),
                    'start_time': info['start_time'],
                    'duration_32': duration_32,
                    'max_magnitude': info['max_mag']
                })

        # Обрабатываем новые ноты
        for midi, mag in current_notes.items():
            if midi not in active_notes:
                # Новая нота
                active_notes[midi] = {
                    'start_idx': idx,
                    'start_time': seg['time'],
                    'start_bar': seg['bar'],
                    'start_pos': seg['position'],
                    'max_mag': mag,
                    'last_mag': mag
                }
            else:
                # Нота уже активна, обновим максимум (уже сделано выше)
                pass

    # Завершаем все оставшиеся ноты после последнего сегмента
    for midi, info in active_notes.items():
        duration_32 = num_segments - info['start_idx']
        if duration_32 > 0:
            events.append({
                'bar': info['start_bar'],
                'position': info['start_pos'],
                'midi': midi,
                'note_name': midi_note_name(midi),
                'start_time': info['start_time'],
                'duration_32': duration_32,
                'max_magnitude': info['max_mag']
            })

    # Сортируем по времени
    events.sort(key=lambda e: e['start_time'])

    # 6. Фильтрация слабых событий по перцентилю магнитуды
    if len(events) > 0:
        magnitudes = np.array([ev['max_magnitude'] for ev in events])
        threshold_mag = np.percentile(magnitudes, magnitude_percentile)
        events = [ev for ev in events if ev['max_magnitude'] >= threshold_mag]

    if verbose:
        print(f"После фильтрации по перцентилю {magnitude_percentile:.1f} осталось событий: {len(events)}")

    # 7. Объединение событий одной высоты с учётом пауз (по аналогии с исходным кодом, но с параметром max_gap_segments)
    # Добавляем глобальные индексы
    for ev in events:
        ev['global_start'] = (ev['bar'] - 1) * thirty_seconds_per_bar + (ev['position'] - 1)
        ev['global_end'] = ev['global_start'] + ev['duration_32'] - 1

    # Группировка по MIDI
    by_midi = {}
    for ev in events:
        by_midi.setdefault(ev['midi'], []).append(ev)

    merged_events = []
    for midi, ev_list in by_midi.items():
        ev_list.sort(key=lambda x: x['global_start'])
        i = 0
        while i < len(ev_list):
            current = ev_list[i].copy()
            j = i + 1
            while j < len(ev_list):
                next_ev = ev_list[j]
                # Разрыв между концом текущего и началом следующего (в 1/32 долях)
                gap = next_ev['global_start'] - current['global_end'] - 1
                if gap <= max_gap_segments:
                    # Объединяем
                    new_duration = (next_ev['global_start'] + next_ev['duration_32'] - current['global_start'])
                    current['duration_32'] = new_duration
                    current['max_magnitude'] = max(current['max_magnitude'], next_ev['max_magnitude'])
                    current['global_end'] = current['global_start'] + new_duration - 1
                    j += 1
                else:
                    break
            merged_events.append(current)
            i = j

    # Сортируем объединённые события по глобальному старту
    merged_events.sort(key=lambda x: x['global_start'])

    # Убираем временные поля global_start/end
    for ev in merged_events:
        del ev['global_start']
        del ev['global_end']

    if verbose:
        print(f"После объединения событий: {len(merged_events)}")

    return merged_events, tempo

# Пример использования
if __name__ == "__main__":
    # Предполагается, что функция createXMLNoteFile определена в модуле pdf
    from pdf import createXMLNoteFile
    import sys
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        events, tempo = analyze_notes(
            audio_path,
            threshold=0.1,
            tolerance_semitones=0.5,
            beats_per_bar=4,
            verbose=True,
            use_harmonic_check=True,
            magnitude_percentile=5,
            max_gap_segments=2
        )

        # Преобразуем в формат notes для createXMLNoteFile (как в исходном коде)
        notes = []
        for ev in events:
            notes.append({
                'bar': ev['bar'],
                'position': ev['position'],
                'midi': ev['midi'],  # исходно ожидается множество? но в createXMLNoteFile, вероятно, число
                'duration': ev['duration_32'],
                'note_name': ev['note_name'],
                'max_magnitude': ev['max_magnitude'],
                'start_time': ev['start_time']
            })

        # Дополнительная группировка одновременных нот (аккорды) с допуском по времени
        thirty_second_duration = 60.0 / tempo / 8.0
        time_tolerance = thirty_second_duration * 0.5
        notes.sort(key=lambda x: x['start_time'])
        i = 0
        while i < len(notes):
            j = i + 1
            while j < len(notes) and (notes[j]['start_time'] - notes[i]['start_time']) < time_tolerance:
                notes[j]['bar'] = notes[i]['bar']
                notes[j]['position'] = notes[i]['position']
                j += 1
            i = j

        print(f"Темп: {tempo:.2f} BPM")
        createXMLNoteFile(notes, tempo)
    else:
        print("Укажите путь к аудиофайлу как аргумент командной строки.")