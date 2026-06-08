"""
Arrangr.py — MP3 → SATB + Soloist A Cappella Arrangement (v6)
=============================================================
- Automatic key detection (major/minor)
- Solo = extracted melody (with rhythm)
- SATB follows voice‑leading rules (stepwise motion, contrary motion, no parallel 5ths/8ves)
- Texture changes by section (intro/verse/chorus/bridge/outro)
- Dynamic chord voicings (no fixed dictionary)
"""

import inspect
import json
import re
import warnings
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import librosa
import numpy as np
from music21 import clef, key, metadata, meter, note, stream, tempo
from scipy.ndimage import median_filter

warnings.filterwarnings('ignore')

_WHISPER_MODEL = None

# ----------------------------------------------------------------------
# Constants & helpers
# ----------------------------------------------------------------------
_NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
_BASS_ALT = ['doo', 'dum']

# Krumhansl‑Schmuckler key profiles (major, minor)
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                           2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                           2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Voice ranges (lowest comfortable, highest comfortable)
_RANGES = {
    'S': (60, 81),   # C4 – A5
    'A': (55, 74),   # G3 – D5
    'T': (48, 67),   # C3 – G4
    'B': (40, 55),   # E2 – D4
    'solo': (55, 84) # wider for soloist
}

# Section → texture style
_TEXTURE_STYLES = {
    'intro':  {'density': 'sparse',  'spacing': 'wide',   'rhythm': 'simple'},
    'verse':  {'density': 'normal',  'spacing': 'normal', 'rhythm': 'simple'},
    'chorus': {'density': 'full',    'spacing': 'tight',  'rhythm': 'busy'},
    'bridge': {'density': 'normal',  'spacing': 'wide',   'rhythm': 'medium'},
    'outro':  {'density': 'sparse',  'spacing': 'wide',   'rhythm': 'simple'}
}

# ----------------------------------------------------------------------
# Key detection
# ----------------------------------------------------------------------
def detect_key(chroma: np.ndarray) -> Tuple[str, int]:
    """
    Detect key from chroma (12 bins) using Krumhansl‑Schmuckler.
    Returns (mode, sharps) where mode is 'major' or 'minor'.
    """
    chroma_mean = np.mean(chroma, axis=1)
    chroma_mean = chroma_mean / (np.linalg.norm(chroma_mean) + 1e-9)

    best_key = None
    best_score = -np.inf
    best_mode = 'major'
    for root in range(12):
        # Major
        rolled = np.roll(chroma_mean, -root)
        score = np.dot(rolled, _MAJOR_PROFILE)
        if score > best_score:
            best_score = score
            best_key = root
            best_mode = 'major'
        # Minor
        rolled_min = np.roll(chroma_mean, -root)
        score_min = np.dot(rolled_min, _MINOR_PROFILE)
        if score_min > best_score:
            best_score = score_min
            best_key = root
            best_mode = 'minor'

    # Convert root to sharps (music21 key signature)
    from music21 import key
    try:
        best_key_int = int(best_key)
    except Exception:
        best_key_int = 0
    tonic = _NOTE_NAMES[best_key_int] if 0 <= best_key_int < len(_NOTE_NAMES) else 'C'
    try:
        if best_mode == 'major':
            k = key.Key(tonic)
        else:
            k = key.Key(tonic, 'minor')
    except Exception as e:
        print(f"[WARN] detect_key fallback to C major because music21 rejected tonic={tonic} mode={best_mode}: {e}")
        k = key.Key('C')
    return best_mode, k.sharps

# ----------------------------------------------------------------------
# Rhythm extraction from melody onsets
# ----------------------------------------------------------------------
def extract_melody_rhythm(y, sr, beat_times, melody_midi, beats_per_measure=4):
    """
    Compute note durations (in quarter lengths) for each measure based on onset detection.
    Returns list of lists: rhythm_patterns_per_measure.
    """
    hop = 512
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop, backtrack=True)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop)

    # For each measure, find onsets within the measure and compute gaps
    measures_rhythm = []
    for m_idx, beat_start in enumerate(beat_times[:-1:beats_per_measure]):
        measure_end = beat_times[min((m_idx+1)*beats_per_measure, len(beat_times)-1)]
        # onsets inside this measure
        ons = onset_times[(onset_times >= beat_start) & (onset_times < measure_end)]
        if len(ons) == 0:
            # no onsets → whole note
            measures_rhythm.append([4.0])
        else:
            # build durations from consecutive onsets, plus final to measure end
            durations = []
            prev = beat_start
            for t in ons:
                dur = t - prev
                if dur > 0.05:
                    # quantize to nearest 0.25 (sixteenth)
                    qdur = round(dur / 0.25) * 0.25
                    if qdur > 0:
                        durations.append(qdur)
                prev = t
            last_dur = measure_end - prev
            if last_dur > 0.05:
                qlast = round(last_dur / 0.25) * 0.25
                if qlast > 0:
                    durations.append(qlast)
            # sum to 4.0 if necessary
            total = sum(durations)
            if abs(total - 4.0) > 0.1 and len(durations) > 0:
                durations[-1] += (4.0 - total)
            measures_rhythm.append(durations if durations else [4.0])
    # if fewer measures than melody, pad
    while len(measures_rhythm) < len(melody_midi):
        measures_rhythm.append([4.0])
    return measures_rhythm

# ----------------------------------------------------------------------
# Voice‑leading SATB generator (rules‑based)
# ----------------------------------------------------------------------
class SATBVoicer:
    def __init__(self, key_mode, key_sharps):
        self.key_mode = key_mode
        self.key_sharps = key_sharps
        self.prev_voices = {'S': 72, 'A': 67, 'T': 60, 'B': 48}  # start in C major
        self.prev_chord_tones = None

    def _chord_tones(self, root_pc, quality):
        """Return list of pitch classes (0‑11) for the chord."""
        if quality == 'maj':
            return [root_pc, (root_pc+4)%12, (root_pc+7)%12]
        elif quality == 'min':
            return [root_pc, (root_pc+3)%12, (root_pc+7)%12]
        elif quality == 'dim':
            return [root_pc, (root_pc+3)%12, (root_pc+6)%12]
        elif quality == 'aug':
            return [root_pc, (root_pc+4)%12, (root_pc+8)%12]
        elif quality == 'dom7':
            return [root_pc, (root_pc+4)%12, (root_pc+7)%12, (root_pc+10)%12]
        elif quality == 'maj7':
            return [root_pc, (root_pc+4)%12, (root_pc+7)%12, (root_pc+11)%12]
        elif quality == 'min7':
            return [root_pc, (root_pc+3)%12, (root_pc+7)%12, (root_pc+10)%12]
        else:
            return [root_pc, (root_pc+4)%12, (root_pc+7)%12]

    def _closest_pitch(self, pc, target, low, high, avoid=None):
        candidates = []
        for oct in range(-2, 9):
            p = pc + 12*oct
            if low <= p <= high:
                if avoid is not None and abs(p - avoid) <= 1:
                    continue
                candidates.append(p)
        if not candidates:
            return None
        return min(candidates, key=lambda x: abs(x - target))

    def voice_chord(self, chord_info, solo_midi, section_style, prev_voices=None):
        """
        Returns dict with SATB midi numbers, respecting voice‑leading rules.
        """
        if prev_voices:
            self.prev_voices = prev_voices.copy()
        root_pc = chord_info['root_pc']
        quality = chord_info['quality']
        tones = self._chord_tones(root_pc, quality)

        # Decide spacing based on section density
        density = _TEXTURE_STYLES.get(section_style, {}).get('density', 'normal')
        spacing = _TEXTURE_STYLES.get(section_style, {}).get('spacing', 'normal')

        if density == 'sparse':
            # wider spacing, maybe omit 5th
            target_spacing = {'S': solo_midi + 8, 'A': solo_midi, 'T': solo_midi - 8, 'B': solo_midi - 16}
        elif density == 'full':
            target_spacing = {'S': solo_midi + 4, 'A': solo_midi, 'T': solo_midi - 6, 'B': solo_midi - 14}
        else:  # normal
            target_spacing = {'S': solo_midi + 5, 'A': solo_midi - 2, 'T': solo_midi - 9, 'B': solo_midi - 15}

        # Apply range limits
        for v in ['S','A','T','B']:
            low, high = _RANGES[v]
            target_spacing[v] = np.clip(target_spacing[v], low, high)

        # Assign notes using closest pitch within chord tones
        new_voices = {}
        for v, target in target_spacing.items():
            low, high = _RANGES[v]
            # prefer previous voice's pitch class if possible
            prev_pc = self.prev_voices[v] % 12
            if prev_pc in tones:
                cand = self._closest_pitch(prev_pc, target, low, high)
                if cand is not None:
                    new_voices[v] = cand
                    continue
            # otherwise try each chord tone
            best = None
            best_dist = float('inf')
            for pc in tones:
                cand = self._closest_pitch(pc, target, low, high)
                if cand is not None:
                    dist = abs(cand - target)
                    # penalize large leaps (more than 5 semitones)
                    if abs(cand - self.prev_voices[v]) > 5:
                        dist += 2
                    if dist < best_dist:
                        best_dist = dist
                        best = cand
            if best is not None:
                new_voices[v] = best
            else:
                # fallback: stay on previous note if within chord, else root
                if self.prev_voices[v] % 12 in tones and low <= self.prev_voices[v] <= high:
                    new_voices[v] = self.prev_voices[v]
                else:
                    root_pitch = self._closest_pitch(root_pc, target, low, high)
                    new_voices[v] = root_pitch if root_pitch is not None else target

        # Avoid parallel 5ths/8ves (simple check: if both S and T move same interval > 4 semitones, adjust T)
        s_interval = new_voices['S'] - self.prev_voices['S']
        t_interval = new_voices['T'] - self.prev_voices['T']
        if abs(s_interval) == abs(t_interval) and abs(s_interval) >= 4 and s_interval != 0:
            # try moving tenor differently
            low_t, high_t = _RANGES['T']
            alternatives = []
            for pc in tones:
                cand = self._closest_pitch(pc, new_voices['T'] + 1, low_t, high_t)
                if cand is not None and cand != new_voices['T']:
                    alternatives.append(cand)
                cand2 = self._closest_pitch(pc, new_voices['T'] - 1, low_t, high_t)
                if cand2 is not None and cand2 != new_voices['T']:
                    alternatives.append(cand2)
            if alternatives:
                new_voices['T'] = min(alternatives, key=lambda x: abs(x - self.prev_voices['T']))

        # Also avoid parallel octaves between S and B
        if abs(new_voices['S'] - new_voices['B']) % 12 == 0 and abs((new_voices['S'] - self.prev_voices['S']) - (new_voices['B'] - self.prev_voices['B'])) < 2:
            # move bass by step
            low_b, high_b = _RANGES['B']
            alt_b = self._closest_pitch(root_pc, self.prev_voices['B'] + 1, low_b, high_b)
            if alt_b is None:
                alt_b = self._closest_pitch(root_pc, self.prev_voices['B'] - 1, low_b, high_b)
            if alt_b is not None:
                new_voices['B'] = alt_b

        self.prev_voices = new_voices
        return new_voices

# ----------------------------------------------------------------------
# Audio analysis pipeline (melody, chords, key, beats)
# ----------------------------------------------------------------------
def load_audio(path: str, sr=22050):
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr, len(y)/sr

def extract_tempo_beats(y, sr, beats_per_measure=4):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo_est = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
    start_bpm = float(np.clip(np.atleast_1d(tempo_est)[0], 60, 200))
    result = librosa.beat.beat_track(y=y, sr=sr, start_bpm=start_bpm, trim=False, units='time')
    beat_times = result[1]
    bpm = int(round(float(np.atleast_1d(result[0])[0])))
    # ensure whole measures
    n_measures = len(beat_times) // beats_per_measure
    if n_measures == 0:
        n_measures = 1
    return bpm, beat_times, n_measures

def extract_melody_pyin(y, sr, beat_times, n_measures, beats_per_measure=4, solo_range=(55,84)):
    hop = 512
    fmin = librosa.midi_to_hz(solo_range[0])
    fmax = librosa.midi_to_hz(solo_range[1])
    try:
        f0, voiced_flag, voiced_prob = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop, fill_na=np.nan)
    except:
        # fallback to chroma
        return extract_melody_chroma(y, sr, beat_times, n_measures, beats_per_measure, solo_range)
    frame_times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop)
    confident = voiced_flag & (voiced_prob >= 0.3)
    midi_contour = np.full(len(f0), np.nan)
    for i in np.where(confident)[0]:
        m = 12*np.log2(f0[i]/440.0) + 69
        if np.isfinite(m):
            midi_contour[i] = int(round(m))
    # median filter
    midi_smoothed = median_filter(midi_contour, size=5, mode='constant', cval=np.nan)
    melody = []
    for m in range(n_measures):
        b0 = m * beats_per_measure
        b1 = min(b0+beats_per_measure, len(beat_times))
        t0 = beat_times[b0]
        t1 = beat_times[b1-1] + 0.1
        mask = (frame_times >= t0) & (frame_times < t1)
        vals = midi_smoothed[mask]
        valid = vals[~np.isnan(vals)]
        if len(valid) == 0:
            melody.append(melody[-1] if melody else 60)
        else:
            melody.append(int(round(np.median(valid))))
    return melody

def extract_melody_chroma(y, sr, beat_times, n_measures, beats_per_measure, solo_range):
    hop = 512
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    frame_t = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop)
    melody = []
    for m in range(n_measures):
        b0 = m * beats_per_measure
        b1 = min(b0+beats_per_measure, len(beat_times))
        t0, t1 = beat_times[b0], beat_times[b1-1]+0.1
        mask = (frame_t >= t0) & (frame_t < t1)
        if mask.sum() == 0:
            melody.append(60)
            continue
        mc = chroma[:, mask].mean(axis=1)
        pc = np.argmax(mc)
        midi = 60 + pc
        while midi < solo_range[0]: midi += 12
        while midi > solo_range[1]: midi -= 12
        melody.append(midi)
    return melody

def detect_chords(y, sr, beat_times, n_measures, beats_per_measure=4):
    hop = 512
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    frame_t = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop)
    chords = []
    for m in range(n_measures):
        b0 = m * beats_per_measure
        b1 = min(b0+beats_per_measure, len(beat_times))
        t0, t1 = beat_times[b0], beat_times[b1-1]+0.1
        mask = (frame_t >= t0) & (frame_t < t1)
        if mask.sum() == 0:
            chords.append({'root_pc': 0, 'quality': 'maj', 'name': 'C'})
            continue
        mc = chroma[:, mask].mean(axis=1)
        mc /= mc.max() + 1e-9
        best_score, best_root, best_qual = -1.0, 0, 'maj'
        for root in range(12):
            rot = np.roll(mc, -root)
            for qual, tmpl in _CHORD_TEMPLATES.items():
                s = np.dot(rot, tmpl)
                if s > best_score:
                    best_score, best_root, best_qual = s, root, qual
        suffix = _QUALITY_SUFFIX.get(best_qual, '')
        chords.append({'root_pc': best_root, 'quality': best_qual, 'name': f"{_NOTE_NAMES[best_root]}{suffix}"})
    # smoothing
    for i in range(1, n_measures-1):
        if chords[i]['name'] != chords[i-1]['name'] and chords[i]['name'] != chords[i+1]['name']:
            chords[i] = chords[i-1]
    return chords

def detect_vocal_sections(y, sr, beat_times, n_measures, beats_per_measure=4):
    hop = 512
    y_harm, _ = librosa.effects.hpss(y, margin=2.0)
    rms = librosa.feature.rms(y=y_harm, hop_length=hop)[0]
    rms_t = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    centroid = librosa.feature.spectral_centroid(y=y_harm, sr=sr, hop_length=hop)[0]
    cent_t = librosa.frames_to_time(np.arange(len(centroid)), sr=sr, hop_length=hop)
    scores = []
    for m in range(n_measures):
        b0 = m * beats_per_measure
        b1 = min(b0+beats_per_measure, len(beat_times))
        t0, t1 = beat_times[b0], beat_times[b1-1]+0.1
        mask_r = (rms_t >= t0) & (rms_t < t1)
        mask_c = (cent_t >= t0) & (cent_t < t1)
        r = np.mean(rms[mask_r]) if mask_r.sum()>0 else 0.0
        c = np.median(centroid[mask_c]) if mask_c.sum()>0 else 0.0
        vocal_weight = 1.0 + 0.5 * np.clip((c-1000)/3000, 0, 1)
        scores.append(r * vocal_weight)
    thresh = np.percentile(scores, 60)
    vocal = [s > thresh for s in scores]
    # smooth
    for i in range(1, n_measures-1):
        if vocal[i-1] == vocal[i+1] and vocal[i] != vocal[i-1]:
            vocal[i] = vocal[i-1]
    return vocal

def analyze_song_sections(y, sr, beat_times, n_measures, beats_per_measure=4):
    # simplified section detection based on RMS and novelty
    hop = 512
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms_t = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    measure_rms = [_measure_average(rms, rms_t, beat_times, n_measures, beats_per_measure)[m] for m in range(n_measures)]
    # threshold based on median and iqr
    q75, q25 = np.percentile(measure_rms, [75,25])
    iqr = q75 - q25
    high = q75 + 0.5*iqr
    low = q25 - 0.5*iqr
    labels = ['verse'] * n_measures
    for i, r in enumerate(measure_rms):
        if r > high:
            labels[i] = 'chorus'
        elif r < low:
            if i < n_measures//4:
                labels[i] = 'intro'
            elif i > 3*n_measures//4:
                labels[i] = 'outro'
            else:
                labels[i] = 'bridge'
    return labels

def _measure_average(feature, frame_times, beat_times, n_measures, beats_per_measure):
    vals = np.zeros(n_measures)
    for m in range(n_measures):
        b0 = m * beats_per_measure
        b1 = min(b0+beats_per_measure, len(beat_times))
        t0, t1 = beat_times[b0], beat_times[b1-1]+0.1
        mask = (frame_times >= t0) & (frame_times < t1)
        vals[m] = np.mean(feature[mask]) if mask.sum()>0 else 0.0
    return vals

# chord templates (same as before)
_CHORD_TEMPLATES = {
    'maj':  np.array([1,0,0,0,1,0,0,1,0,0,0,0]),
    'min':  np.array([1,0,0,1,0,0,0,1,0,0,0,0]),
    'dim':  np.array([1,0,0,1,0,0,1,0,0,0,0,0]),
    'aug':  np.array([1,0,0,0,1,0,0,0,1,0,0,0]),
    'dom7': np.array([1,0,0,0,1,0,0,1,0,0,1,0]),
    'maj7': np.array([1,0,0,0,1,0,0,1,0,0,0,1]),
    'min7': np.array([1,0,0,1,0,0,0,1,0,0,1,0]),
    'sus2': np.array([1,0,1,0,0,0,0,1,0,0,0,0]),
    'sus4': np.array([1,0,0,0,0,1,0,1,0,0,0,0]),
}
_QUALITY_SUFFIX = {'maj':'','min':'m','dim':'dim','aug':'aug','dom7':'7','maj7':'maj7','min7':'m7','sus2':'sus2','sus4':'sus4'}

# ----------------------------------------------------------------------
# Main arrangement function
# ----------------------------------------------------------------------
def audio_to_chords_and_melody(audio_file_path, beats_per_measure=4, key_sharps=None, progression=None):
    y, sr, duration = load_audio(audio_file_path)
    print(f"Loaded {duration:.1f}s")
    bpm, beat_times, n_measures = extract_tempo_beats(y, sr, beats_per_measure)
    print(f"Tempo {bpm} BPM, {n_measures} measures")

    # melody
    melody = extract_melody_pyin(y, sr, beat_times, n_measures, beats_per_measure)
    # rhythm
    melody_rhythm = extract_melody_rhythm(y, sr, beat_times, melody, beats_per_measure)

    # chords
    chords = detect_chords(y, sr, beat_times, n_measures, beats_per_measure)
    # key detection from chroma
    hop = 512
    chroma_full = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    mode, key_sharps = detect_key(chroma_full)
    print(f"Detected key: {mode.upper()} with {key_sharps} sharps")

    # vocal sections
    vocal = detect_vocal_sections(y, sr, beat_times, n_measures, beats_per_measure)
    sections = analyze_song_sections(y, sr, beat_times, n_measures, beats_per_measure)

    # SATB voicing with voice-leading
    voicer = SATBVoicer(mode, key_sharps)
    satb_parts = {v: [] for v in ['solo','S','A','T','B']}
    satb_parts['_sections'] = sections
    satb_parts['_rhythm'] = {'solo': melody_rhythm}

    for m_idx in range(n_measures):
        solo_midi = melody[m_idx] if vocal[m_idx] else None
        section_name = sections[m_idx]
        style = _TEXTURE_STYLES.get(section_name, _TEXTURE_STYLES['verse'])
        if solo_midi is not None:
            # generate SATB supporting this solo note
            satb = voicer.voice_chord(chords[m_idx], solo_midi, section_name)
            satb_parts['solo'].append(solo_midi)
            satb_parts['S'].append(satb['S'])
            satb_parts['A'].append(satb['A'])
            satb_parts['T'].append(satb['T'])
            satb_parts['B'].append(satb['B'])
        else:
            # no solo, SATB sings chord without melody (maybe hum)
            satb = voicer.voice_chord(chords[m_idx], 60, section_name)  # use middle C as dummy target
            satb_parts['solo'].append(None)
            satb_parts['S'].append(satb['S'])
            satb_parts['A'].append(satb['A'])
            satb_parts['T'].append(satb['T'])
            satb_parts['B'].append(satb['B'])

    # Optional lyrics (simplified)
    syllables = {'solo': ['ah']*n_measures, 'S':'oo', 'A':'ah', 'T':'oh', 'B':'doo'}
    return satb_parts, syllables, bpm, key_sharps

# ----------------------------------------------------------------------
# Build music21 score (with rhythm)
# ----------------------------------------------------------------------
def build_score(parts, syllables, bpm, key_sharps, title='Arrangement', artist=''):
    sc = stream.Score()
    sc.metadata = metadata.Metadata(title=title, composer=artist)
    voice_config = [
        ('Soloist', 'Solo', 'solo', clef.TrebleClef()),
        ('Soprano', 'S',   'S',    clef.TrebleClef()),
        ('Alto',    'A',   'A',    clef.TrebleClef()),
        ('Tenor',   'T',   'T',    clef.TrebleClef()),
        ('Bass',    'B',   'B',    clef.BassClef())
    ]
    for name, abbr, vkey, vclef in voice_config:
        part = stream.Part()
        part.partName = name
        part.partAbbreviation = abbr
        for m_idx, midi_val in enumerate(parts[vkey]):
            meas = stream.Measure(number=m_idx+1)
            if m_idx == 0:
                meas.append(vclef)
                meas.append(key.KeySignature(key_sharps))
                meas.append(meter.TimeSignature('4/4'))
                meas.append(tempo.MetronomeMark(number=bpm))
            if midi_val is None:
                meas.append(note.Rest(quarterLength=4.0))
            else:
                # use rhythm from solo part for all voices (simplified)
                rhythm = parts.get('_rhythm', {}).get('solo', [[4.0]])[m_idx]
                for dur in rhythm:
                    n = note.Note(midi_val)
                    n.duration.quarterLength = dur
                    # add lyric only on first note of measure for that voice
                    if vkey == 'B':
                        n.lyric = _BASS_ALT[m_idx%2]
                    elif isinstance(syllables[vkey], list) and m_idx < len(syllables[vkey]):
                        n.lyric = syllables[vkey][m_idx]
                    else:
                        n.lyric = syllables.get(vkey, 'ah')
                    meas.append(n)
            part.append(meas)
        sc.append(part)
    return sc

def arrange(parts, syllables, title='Arrangement', artist='', key_signature=3, tempo_bpm=120):
    return build_score(parts, syllables, tempo_bpm, key_signature, title, artist)

def arrange_mp3(mp3_path, output_xml='arrangement.musicxml', output_json='arrangement.json'):
    parts, syllables, bpm, key_sharps = audio_to_chords_and_melody(mp3_path)
    score = build_score(parts, syllables, bpm, key_sharps, title='A Cappella Arrangement')
    score.write('musicxml', fp=output_xml)
    print(f"Saved MusicXML to {output_xml}")
    # also save a json summary
    summary = {
        'tempo_bpm': bpm,
        'key_sharps': key_sharps,
        'measures': len(parts['solo']),
        'solo_notes': [m for m in parts['solo'] if m is not None]
    }
    with open(output_json, 'w') as f:
        json.dump(summary, f, indent=2)
    return summary

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        arrange_mp3(sys.argv[1])
    else:
        print("Usage: python Arrangr.py <song.mp3>")