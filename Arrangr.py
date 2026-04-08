"""
Arrangr.py — MP3 → SATB + Soloist A Cappella Arrangement  (v3)
===============================================================
Architecture:
    ① Variable rhythms per 4-beat measure (half/quarter/eighth/sixteenth)
  ② Part order: Soloist → Soprano → Alto → Tenor → Bass
  ③ Bass clef for Bass; treble for all others
    ④ SATB supports solo with chord-aware backing voicings
  ⑤ Vocal detection via RMS energy
  ⑥ Chroma-based melody (1 singable note per measure)
  ⑦ Melodic smoothing — no jumps > a perfect 5th
"""

import json
import re
import warnings
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
from music21 import clef, key, metadata, meter, note, stream, tempo
from requests.exceptions import RequestsDependencyWarning

warnings.filterwarnings('ignore', category=RequestsDependencyWarning)

_WHISPER_MODEL = None

# ── Constants ──────────────────────────────────────────────────────────────────
_NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
_BASS_ALT   = ['doo', 'dum']

A_MAJOR_PCS = {9, 11, 1, 2, 4, 6, 8}   # A B C# D E F# G#

DEFAULT_PROGRESSION = [
    {'root_pc': 9,  'quality': 'maj', 'name': 'A'},
    {'root_pc': 4,  'quality': 'maj', 'name': 'E'},
    {'root_pc': 6,  'quality': 'min', 'name': 'F#m'},
    {'root_pc': 2,  'quality': 'maj', 'name': 'D'},
]

DEFAULT_VOICINGS = {
    'A':   {'S': 69, 'A': 64, 'T': 57, 'B': 45},
    'E':   {'S': 71, 'A': 64, 'T': 59, 'B': 40},
    'F#m': {'S': 69, 'A': 66, 'T': 61, 'B': 42},
    'D':   {'S': 69, 'A': 66, 'T': 62, 'B': 38},
}

RHYTHM_LIBRARY = {
    'solo': {
        'simple': [[2.0, 2.0], [1.0, 1.0, 1.0, 1.0]],
        'medium': [[1.0, 0.5, 0.5, 1.0, 1.0], [2.0, 1.0, 1.0]],
        'busy': [[1.0, 0.25, 0.25, 0.5, 1.0, 1.0], [0.5, 0.5, 1.0, 0.5, 0.5, 1.0]],
    },
    'S': {
        'simple': [[2.0, 2.0], [2.0, 1.0, 1.0]],
        'medium': [[1.0, 1.0, 1.0, 1.0], [1.0, 0.5, 0.5, 1.0, 1.0]],
        'busy': [[0.5, 0.5, 1.0, 0.5, 0.5, 1.0], [1.0, 0.25, 0.25, 1.0, 0.5, 1.0]],
    },
    'A': {
        'simple': [[2.0, 2.0], [1.0, 1.0, 1.0, 1.0]],
        'medium': [[2.0, 1.0, 1.0], [1.0, 0.5, 0.5, 1.0, 1.0]],
        'busy': [[0.5, 0.5, 1.0, 0.5, 0.5, 1.0], [1.0, 0.25, 0.25, 0.5, 1.0, 1.0]],
    },
    'T': {
        'simple': [[2.0, 2.0], [2.0, 1.0, 1.0]],
        'medium': [[1.0, 1.0, 1.0, 1.0], [0.5, 0.5, 1.0, 0.5, 0.5, 1.0]],
        'busy': [[1.0, 0.25, 0.25, 0.5, 1.0, 1.0], [0.5, 0.5, 0.5, 0.5, 1.0, 1.0]],
    },
    'B': {
        'simple': [[2.0, 2.0], [1.0, 1.0, 1.0, 1.0]],
        'medium': [[2.0, 1.0, 1.0], [1.0, 0.5, 0.5, 1.0, 1.0]],
        'busy': [[0.5, 0.5, 1.0, 0.5, 0.5, 1.0], [0.5, 0.5, 0.5, 0.5, 1.0, 1.0]],
    },
}

SECTION_DENSITY = {
    'intro': 'simple',
    'verse': 'medium',
    'chorus': 'busy',
    'bridge': 'medium',
    'outro': 'simple',
}

BACKING_STYLE_RULES = {
    'intro': 'tight',
    'verse': 'tight',
    'chorus': 'loose',
    'bridge': 'loose',
    'outro': 'tight',
}


def _z_norm(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd < 1e-9:
        return np.zeros_like(x, dtype=float)
    return (x - mu) / sd


def _measure_average(feature: np.ndarray, frame_times: np.ndarray,
                     beat_times: np.ndarray, n_measures: int,
                     beats_per_measure: int = 4) -> np.ndarray:
    vals = np.zeros(n_measures, dtype=float)
    for m in range(n_measures):
        b0 = m * beats_per_measure
        b1 = min(b0 + beats_per_measure, len(beat_times))
        t0 = beat_times[b0]
        t1 = beat_times[b1 - 1] + 0.1
        mask = (frame_times >= t0) & (frame_times < t1)
        vals[m] = float(np.mean(feature[mask])) if np.any(mask) else 0.0
    return vals


def analyze_song_sections(y, sr, beat_times, n_measures: int,
                          beats_per_measure: int = 4) -> list:
    """Detect section labels from audio features (novelty/onset/energy)."""
    if n_measures <= 0:
        return []
    if n_measures < 10:
        return ['verse'] * n_measures

    hop = 512
    frame_times = librosa.frames_to_time(np.arange(len(y) // hop + 1), sr=sr, hop_length=hop)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    onset_t = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=hop)

    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms_t = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    novelty = np.zeros(chroma.shape[1], dtype=float)
    if chroma.shape[1] > 1:
        novelty[1:] = np.linalg.norm(np.diff(chroma, axis=1), axis=0)
    novelty_t = librosa.frames_to_time(np.arange(len(novelty)), sr=sr, hop_length=hop)

    m_onset = _measure_average(onset_env, onset_t, beat_times, n_measures, beats_per_measure)
    m_energy = _measure_average(rms, rms_t, beat_times, n_measures, beats_per_measure)
    m_novel = _measure_average(novelty, novelty_t, beat_times, n_measures, beats_per_measure)

    z_onset = _z_norm(m_onset)
    z_energy = _z_norm(m_energy)
    z_novel = _z_norm(m_novel)

    intensity = 0.45 * z_onset + 0.35 * z_energy + 0.20 * z_novel
    kernel = np.array([0.25, 0.5, 0.25])
    intensity_sm = np.convolve(intensity, kernel, mode='same')

    # Boundary strength combines intensity and novelty changes.
    d_int = np.abs(np.diff(intensity_sm, prepend=intensity_sm[0]))
    d_nov = np.abs(np.diff(z_novel, prepend=z_novel[0]))
    boundary_strength = d_int + 0.5 * d_nov

    min_gap = max(4, beats_per_measure)
    thresh = float(np.percentile(boundary_strength[1:], 70)) if n_measures > 1 else 0.0
    cut_points = [0]
    for m in range(1, n_measures - 1):
        if boundary_strength[m] >= thresh and (m - cut_points[-1]) >= min_gap:
            cut_points.append(m)
    if (n_measures - cut_points[-1]) < min_gap and len(cut_points) > 1:
        cut_points.pop()
    cut_points.append(n_measures)

    segments = []
    for i in range(len(cut_points) - 1):
        s, e = cut_points[i], cut_points[i + 1]
        if e > s:
            segments.append({'start': s, 'end': e})
    if not segments:
        return ['verse'] * n_measures

    for seg in segments:
        s, e = seg['start'], seg['end']
        seg['mean_intensity'] = float(np.mean(intensity_sm[s:e]))
        seg['mean_energy'] = float(np.mean(z_energy[s:e]))
        seg['mean_novelty'] = float(np.mean(z_novel[s:e]))

    labels = ['verse'] * n_measures
    global_energy = float(np.mean(z_energy))

    # Intro/outro: low-energy edges become sparse regions.
    first = segments[0]
    if first['mean_energy'] < global_energy - 0.15 or (first['end'] - first['start']) <= 4:
        for m in range(first['start'], first['end']):
            labels[m] = 'intro'

    last = segments[-1]
    if last['mean_energy'] < global_energy - 0.10 or (last['end'] - last['start']) <= 4:
        for m in range(last['start'], last['end']):
            labels[m] = 'outro'

    middle_indices = [
        i for i in range(len(segments))
        if labels[segments[i]['start']] == 'verse'
    ]
    if not middle_indices:
        return labels

    seg_intensity = np.array([segments[i]['mean_intensity'] for i in middle_indices], dtype=float)
    chorus_thr = float(np.percentile(seg_intensity, 65)) if seg_intensity.size else 0.0
    chorus_candidates = [
        i for i in middle_indices
        if segments[i]['mean_intensity'] >= chorus_thr
    ]
    if not chorus_candidates:
        chorus_candidates = [middle_indices[int(np.argmax(seg_intensity))]]

    for idx in chorus_candidates:
        s, e = segments[idx]['start'], segments[idx]['end']
        for m in range(s, e):
            labels[m] = 'chorus'

    non_chorus = [i for i in middle_indices if i not in chorus_candidates]
    if non_chorus:
        bridge_pool = [i for i in non_chorus if segments[i]['start'] >= n_measures // 2]
        bridge_candidates = bridge_pool if bridge_pool else non_chorus
        bridge_idx = max(bridge_candidates, key=lambda i: segments[i]['mean_novelty'])
        if segments[bridge_idx]['mean_novelty'] > 0.20:
            s, e = segments[bridge_idx]['start'], segments[bridge_idx]['end']
            for m in range(s, e):
                labels[m] = 'bridge'

    return labels


def _get_rhythm_pattern(vkey: str, measure_idx: int,
                        section_name: str = 'verse', solo_active: bool = False) -> list:
    """Return a 4/4-safe rhythm pattern using section-aware density."""
    voice_map = RHYTHM_LIBRARY.get(vkey)
    if voice_map is None:
        return [4.0]

    density = SECTION_DENSITY.get(section_name, 'medium')
    if solo_active and vkey in ('S', 'A', 'T', 'B'):
        if density == 'busy':
            density = 'medium'
        elif density == 'medium':
            density = 'simple'

    patterns = voice_map.get(density, voice_map.get('medium', [[4.0]]))
    pattern = patterns[measure_idx % len(patterns)]
    total = float(sum(pattern))
    if abs(total - 4.0) > 1e-9:
        return [4.0]
    return pattern


def _chord_pitch_classes(chord_info: dict) -> list:
    root = int(chord_info.get('root_pc', 9)) % 12
    quality = chord_info.get('quality', 'maj')
    third = 3 if quality == 'min' else 4
    return [root, (root + third) % 12, (root + 7) % 12]


def _closest_pitch_for_pc(pc: int, target_midi: int,
                          low: int = 64, high: int = 81) -> Optional[int]:
    candidates = []
    for octv in range(-2, 11):
        midi_val = pc + 12 * octv
        if low <= midi_val <= high:
            candidates.append(midi_val)
    if not candidates:
        return None
    return min(candidates, key=lambda v: abs(v - target_midi))


def _pick_note_from_pcs(pcs: list, target_midi: int, low: int, high: int,
                        avoid_midi: Optional[int] = None,
                        preferred_pc: Optional[int] = None) -> Optional[int]:
    best = None
    for pc in pcs:
        cand = _closest_pitch_for_pc(pc, target_midi, low=low, high=high)
        if cand is None:
            continue
        score = abs(cand - target_midi)
        if avoid_midi is not None and abs(cand - avoid_midi) <= 1:
            score += 2.0
        if preferred_pc is not None and (cand % 12) == preferred_pc:
            score -= 0.5
        if best is None or score < best[0]:
            best = (score, cand)
    return best[1] if best is not None else None


def _auto_backing_style(section_name: str, solo_midi: int,
                        measure_idx: int, prev_style: str = 'tight') -> str:
    """Pick tight/loose backing automatically from section + melody context."""
    style = BACKING_STYLE_RULES.get(section_name, 'tight')

    # High solo melodies generally blend better with tighter support.
    if solo_midi >= 72:
        style = 'tight'
    # Low solo melodies can take wider spacing, especially in bigger sections.
    elif solo_midi <= 66 and section_name in ('chorus', 'bridge'):
        style = 'loose'

    # In verses, briefly open the voicing every 8 measures for contrast.
    if section_name == 'verse' and measure_idx % 8 == 4 and solo_midi <= 69:
        style = 'loose'

    # Avoid jumpy toggling: keep previous style on odd bars in same section class.
    if style != prev_style and measure_idx % 2 == 1 and section_name in ('verse', 'intro', 'outro'):
        style = prev_style

    return style


def _choose_satb_support_notes(solo_midi: int, chord_info: dict,
                               fallback_voicing: dict,
                               backing_style: str = 'tight') -> dict:
    """Choose SATB chord tones that support the solo in each measure."""
    pcs = _chord_pitch_classes(chord_info)
    root_pc = int(chord_info.get('root_pc', pcs[0])) % 12

    if backing_style == 'loose':
        s_target, a_target, t_target, b_target = solo_midi + 8, solo_midi - 6, solo_midi - 14, 40
        s_rng, a_rng, t_rng, b_rng = (64, 84), (53, 74), (45, 66), (34, 54)
        gap_sa, gap_at, gap_tb = 5, 7, 10
    else:
        s_target, a_target, t_target, b_target = solo_midi + 4, solo_midi - 3, solo_midi - 10, 43
        s_rng, a_rng, t_rng, b_rng = (64, 81), (55, 74), (48, 67), (36, 55)
        gap_sa, gap_at, gap_tb = 3, 4, 6

    s = _pick_note_from_pcs(pcs, s_target, s_rng[0], s_rng[1], avoid_midi=solo_midi)
    a = _pick_note_from_pcs(pcs, a_target, a_rng[0], a_rng[1], avoid_midi=solo_midi)
    t = _pick_note_from_pcs(pcs, t_target, t_rng[0], t_rng[1], avoid_midi=solo_midi)
    b = _pick_note_from_pcs(pcs, b_target, b_rng[0], b_rng[1], preferred_pc=root_pc)

    # Keep vertical spacing sensible for SATB when supporting the solo.
    if s is not None and a is not None and (s - a) < gap_sa:
        a = _pick_note_from_pcs(
            pcs, s - gap_sa, a_rng[0], min(a_rng[1], s - 1), avoid_midi=solo_midi
        ) or a
    if a is not None and t is not None and (a - t) < gap_at:
        t = _pick_note_from_pcs(
            pcs, a - gap_at, t_rng[0], min(t_rng[1], a - 1), avoid_midi=solo_midi
        ) or t
    if t is not None and b is not None and (t - b) < gap_tb:
        b = _pick_note_from_pcs(
            pcs, t - gap_tb, b_rng[0], min(b_rng[1], t - 1), preferred_pc=root_pc
        ) or b

    return {
        'S': s if s is not None else fallback_voicing['S'],
        'A': a if a is not None else fallback_voicing['A'],
        'T': t if t is not None else fallback_voicing['T'],
        'B': b if b is not None else fallback_voicing['B'],
    }


# ── Step 1: Load Audio ─────────────────────────────────────────────────────────
def load_audio(path: str, sr: int = 22050):
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr, len(y) / sr


# ── Step 2: Tempo + Beat Grid ──────────────────────────────────────────────────
def extract_tempo_beats(y, sr, beats_per_measure: int = 4):
    result      = librosa.beat.beat_track(y=y, sr=sr, units='frames')
    tempo_arr   = np.atleast_1d(result[0])
    beat_frames = result[1]
    bpm         = int(round(float(tempo_arr[0])))
    beat_times  = librosa.frames_to_time(beat_frames, sr=sr)
    n_measures  = len(beat_times) // beats_per_measure
    return bpm, beat_times, n_measures


# ── Step 3: Melody (chroma-based, 1 note per measure) ─────────────────────────
def _extract_melody_chroma(y, sr, beat_times, n_measures: int,
                            beats_per_measure: int = 4,
                            scale_pcs: set = None,
                            solo_range: tuple = (64, 76)) -> list:
    if scale_pcs is None:
        scale_pcs = A_MAJOR_PCS
    hop     = 512
    chroma  = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    frame_t = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop)

    melody = []
    for m in range(n_measures):
        b0 = m * beats_per_measure
        b1 = min(b0 + beats_per_measure, len(beat_times))
        t0, t1 = beat_times[b0], beat_times[b1 - 1] + 0.1
        mask = (frame_t >= t0) & (frame_t < t1)
        if mask.sum() == 0:
            melody.append(solo_range[0])
            continue
        mc = chroma[:, mask].mean(axis=1)
        mc /= mc.max() + 1e-9
        scored = sorted([(mc[pc] * (2.0 if pc in scale_pcs else 0.3), pc)
                         for pc in range(12)], reverse=True)
        mel_pc = next((pc for _, pc in scored if pc in scale_pcs), scored[0][1])
        midi = 60 + mel_pc
        while midi < solo_range[0]: midi += 12
        while midi > solo_range[1]: midi -= 12
        melody.append(int(np.clip(midi, solo_range[0], solo_range[1])))

    # Smooth: cap jumps at 7 semitones
    for i in range(1, len(melody)):
        if melody[i] and melody[i - 1]:
            diff = melody[i] - melody[i - 1]
            if abs(diff) > 7:
                melody[i] -= 12 * int(diff / abs(diff))
            melody[i] = int(np.clip(melody[i], solo_range[0], solo_range[1]))
    return melody


# ── Step 4: Chord Detection (one per measure) ─────────────────────────────────
def _detect_chords_chroma(y, sr, beat_times, n_measures: int,
                           beats_per_measure: int = 4,
                           progression: list = None) -> list:
    if progression is not None:
        return [progression[m % len(progression)] for m in range(n_measures)]
    hop     = 512
    chroma  = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    frame_t = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop)
    templates = {
        'maj': np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], float),
        'min': np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], float),
    }
    chords = []
    for m in range(n_measures):
        b0 = m * beats_per_measure
        b1 = min(b0 + beats_per_measure, len(beat_times))
        t0, t1 = beat_times[b0], beat_times[b1 - 1] + 0.1
        mask = (frame_t >= t0) & (frame_t < t1)
        if mask.sum() == 0:
            chords.append(DEFAULT_PROGRESSION[m % 4])
            continue
        mc = chroma[:, mask].mean(axis=1)
        mc /= mc.max() + 1e-9
        best_score, best_root, best_qual = -1, 9, 'maj'
        for root in range(12):
            rot = np.roll(mc, -root)
            for qual, tmpl in templates.items():
                s = float(np.dot(rot, tmpl))
                if s > best_score:
                    best_score, best_root, best_qual = s, root, qual
        chords.append({'root_pc': best_root, 'quality': best_qual,
                       'name': f"{_NOTE_NAMES[best_root]}{'m' if best_qual == 'min' else ''}"})
    return chords


# ── Step 5: Vocal Section Detection (RMS) ─────────────────────────────────────
def detect_vocal_sections(y, sr, beat_times, n_measures: int,
                           beats_per_measure: int = 4) -> list:
    hop   = 512
    rms   = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms_t = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    measure_rms = []
    for m in range(n_measures):
        b0 = m * beats_per_measure
        b1 = min(b0 + beats_per_measure, len(beat_times))
        t0, t1 = beat_times[b0], beat_times[b1 - 1] + 0.1
        mask = (rms_t >= t0) & (rms_t < t1)
        measure_rms.append(float(rms[mask].mean()) if mask.sum() > 0 else 0.0)
    intro_rms = float(np.mean(measure_rms[:min(7, n_measures)]))
    threshold = intro_rms * 1.2
    vocal = [r > threshold for r in measure_rms]
    for i in range(min(7, n_measures)):
        vocal[i] = False
    return vocal


# ── Step 6: Assign SATB Voices ────────────────────────────────────────────────
def assign_satb(melody: list, chords: list, vocal_sections: list,
                voicings: dict = None,
                section_labels: Optional[list] = None) -> dict:
    if voicings is None:
        voicings = DEFAULT_VOICINGS
    if section_labels is None:
        section_labels = ['verse'] * len(melody)
    parts = {v: [] for v in ('solo', 'S', 'A', 'T', 'B')}
    backing_styles = []
    prev_style = 'tight'
    for m in range(len(melody)):
        solo_midi  = melody[m] if vocal_sections[m] else None
        solo_sings = solo_midi is not None
        section_name = section_labels[m] if m < len(section_labels) else 'verse'
        cname      = chords[m].get('name', 'A')
        vc         = voicings.get(cname, voicings['A'])
        if solo_sings:
            style = _auto_backing_style(section_name, solo_midi, m, prev_style)
            support_vc = _choose_satb_support_notes(solo_midi, chords[m], vc, backing_style=style)
            prev_style = style
            backing_styles.append(style)
            sopr_midi = support_vc['S']
            alto_midi = support_vc['A']
            tenor_midi = support_vc['T']
            bass_midi = support_vc['B']
        else:
            backing_styles.append('tight')
            sopr_midi = vc['S']
            alto_midi = vc['A']
            tenor_midi = vc['T']
            bass_midi = vc['B']
        parts['solo'].append(solo_midi)
        parts['S'].append(sopr_midi)
        parts['A'].append(alto_midi)
        parts['T'].append(tenor_midi)
        parts['B'].append(bass_midi)
    parts['_backing_style'] = backing_styles
    return parts


# ── Step 7: Assign Syllables ──────────────────────────────────────────────────
def assign_syllables(lyrics: Optional[list] = None) -> dict:
    return {'solo': lyrics if lyrics else 'ah', 'S': 'oo', 'A': 'ah', 'T': 'oh', 'B': 'doo'}


def _load_whisper_model(model_name: str = 'tiny'):
    global _WHISPER_MODEL
    if _WHISPER_MODEL is not None:
        return _WHISPER_MODEL
    try:
        import whisper
    except ImportError:
        return None
    try:
        _WHISPER_MODEL = whisper.load_model(model_name)
    except Exception:
        return None
    return _WHISPER_MODEL


def _segment_to_word_timestamps(seg: dict) -> list:
    text = (seg.get('text') or '').strip()
    if not text:
        return []
    words = re.findall(r"[A-Za-z']+", text)
    if not words:
        return []
    start = float(seg.get('start', 0.0) or 0.0)
    end = float(seg.get('end', start + 0.1) or (start + 0.1))
    dur = max(0.1, end - start)
    step = dur / len(words)
    return [
        {'word': w.lower(), 'time': start + (i + 0.5) * step}
        for i, w in enumerate(words)
    ]


def _transcribe_word_timeline(audio_file_path: str) -> list:
    model = _load_whisper_model('tiny')
    if model is None:
        return []
    try:
        result = model.transcribe(
            audio_file_path,
            task='transcribe',
            verbose=False,
            fp16=False,
        )
    except Exception:
        return []
    timeline = []
    for seg in result.get('segments', []):
        timeline.extend(_segment_to_word_timestamps(seg))
    return timeline


def _align_solo_lyrics_to_measures(word_timeline: list, beat_times,
                                   n_measures: int, vocal_sections: list,
                                   bpm: int, beats_per_measure: int = 4) -> Optional[list]:
    if not word_timeline or n_measures <= 0:
        return None
    lyrics_by_measure = [''] * n_measures
    beat_dur = 60.0 / max(1, bpm)
    wi = 0

    for m in range(n_measures):
        if not vocal_sections[m]:
            continue
        b0 = m * beats_per_measure
        b1 = min(b0 + beats_per_measure, len(beat_times))
        if b0 >= len(beat_times):
            break
        t0 = beat_times[b0]
        t1 = beat_times[b1 - 1] + beat_dur

        while wi < len(word_timeline) and word_timeline[wi]['time'] < t0:
            wi += 1
        if wi < len(word_timeline) and word_timeline[wi]['time'] < t1:
            lyrics_by_measure[m] = word_timeline[wi]['word']
            wi += 1
        elif wi < len(word_timeline):
            lyrics_by_measure[m] = word_timeline[wi]['word']
            wi += 1
        else:
            lyrics_by_measure[m] = 'ah'

    return lyrics_by_measure


def detect_solo_lyrics(audio_file_path: str, beat_times, n_measures: int,
                       vocal_sections: list, bpm: int,
                       beats_per_measure: int = 4) -> Optional[list]:
    """Transcribe lyrics from audio and align one lyric token per sung solo measure."""
    word_timeline = _transcribe_word_timeline(audio_file_path)
    return _align_solo_lyrics_to_measures(
        word_timeline, beat_times, n_measures, vocal_sections, bpm, beats_per_measure
    )


# ── Step 8: Build music21 Score ───────────────────────────────────────────────
def build_score(parts: dict, syllables: dict, bpm: int,
                key_sharps: int = 3,
                title: str = 'A Cappella Arrangement',
                artist: str = '') -> stream.Score:
    sc = stream.Score()
    sc.metadata = metadata.Metadata()
    sc.metadata.title = title
    if artist:
        sc.metadata.addContributor(metadata.Contributor(role='composer', name=artist))
    sc.metadata.addContributor(metadata.Contributor(role='arranger', name='Arrangr'))
    VOICE_CONFIG = [
        ('Soloist', 'Solo.', 'solo', clef.TrebleClef()),
        ('Soprano', 'S.',    'S',    clef.TrebleClef()),
        ('Alto',    'A.',    'A',    clef.TrebleClef()),
        ('Tenor',   'T.',    'T',    clef.TrebleClef()),
        ('Bass',    'B.',    'B',    clef.BassClef()),
    ]
    for (vname, vabbr, vkey, vclef) in VOICE_CONFIG:
        p   = stream.Part()
        p.partName = vname
        p.partAbbreviation = vabbr
        syl = syllables.get(vkey, 'ah')
        section_labels = parts.get('_sections', ['verse'] * len(parts.get(vkey, [])))
        for m_idx, midi_val in enumerate(parts[vkey]):
            m_obj = stream.Measure(number=m_idx + 1)
            if m_idx == 0:
                m_obj.append(vclef)
                m_obj.append(key.KeySignature(key_sharps))
                m_obj.append(meter.TimeSignature('4/4'))
                m_obj.append(tempo.MetronomeMark(number=bpm))
            if midi_val is None:
                r = note.Rest()
                r.duration.type = 'whole'
                m_obj.append(r)
            else:
                lyric_text = None
                if vkey == 'B':
                    lyric_text = _BASS_ALT[m_idx % 2]
                elif isinstance(syl, list):
                    lyric_text = syl[m_idx] if m_idx < len(syl) else syl[-1]
                else:
                    lyric_text = syl

                section_name = section_labels[m_idx] if m_idx < len(section_labels) else 'verse'
                solo_active = parts['solo'][m_idx] is not None
                rhythm_pattern = _get_rhythm_pattern(vkey, m_idx, section_name, solo_active)

                for n_idx, quarter_len in enumerate(rhythm_pattern):
                    n_ = note.Note(midi_val)
                    n_.duration.quarterLength = quarter_len
                    if n_idx == 0 and lyric_text:
                        n_.lyric = lyric_text
                    m_obj.append(n_)
            p.append(m_obj)
        sc.append(p)
    return sc


# ── Public API (called by app.py) ──────────────────────────────────────────────
def audio_to_chords_and_melody(audio_file_path: str,
                                beats_per_measure: int = 4,
                                key_sharps: int = 3,
                                progression: list = None,
                                auto_detect_lyrics: bool = True):
    """
    Analyse an audio file and return voice parts + metadata.
    Returns (parts, syllables, bpm, key_sharps).
    """
    print(f"[1/5] Loading audio: {audio_file_path}")
    y, sr, duration = load_audio(audio_file_path)
    print(f"      Duration: {duration:.1f}s")

    print("[2/5] Extracting tempo + beat grid …")
    bpm, beat_times, n_measures = extract_tempo_beats(y, sr, beats_per_measure)
    print(f"      BPM={bpm}  measures={n_measures}")

    print("[3/5] Extracting melody …")
    melody = _extract_melody_chroma(y, sr, beat_times, n_measures, beats_per_measure)

    print("[4/5] Detecting chords …")
    prog   = progression if progression is not None else DEFAULT_PROGRESSION
    chords = _detect_chords_chroma(y, sr, beat_times, n_measures, beats_per_measure,
                                    progression=prog)

    print("[5/5] Assigning voices …")
    vocal     = detect_vocal_sections(y, sr, beat_times, n_measures, beats_per_measure)
    sections  = analyze_song_sections(y, sr, beat_times, n_measures, beats_per_measure)
    parts     = assign_satb(melody, chords, vocal, section_labels=sections)
    parts['_sections'] = sections
    solo_lyrics = None
    if auto_detect_lyrics:
        print("      Detecting solo lyrics …")
        solo_lyrics = detect_solo_lyrics(
            audio_file_path,
            beat_times,
            n_measures,
            vocal,
            bpm,
            beats_per_measure,
        )
        if solo_lyrics:
            lyric_count = sum(1 for w in solo_lyrics if w)
            print(f"      Lyrics aligned: {lyric_count} tokens")
        else:
            print("      Lyrics unavailable (fallback to default syllable)")

    syllables = assign_syllables(solo_lyrics)
    n_vocal   = sum(vocal)
    print(f"      Vocal: {n_vocal}  Instrumental: {n_measures - n_vocal}")

    return parts, syllables, bpm, key_sharps


def arrange(parts, syllables, title: str = 'A Cappella Arrangement',
            artist: str = '', key_signature: int = 3, tempo_bpm: int = 120) -> stream.Score:
    """Build and return a music21 Score. Called by app.py after audio_to_chords_and_melody()."""
    return build_score(parts, syllables, tempo_bpm, key_signature, title=title, artist=artist)


# ── Full pipeline (MP3 → JSON + MusicXML) ─────────────────────────────────────
def arrange_mp3(mp3_path: str,
                output_json: str = 'arrangement.json',
                output_xml:  str = 'arrangement.xml',
                lyrics: Optional[list] = None,
                beats_per_measure: int = 4,
                key_sharps: int = 3,
                progression: list = None,
                voicings: dict = None) -> dict:
    """Full pipeline: MP3 → MusicXML + JSON."""
    parts, syllables, bpm, _ = audio_to_chords_and_melody(
        mp3_path, beats_per_measure, key_sharps, progression
    )
    if lyrics:
        syllables = assign_syllables(lyrics)

    score = build_score(parts, syllables, bpm, key_sharps)
    score.write('musicxml', fp=output_xml)
    print(f"✓ MusicXML → {output_xml}")

    n_measures = len(parts['solo'])
    arrangement = {
        'title':          'A Cappella Arrangement',
        'source':         mp3_path,
        'key_sharps':     key_sharps,
        'tempo_bpm':      bpm,
        'total_measures': n_measures,
        'parts_order':    ['solo', 'S', 'A', 'T', 'B'],
        'clefs':          {'solo': 'treble', 'S': 'treble', 'A': 'treble',
                           'T': 'treble', 'B': 'bass'},
        'syllables':      syllables,
        'parts': {
            vk: [
                {
                    'measure':  i + 1,
                    'pitch':    f"{_NOTE_NAMES[v % 12]}{v // 12 - 1}" if v else 'rest',
                    'midi':     v,
                    'duration': 'whole',
                    'syllable': (_BASS_ALT[i % 2] if vk == 'B' and v else
                                 (syllables[vk] if isinstance(syllables[vk], str)
                                  else syllables[vk][i]) if v else None)
                }
                for i, v in enumerate(parts[vk])
            ]
            for vk in ('solo', 'S', 'A', 'T', 'B')
        }
    }
    with open(output_json, 'w') as f:
        json.dump(arrangement, f, indent=2)
    print(f"✓ JSON      → {output_json}")

    print("\n── Arrangement Summary ──")
    print(f"  Tempo   : {bpm} BPM  |  Measures: {n_measures}")
    for vk in ('solo', 'S', 'A', 'T', 'B'):
        nn = sum(1 for v in parts[vk] if v is not None)
        print(f"  {vk:8s}: {nn:3d} notes  {n_measures - nn:3d} rests")
    return arrangement


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        mp3 = sys.argv[1]
        if not Path(mp3).exists():
            print(f"File '{mp3}' not found.")
            print("Usage: python Arrangr.py <song.mp3>")
        else:
            arrange_mp3(mp3_path=mp3, output_json='arrangement.json',
                        output_xml='arrangement.xml', key_sharps=3,
                        progression=DEFAULT_PROGRESSION)
    else:
        print("Arrangr — SATB + Soloist A Cappella Arranger")
        print("To run the web app: python app.py")
