# Arrangr 🎵

**Arrangr** is an automatic SATB + Solo arranger that converts audio files (MP3/WAV) into sheet music arrangements.

## Features

- 🎧 Upload MP3 or WAV audio files
- 🎼 Automatically extract melody and chord progressions
- 🎵 Create SATB (Soprano, Alto, Tenor, Bass) arrangements
- 🗣️ Auto-detect solo lyrics from audio and align words to sung notes
- 📄 Export to MusicXML format (compatible with MuseScore, Finale, Sibelius)
- 🖥️ Easy web-based interface with drag-and-drop

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/bowenzou3/Arrangr.git
   cd Arrangr
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Web Interface (Recommended)

1. **Start the web server:**
   ```bash
   python app.py
   ```

2. **Open your browser:**
   - Go to `http://localhost:5000`
   - Click to select an audio file (MP3 or WAV)
   - Drag and drop is also supported
   - Sit back while the app analyzes your audio

3. **Download your arrangement:**
   - Once processing is complete, download the `.musicxml` file
   - Open it in MuseScore, Finale, Sibelius, or any MusicXML-compatible software

## How It Works

1. **Audio Analysis** - Extracts melody using librosa's probabilistic YIN algorithm
2. **Chord Detection** - Analyzes chroma features to detect chord progressions
3. **SATB Voicing** - Intelligently distributes notes across four vocal parts
4. **Lyric Alignment** - Transcribes solo lyrics from audio (when available) and aligns to sung measures
5. **MusicXML Export** - Generates a professional sheet music file

## Requirements

- Python 3.8+
- Flask
- music21
- librosa
- numpy

See `requirements.txt` for full dependencies.

## Output Format

Arrangr exports files in **MusicXML** format, which can be opened in:
- [MuseScore](https://musescore.org/) - Free and open-source ⭐ Recommended
- Finale
- Sibelius
- Notion
- And many other music notation software

## Troubleshooting

**Issue:** Flask won't start  
→ Make sure you have Flask installed: `pip install -q flask`  
→ Check that port 5000 is not already in use

**Issue:** Audio analysis takes a long time  
→ This is normal for longer audio files (2+ minutes)  
→ Processing time depends on file length and your computer

**Issue:** Downloaded .musicxml file won't open  
→ Make sure you have MusicXML-compatible software installed  
→ Try [MuseScore](https://musescore.org/) (free alternative)

## Supported Formats

- **Input:** MP3, WAV
- **Output:** MusicXML (.musicxml)

## License

This project is open source.

## Author

Created by Bowen Zou
