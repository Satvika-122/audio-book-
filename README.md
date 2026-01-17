# 🎧 Audiobook TTS Generator

A powerful, single-cell Colab notebook that converts text chapters into professional audiobooks with multiple character voices, emotion detection, and natural dialogue flow.

## ✨ Features

### **Voice System**
- **3 Distinct Character Voices:**
  - 📖 **Narrator**: British male voice (gTTS UK English)
  - 👨 **Jonah**: American male voice (Facebook MMS TTS model)
  - 👩 **Mira**: American female voice (gTTS US English)

### **Advanced Processing**
- 🎭 **Emotion Detection**: Automatically detects and applies appropriate vocal emotions (whispering, fearful, urgent, calm, excited)
- 💬 **Smart Dialogue Flow**: Keeps dialogue and narration together with natural pauses (e.g., "Hello," he said.)
- 📊 **Progress Tracking**: Real-time progress display during generation
- 🎚️ **Audio Normalization**: Automatic volume balancing and quality optimization

### **Output Options**
- 🔊 **Complete Chapter**: Full chapter audio in WAV format
- ⏱️ **Smart Previews**: Automatic preview generation for long chapters
- 📁 **Multiple Formats**: WAV files with proper metadata

## 🚀 Quick Start

### **Option 1: Run in Google Colab (Recommended)**
1. Open [Google Colab](https://colab.research.google.com/)
2. Create a new notebook
3. Copy the entire code from the main notebook
4. Run all cells (or use the single-cell version)
5. Your audiobook will be generated in `/content/audio_book/audio/`

### **Option 2: Local Setup**
```bash
# Clone the repository
git clone https://github.com/yourusername/audiobook-tts.git
cd audiobook-tts

# Install dependencies
pip install transformers accelerate sentencepiece gtts pydub soundfile numpy

# Run the script
python audiobook_generator.py
