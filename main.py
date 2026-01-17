# =========================
# AUDIOBOOK TTS WITH IMPROVED DIALOGUE-NARRATION FLOW
# =========================

!pip install -q transformers accelerate sentencepiece gtts pydub soundfile numpy
!apt-get install -y ffmpeg > /dev/null 2>&1

import os, re
import numpy as np
import soundfile as sf
from gtts import gTTS
from IPython.display import Audio, display
from pydub import AudioSegment
from pydub.effects import speedup
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

# =========================
# CONFIGURATION
# =========================
BASE_DIR = "/content/audio_book"
AUDIO_DIR = f"{BASE_DIR}/audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

SAMPLE_RATE = 24000

# =========================
# CHAPTER TEXT (same as before)
# =========================
CHAPTER_TEXT = """Title: The Copper Key
Chapter 3: Rain on Platform Nine

The rain started as a polite tapping, then turned into a steady drumming that made the streetlights blur. Mira pulled her hood tighter and checked the time on the cracked screen of her phone. 11:47 p.m. Platform Nine would be empty at this hour—empty enough, she told herself, for a quick mistake and a quick exit.

Across the station forecourt, an old clock ticked with the confidence of something that had never met a deadline. Beneath it, the entrance doors sighed open and shut as the wind pushed them like tired lungs.

Mira stepped inside. Warmth, stale and metallic, wrapped around her. The station smelled of wet concrete and coffee that had been burned twice. A cleaning robot—more scuffed than helpful—rolled past her ankle and chirped an apologetic beep as if it recognized her from another bad decision.

She followed the signs past shuttered kiosks and benches that looked like they had swallowed a century of secrets. Somewhere deeper, a public announcement crackled and died mid-sentence.

Platform Nine sat at the far end, unmarked. Not hidden, exactly—just ignored. The overhead lights above it flickered with a stubborn rhythm, like they were trying to remember what "bright" meant.

Mira hesitated at the stairs down. She hadn't told anyone she was coming. She hadn't told anyone about the letter, either—the one that appeared in her mailbox with no stamp, no return address, just a single line written in clean, slanted ink:

BRING THE KEY. COME ALONE.

She wasn't sure which part was worse. That someone knew about the key, or that they believed she would obey.

Her hand tightened around the object in her pocket. It wasn't large—just a key made of copper, warm even in the cold rain, with grooves that didn't match any lock she'd ever seen. She'd found it three weeks ago tucked inside an old library book, the kind that smelled like dust and dried flowers. Since then, odd things happened when she touched it. Streetlights blinked. Old radios woke up. Once, the elevator in her building stopped between floors and played a song from fifty years ago, even though there was no radio inside.

Mira descended the steps.

The platform was quieter than quiet. The kind of quiet that felt intentional.

At the edge of the tracks, a single bench sat under a leaking ceiling tile. Water fell in slow droplets, each one loud as a metronome. A vending machine at the far wall glowed with dead-eyed patience, offering snacks that looked older than the station itself.

And then she saw him.

A boy—no, a young man, maybe her age—was sitting on the bench with his hands folded neatly in his lap. His hair was dark and damp at the ends, as if he'd come in from the rain and never bothered to dry. A canvas messenger bag rested beside him like a loyal dog. He looked up as Mira approached, and his eyes flashed with recognition that made her stomach drop.

"You're late," he said.

Mira stopped a safe distance away. "Do I know you?"

He blinked, surprised—then covered it quickly, like someone shutting a door. "Not yet."

The answer was wrong in a way that made the air feel thinner.

Mira's pulse hammered against the copper key in her pocket. "You wrote the letter."

"I did." He nodded toward the tracks. "The last train doesn't come down here. That's why it's safe."

"Safe from who?"

He hesitated, and for a moment his confident posture slipped. "From the people who think the key belongs to them."

Mira let out a short laugh that didn't reach her throat. "That's comforting."

"I'm not trying to scare you," he said, and his voice softened. "I'm trying to keep you alive."

A gust of wind ran along the tracks like a hand dragging nails over steel. Mira flinched. Somewhere above, a door slammed. The sound echoed too long.

The young man stood. He was taller than Mira expected, and when he moved, it was with the careful precision of someone who had practiced being calm. "My name is Jonah."

Mira didn't offer her own. "Why me?"

Jonah took a step, then stopped himself as if remembering an invisible boundary. "Because the key chose you."

"That's not—" Mira started, but the word caught. She could feel it now: the copper warmth, pulsing faintly, almost like a heartbeat.

Jonah noticed her expression. "You've felt it."

Mira's lips pressed together. Lying seemed pointless. "Sometimes. It... reacts."

"It's waking up," Jonah said. He glanced at the shadows beyond the vending machine, then back to her. "And when it wakes up, it calls."

A slow drip. Another. Mira forced herself to breathe evenly. "Calls who?"

Jonah's eyes flicked upward again, like he expected the ceiling to answer. "Anyone who's listening."

Mira pulled her hand from her pocket and held up the key. Under the flickering lights, the copper surface looked almost alive, catching and releasing the light in soft waves.

Jonah's expression tightened with something like relief. "Good. You brought it."

"Don't get excited," Mira said. "I brought it because I want to know what's happening. And because your letter sounded like a threat."

"It wasn't." Jonah's voice sharpened. "But if you hadn't come, it might have become one."

Mira's fingers curled around the key. "So talk."

Jonah opened his mouth—and then froze.

Mira saw it too: the vending machine's glow flickered, dimmed, then surged. The station lights above the platform blinked in sequence, like a message being tapped out. The dripping water paused mid-fall for a fraction of a second, hanging like a bead of glass, then dropped all at once.

The air changed. The quiet stopped being empty and started being crowded.

Jonah whispered, "They found us."

Mira's throat went dry. "Who?"

Before Jonah could answer, the key burned hot in Mira's hand. She gasped and nearly dropped it. The copper grooves shimmered, lines rearranging themselves with a soft metallic scrape, as if the key were turning into something else.

Then, from the tunnel, a sound rolled toward them.

Not a train. Not exactly.

A low hum, like a distant engine, layered with a thin, high tone that made Mira's teeth ache. The tracks began to vibrate. Dust lifted from the ground in trembling puffs.

Jonah grabbed Mira's wrist—not hard, but urgent. "Listen to me. When I say run, you run. Do you understand?"

Mira pulled back instinctively. "You don't get to—"

"Mira." He said her name.

She stared. "How do you know—"

"No time." Jonah's eyes were locked on the tunnel. "They use sound first. It disorients you. Don't look into the dark. If you see the light, close your eyes."

"Light?" Mira echoed, but the hum grew louder, pressing into her chest.

Somewhere above them, an old announcement speaker sparked back to life. It crackled, then produced a warped, cheerful voice: "Now arriving... now arriving..."

The words dissolved into static.

Mira's heart slammed. "Is that the train?"

Jonah shook his head. "It's bait."

The hum became a roar. The tunnel spilled a faint glow, pale and cold, like moonlight in a freezer. The vending machine's screen flashed symbols Mira didn't recognize—circles and lines that felt strangely familiar, like a half-remembered dream.

Mira squeezed the key. It pulsed. Warm. Then warmer.

Jonah leaned close, voice low and steady despite the chaos. "The key opens doors that aren't supposed to exist. Tonight it wants to open one. You can either control it, or it will control you."

Mira's breath came in sharp bursts. "I don't know how."

"You will," Jonah said. "But not here."

The pale light surged forward. Mira's vision began to swim, as if the air itself had turned to water. She felt the urge to stare into the tunnel, to step toward it, to follow the sound like a lullaby.

Jonah's grip tightened. "Mira. Close your eyes. Now."

Mira hesitated—then snapped her eyes shut.

Behind her eyelids, the hum turned into voices. Not words—feelings. Curiosity. Hunger. A promise of answers if she just opened her hand.

The key throbbed once, twice.

Jonah pulled her backward, and Mira stumbled. Her shoulder hit the bench. Water splashed. The drip-drip-drip became a chaotic splash-splash-splash as if the ceiling had finally given up.

"Run!" Jonah shouted.

Mira's eyes flew open. The platform lights were strobing. The tunnel was glowing brighter, spilling a shape that was not a train but wanted to be one.

Jonah shoved the messenger bag into Mira's arms. "Take it!"

"What—"

"No questions!" Jonah's face was lit by the cold glow, and his expression was fierce and terrified all at once. "Go up the stairs. Don't stop. Don't look back."

Mira clutched the bag. The key was still in her other hand, burning like a coal. "Come with me!"

Jonah shook his head once, sharply. "I'll slow you down."

"That's—" Mira began, anger flaring, but the roar from the tunnel swallowed her words.

She turned and ran.

Her boots slapped wet concrete. The stairs seemed longer than they had on the way down. Above, the station doors sighed again, and the normal world felt impossibly far away.

Halfway up, Mira risked a glance over her shoulder.

Jonah stood at the platform's edge, facing the tunnel. The cold light wrapped around him like fog. He lifted his hand as if he was holding something invisible, steadying it.

Then he looked directly at Mira, and even from that distance she saw it: an apology, sharp as glass.

Mira's chest tightened. She turned away and sprinted the last steps.

At the top, the station hallway stretched empty, lights buzzing. The rain was louder here, pounding the roof like impatient fingers. Mira didn't stop until she burst through the entrance doors into the night air.

She stumbled into the rain, gasping, the messenger bag heavy against her ribs.

The key's heat faded slowly, like an ember dying. Mira looked down at it, drenched, shaking, heart racing.

For the first time, she noticed the tiny engraving along its side—letters she hadn't seen before.

It read: PLATFORM NINE, DOOR ONE.

Mira stared at the words until they blurred with the rain.

Somewhere behind her, deep in the station, the hum cut off suddenly—like a switch flipped.

The silence that followed felt worse than the noise.

Mira tightened her grip on the key and the bag and stepped into the streetlights, knowing she had just crossed into a story that wouldn't let her go.
"""

print(f"📖 Chapter loaded: {len(CHAPTER_TEXT)} characters, approximately {len(CHAPTER_TEXT.split())} words")

# =========================
# LOAD TTS MODELS
# =========================
print("\n" + "=" * 60)
print("🔄 LOADING TTS MODELS")
print("=" * 60)

jonah_tts = None
print("Loading Facebook MMS TTS for Jonah...")
try:
    jonah_tts = pipeline(
        "text-to-speech",
        model="facebook/mms-tts-eng",
        device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    )
    print("✅ Facebook MMS loaded successfully!")
except Exception as e:
    print(f"⚠️  Could not load Facebook MMS: {e}")
    print("Using gTTS fallback for Jonah")

print("\n" + "=" * 60)
print("🎭 VOICE ASSIGNMENTS")
print("=" * 60)
print("• Narrator: gTTS UK English (British male)")
print("• Jonah: Facebook MMS model (American male)")
print("• Mira: gTTS US English (American female)")
print("=" * 60)

# =========================
# IMPROVED TEXT PROCESSING - KEEPS DIALOGUE WITH NARRATION
# =========================
def process_dialogue_blocks(text):
    """
    Process text into blocks where dialogue and its narration stay together
    Example: '"Hello," he said.' becomes one block
    """
    blocks = []
    
    # Split by paragraphs first
    paragraphs = text.strip().split('\n\n')
    
    for para_idx, para in enumerate(paragraphs):
        if not para.strip():
            continue
            
        # Check for title/heading
        if para.startswith("Title:") or para.startswith("Chapter"):
            blocks.append({
                'type': 'heading',
                'content': para.strip(),
                'speaker': 'narrator'
            })
            continue
        
        # Process each paragraph
        lines = para.split('\n')
        for line in lines:
            if not line.strip():
                continue
                
            # Look for dialogue patterns in the line
            # Pattern 1: Dialogue followed by narration (e.g., '"Hello," he said.')
            pattern1 = re.findall(r'"([^"]*)"\s*([^.]*\.)', line)
            # Pattern 2: Narration followed by dialogue (e.g., 'He said, "Hello."')
            pattern2 = re.findall(r'([^"]*)\s*"([^"]*)"', line)
            
            if pattern1:
                # Dialogue first, then narration
                for dialogue, narration in pattern1:
                    if dialogue.strip():
                        blocks.append({
                            'type': 'dialogue_block',
                            'dialogue': dialogue.strip(),
                            'narration': narration.strip(),
                            'order': 'dialogue_first',
                            'original': f'"{dialogue}" {narration}'
                        })
            elif pattern2:
                # Narration first, then dialogue
                for narration, dialogue in pattern2:
                    if dialogue.strip():
                        blocks.append({
                            'type': 'dialogue_block',
                            'dialogue': dialogue.strip(),
                            'narration': narration.strip(),
                            'order': 'narration_first',
                            'original': f'{narration} "{dialogue}"'
                        })
            else:
                # No dialogue, pure narration
                blocks.append({
                    'type': 'narration',
                    'content': line.strip(),
                    'speaker': 'narrator'
                })
    
    return blocks

# =========================
# SPEAKER DETECTION FOR DIALOGUE BLOCKS
# =========================
def detect_speaker_in_block(block):
    """Determine who is speaking based on narration context"""
    if block['type'] != 'dialogue_block':
        return 'narrator'
    
    narration = block['narration'].lower()
    
    # Look for character names in narration
    if "mira" in narration or "she " in narration or "her " in narration:
        return 'mira'
    elif "jonah" in narration or "he " in narration or "his " in narration:
        return 'jonah'
    
    # Look for dialogue tags
    dialogue_tags = {
        'mira': ['mira said', 'said mira', 'mira whispered', 'whispered mira',
                'mira asked', 'asked mira', 'mira replied', 'replied mira'],
        'jonah': ['jonah said', 'said jonah', 'jonah whispered', 'whispered jonah',
                 'jonah asked', 'asked jonah', 'jonah replied', 'replied jonah']
    }
    
    for speaker, tags in dialogue_tags.items():
        for tag in tags:
            if tag in narration:
                return speaker
    
    # Look for gender pronouns
    if " she " in f" {narration} " or " her " in f" {narration} ":
        return 'mira'
    elif " he " in f" {narration} " or " his " in f" {narration} ":
        return 'jonah'
    
    # Default based on conversation flow
    return 'jonah'  # Default to Jonah if uncertain

# =========================
# EMOTION DETECTION FOR BLOCKS
# =========================
def detect_emotion_for_block(block):
    """Detect emotion from dialogue and narration"""
    if block['type'] == 'dialogue_block':
        dialogue = block['dialogue'].lower()
        narration = block['narration'].lower()
        combined = f"{dialogue} {narration}"
        
        # Check for whispering
        if any(word in narration for word in ['whispered', 'murmured', 'softly', 'quietly']):
            return 'whispering'
        
        # Check for urgency/fear
        urgent_words = ['urgent', 'run', 'quick', 'now', 'hurry', 'fast']
        if any(word in combined for word in urgent_words):
            return 'urgent'
        
        # Check for fear
        fear_words = ['fear', 'afraid', 'scared', 'panic', 'terror']
        if any(word in combined for word in fear_words):
            return 'fearful'
        
        # Check for calm
        if 'calm' in narration or 'calmly' in narration:
            return 'calm'
        
        # Check punctuation in dialogue
        if '!' in block['dialogue']:
            return 'excited'
        if '?' in block['dialogue']:
            return 'questioning'
        
        return 'neutral'
    
    else:
        # For narration or heading
        content = block.get('content', '').lower()
        if '!' in content:
            return 'excited'
        return 'neutral'

# =========================
# VOICE GENERATION FUNCTIONS
# =========================
def generate_narrator_voice(text, emotion="neutral"):
    """Generate narrator voice with gTTS UK"""
    try:
        tmp_path = f"{AUDIO_DIR}/temp_narrator.mp3"
        
        # Adjust speed based on emotion
        slow = emotion in ["calm", "whispering"]
        
        tts = gTTS(
            text=text,
            lang='en',
            tld='co.uk',
            slow=slow
        )
        tts.save(tmp_path)
        
        audio = AudioSegment.from_mp3(tmp_path)
        
        # Apply emotion-based adjustments
        if emotion == "whispering":
            audio = audio - 6
        elif emotion == "urgent" or emotion == "fearful":
            audio = audio.speedup(playback_speed=1.1)
        
        # Convert to numpy
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        if audio.channels == 2:
            samples = samples.reshape(-1, 2).mean(axis=1)
        samples = samples / 32768.0
        
        os.remove(tmp_path)
        return samples
        
    except Exception as e:
        print(f"⚠️ Narrator error: {e}")
        return np.zeros(int(SAMPLE_RATE * 0.5), dtype=np.float32)

def generate_jonah_voice(text, emotion="neutral"):
    """Generate Jonah's voice with Facebook MMS or fallback"""
    try:
        if jonah_tts is not None:
            # Use Facebook MMS
            result = jonah_tts(text, forward_params={"speaker_id": 0})
            
            if 'audio' in result:
                audio_data = result['audio']
                
                # Ensure mono
                if isinstance(audio_data, np.ndarray) and audio_data.ndim == 2:
                    audio_data = audio_data.mean(axis=0)
                
                # Resample if needed (MMS is 16000 Hz)
                if SAMPLE_RATE != 16000:
                    ratio = SAMPLE_RATE / 16000
                    new_length = int(len(audio_data) * ratio)
                    audio_data = np.interp(
                        np.linspace(0, len(audio_data)-1, new_length),
                        np.arange(len(audio_data)),
                        audio_data
                    )
                
                # Emotion adjustments
                if emotion == "whispering":
                    audio_data = audio_data * 0.7
                elif emotion == "urgent":
                    audio_data = audio_data * 1.2
                
                return audio_data
        
        # Fallback to gTTS
        return generate_gtts_voice(text, "com", emotion, pitch_adjust=-30)
        
    except Exception as e:
        print(f"⚠️ Jonah voice error: {e}")
        return generate_gtts_voice(text, "com", "neutral", pitch_adjust=-30)

def generate_mira_voice(text, emotion="neutral"):
    """Generate Mira's voice with gTTS US"""
    return generate_gtts_voice(text, "com", emotion, pitch_adjust=20)

def generate_gtts_voice(text, tld, emotion, pitch_adjust=0):
    """Generic gTTS voice generator"""
    try:
        tmp_path = f"{AUDIO_DIR}/temp_gtts.mp3"
        
        # Speed adjustment
        slow = emotion in ["calm", "whispering"]
        
        tts = gTTS(
            text=text,
            lang='en',
            tld=tld,
            slow=slow
        )
        tts.save(tmp_path)
        
        audio = AudioSegment.from_mp3(tmp_path)
        
        # Emotion adjustments
        if emotion == "whispering":
            audio = audio - 8
        elif emotion == "urgent" or emotion == "fearful":
            audio = audio + 2
        
        # Pitch adjustment (simulated with speed)
        if pitch_adjust != 0:
            speed_factor = 1.0 - (pitch_adjust / 200)
            if speed_factor != 1.0:
                new_frame_rate = int(audio.frame_rate * speed_factor)
                audio = audio._spawn(
                    audio.raw_data,
                    overrides={"frame_rate": new_frame_rate}
                ).set_frame_rate(audio.frame_rate)
        
        # Convert to numpy
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        if audio.channels == 2:
            samples = samples.reshape(-1, 2).mean(axis=1)
        samples = samples / 32768.0
        
        os.remove(tmp_path)
        return samples
        
    except Exception as e:
        print(f"⚠️ gTTS error: {e}")
        return np.zeros(int(SAMPLE_RATE * 0.5), dtype=np.float32)

# =========================
# AUDIO PROCESSING
# =========================
def create_silence(duration_ms):
    """Create silence"""
    return np.zeros(int(SAMPLE_RATE * duration_ms / 1000), dtype=np.float32)

def save_audio(path, audio_data):
    """Save audio to WAV file"""
    audio_data = np.clip(audio_data, -1.0, 1.0)
    audio_data = audio_data.astype(np.float32)
    sf.write(path, audio_data, SAMPLE_RATE, subtype='PCM_16')

# =========================
# PROCESS DIALOGUE BLOCKS WITH PROPER TIMING
# =========================
def process_dialogue_block(block, last_speaker=None):
    """
    Process a dialogue block with proper timing between dialogue and narration
    Returns: list of audio segments
    """
    audio_segments = []
    
    speaker = detect_speaker_in_block(block)
    emotion = detect_emotion_for_block(block)
    
    print(f"  💬 {speaker.upper()}: '{block['dialogue'][:40]}...'")
    print(f"     Narration: '{block['narration'][:40]}...'")
    print(f"     Emotion: {emotion}")
    
    # Generate dialogue audio
    if speaker == 'jonah':
        dialogue_audio = generate_jonah_voice(block['dialogue'], emotion)
    else:  # mira
        dialogue_audio = generate_mira_voice(block['dialogue'], emotion)
    
    # Generate narration audio
    narration_audio = generate_narrator_voice(block['narration'], emotion)
    
    # Determine order and timing
    if block['order'] == 'dialogue_first':
        # Dialogue first, then narration (e.g., "Hello," he said.)
        audio_segments.append(dialogue_audio)
        audio_segments.append(create_silence(150))  # Short pause before narration
        audio_segments.append(narration_audio)
    else:
        # Narration first, then dialogue (e.g., He said, "Hello.")
        audio_segments.append(narration_audio)
        audio_segments.append(create_silence(150))  # Short pause before dialogue
        audio_segments.append(dialogue_audio)
    
    return audio_segments

# =========================
# MAIN PROCESSING
# =========================
def process_chapter_with_better_flow():
    """Process chapter with improved dialogue-narration flow"""
    print("\n" + "=" * 60)
    print("📖 PROCESSING WITH IMPROVED DIALOGUE FLOW")
    print("=" * 60)
    
    # Step 1: Process text into dialogue blocks
    print("\n1. Analyzing text structure...")
    blocks = process_dialogue_blocks(CHAPTER_TEXT)
    print(f"   Found {len(blocks)} blocks")
    
    # Step 2: Process each block
    print("\n2. Generating audio...")
    all_audio_segments = []
    
    for i, block in enumerate(blocks):
        print(f"\n[{i+1}/{len(blocks)}] Processing block:")
        
        if block['type'] == 'heading':
            print(f"  📝 HEADING: {block['content'][:50]}...")
            audio = generate_narrator_voice(block['content'], "neutral")
            all_audio_segments.append(audio)
            all_audio_segments.append(create_silence(800))  # Longer pause after headings
            
        elif block['type'] == 'narration':
            print(f"  📖 NARRATION: {block['content'][:50]}...")
            emotion = detect_emotion_for_block(block)
            audio = generate_narrator_voice(block['content'], emotion)
            all_audio_segments.append(audio)
            all_audio_segments.append(create_silence(500))  # Standard pause
            
        elif block['type'] == 'dialogue_block':
            # Process dialogue block with proper timing
            block_audio = process_dialogue_block(block)
            all_audio_segments.extend(block_audio)
            all_audio_segments.append(create_silence(400))  # Pause after dialogue block
        
        # Update progress
        percent = ((i + 1) / len(blocks)) * 100
        print(f"     Progress: {percent:.1f}%", end='\r')
    
    # Step 3: Combine audio
    print("\n\n3. Combining audio...")
    valid_segments = [seg for seg in all_audio_segments if len(seg) > 0]
    
    if not valid_segments:
        print("❌ No valid audio generated!")
        return None
    
    final_audio = np.concatenate(valid_segments)
    
    # Step 4: Normalize
    max_val = np.max(np.abs(final_audio))
    if max_val > 0:
        final_audio = final_audio * (0.8 / max_val)
    
    # Step 5: Save
    output_path = f"{AUDIO_DIR}/chapter_3_improved_flow.wav"
    save_audio(output_path, final_audio)
    
    # Statistics
    duration = len(final_audio) / SAMPLE_RATE
    minutes = int(duration // 60)
    seconds = int(duration % 60)
    
    print(f"\n" + "=" * 60)
    print("✅ PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"📊 Statistics:")
    print(f"   Blocks processed: {len(blocks)}")
    print(f"   Total duration: {minutes}:{seconds:02d}")
    print(f"   Saved to: {output_path}")
    
    return output_path

# =========================
# TEST SPECIFIC DIALOGUE EXAMPLES
# =========================
def test_dialogue_flow():
    """Test specific dialogue examples from the chapter"""
    print("\n" + "=" * 60)
    print("🔍 TESTING DIALOGUE FLOW PATTERNS")
    print("=" * 60)
    
    test_examples = [
        # Dialogue first, then narration
        '"You\'re late," he said.',
        
        # Dialogue with question
        '"Do I know you?" Mira asked.',
        
        # Narration first, then dialogue
        'Jonah whispered, "They found us."',
        
        # Complex example from the chapter
        '"Listen to me," Jonah said, his voice urgent. "When I say run, you run."'
    ]
    
    for example in test_examples:
        print(f"\nExample: {example}")
        
        # Process as a block
        if '"' in example:
            # Try to extract dialogue and narration
            matches = re.findall(r'"([^"]*)"\s*([^.]*\.)', example)
            if matches:
                dialogue, narration = matches[0]
                block = {
                    'type': 'dialogue_block',
                    'dialogue': dialogue,
                    'narration': narration,
                    'order': 'dialogue_first',
                    'original': example
                }
            else:
                matches = re.findall(r'([^"]*)\s*"([^"]*)"', example)
                if matches:
                    narration, dialogue = matches[0]
                    block = {
                        'type': 'dialogue_block',
                        'dialogue': dialogue,
                        'narration': narration,
                        'order': 'narration_first',
                        'original': example
                    }
                else:
                    print("  Could not parse example")
                    continue
        
        # Determine speaker
        speaker = detect_speaker_in_block(block)
        print(f"  Detected speaker: {speaker}")
        
        # Process the block
        test_audio = process_dialogue_block(block)
        
        # Combine and play
        if test_audio:
            combined = np.concatenate(test_audio)
            temp_path = f"{AUDIO_DIR}/test_flow.wav"
            save_audio(temp_path, combined)
            print("  Playing...")
            display(Audio(temp_path))

# =========================
# MAIN EXECUTION
# =========================
if __name__ == "__main__":
    print("=" * 60)
    print("🎬 AUDIOBOOK: IMPROVED DIALOGUE-NARRATION FLOW")
    print("=" * 60)
    
    # First test dialogue flow patterns
    test_dialogue_flow()
    
    print("\n" + "=" * 60)
    print("📖 PROCESSING FULL CHAPTER")
    print("=" * 60)
    print("This will process the entire chapter with improved flow between")
    print("dialogue and narration. It will take some time...")
    print("=" * 60)
    
    # Process the full chapter
    chapter_audio = process_chapter_with_better_flow()
    
    if chapter_audio:
        # Create and play a preview
        print("\n" + "=" * 60)
        print("▶️  PLAYING PREVIEW")
        print("=" * 60)
        
        audio_data, _ = sf.read(chapter_audio)
        
        # Create a 90-second preview from an interesting section
        preview_start = 60 * SAMPLE_RATE  # Start at 1 minute
        preview_duration = 90 * SAMPLE_RATE  # 90 seconds
        preview_end = min(preview_start + preview_duration, len(audio_data))
        
        if preview_end > preview_start:
            preview_audio = audio_data[preview_start:preview_end]
            preview_path = f"{AUDIO_DIR}/flow_preview_90s.wav"
            save_audio(preview_path, preview_audio)
            
            print("Playing 90-second preview (showing dialogue-narration flow)...")
            display(Audio(preview_path))
            
            print(f"\n📁 Files created:")
            print(f"   Full chapter: {chapter_audio}")
            print(f"   90-second preview: {preview_path}")
        else:
            print("Playing full audio (chapter is short)...")
            display(Audio(chapter_audio))
        
        print("\n" + "=" * 60)
        print("🎯 KEY IMPROVEMENTS IN THIS VERSION:")
        print("=" * 60)
        print("1. Dialogue and narration stay together as single blocks")
        print("2. Proper short pauses (150ms) between dialogue and 'he said/she said'")
        print("3. Better speaker detection from narration context")
        print("4. Emotion detection applies to both dialogue and narration")
        print("5. More natural flow for lines like: 'Hello,' he said.")
        print("=" * 60)
