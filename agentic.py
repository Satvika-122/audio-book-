import os
import re
import time
import uuid
import warnings
import numpy as np
import soundfile as sf
from gtts import gTTS
from pydub import AudioSegment
from transformers import pipeline
from IPython.display import Audio, display
import tempfile
import json
from datetime import datetime

warnings.filterwarnings('ignore')

class AgentMessage:
    def __init__(self, sender: str, receiver: str, content: dict, msg_type: str = "task"):
        self.id = str(uuid.uuid4())[:8]
        self.sender = sender
        self.receiver = receiver
        self.content = content
        self.type = msg_type
        self.timestamp = time.time()
        self.status = "pending"
        self.result = None

class Agent:
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.inbox = []

    def log(self, msg: str):
        print(f"🤖 {self.name}: {msg}")

    def process(self, message: AgentMessage) -> AgentMessage:
        raise NotImplementedError

    def receive(self, message: AgentMessage) -> AgentMessage:
        self.log(f"📩 Received {message.type} from {message.sender}")
        self.inbox.append(message)
        result = self.process(message)
        message.status = "processed"
        return result

    def send(self, receiver: 'Agent', message: AgentMessage) -> AgentMessage:
        self.log(f"📤 Sending {message.type} to {receiver.name}")
        return receiver.receive(message)

class OrchestratorAgent(Agent):
    """Coordinates the workflow between agents"""
    def __init__(self):
        super().__init__("Orchestrator", "Workflow Coordinator")
        self.agents = {}

    def register_agent(self, agent: Agent):
        self.agents[agent.name] = agent
        self.log(f"Registered agent: {agent.name} ({agent.role})")

    def process(self, message: AgentMessage) -> AgentMessage:
        # Start the pipeline
        self.log("🚀 Starting audio generation pipeline...")

        # Step 1: Analyze text
        analyzer = self.agents["TextAnalyzer"]
        analysis_result = self.send(analyzer, message)

        # Step 2: Synthesize voices
        synth = self.agents["VoiceSynthesis"]
        synth_result = self.send(synth, analysis_result)

        # Step 3: Create evaluation message with ALL data
        evaluation_message = AgentMessage(
            self.name,
            "Evaluator",
            {
                # Original input
                "text": message.content.get("text", ""),
                # Text analysis results
                "blocks": analysis_result.content.get("blocks", []),
                "analysis_details": {
                    "total_blocks": len(analysis_result.content.get("blocks", [])),
                    "speaker_distribution": self._get_speaker_distribution(
                        analysis_result.content.get("blocks", [])
                    )
                },
                # Voice synthesis results
                "audio_data": synth_result.content.get("audio_data"),
                "sample_rate": synth_result.content.get("sample_rate", 24000),
                "audio_duration": len(synth_result.content.get("audio_data", [])) / synth_result.content.get("sample_rate", 24000) if synth_result.content.get("audio_data") is not None else 0
            },
            "evaluation_request"
        )

        # Step 4: Evaluate quality
        evaluator = self.agents["Evaluator"]
        evaluation_result = self.send(evaluator, evaluation_message)

        # Step 5: Combine all results
        combined_content = {
            **synth_result.content,
            **evaluation_result.content
        }

        return AgentMessage(
            self.name,
            message.sender,
            combined_content,
            "complete_result"
        )

    def _get_speaker_distribution(self, blocks):
        """Helper to calculate speaker distribution"""
        distribution = {}
        for block in blocks:
            speaker = block.get('speaker', 'unknown')
            distribution[speaker] = distribution.get(speaker, 0) + 1
        return distribution

class TextAnalyzerAgent(Agent):
    """Advanced text analyzer with better conversation tracking"""
    def __init__(self):
        super().__init__("TextAnalyzer", "Text Analysis")

    def process(self, message: AgentMessage) -> AgentMessage:
        text = message.content.get("text", "")
        blocks = []

        # Split into paragraphs for context
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]

        current_speaker = None
        conversation_context = []

        for para in paragraphs:
            # Process each paragraph
            para_blocks = self._process_paragraph(para, current_speaker, conversation_context)

            if para_blocks:
                # Update conversation context
                for block in para_blocks:
                    if block['speaker'] in ['jonah', 'mira']:
                        conversation_context.append(block['speaker'])
                        current_speaker = block['speaker']

                blocks.extend(para_blocks)

        self.log(f"📝 Analyzed text into {len(blocks)} blocks")

        # Detailed logging
        self.log("\n📋 SPEAKER ANALYSIS:")
        for i, block in enumerate(blocks):
            speaker_display = block['speaker'].upper() if block['speaker'] in ['jonah', 'mira'] else 'NARRATOR'
            text_preview = block['text'][:80].replace('\n', ' ')
            if len(block['text']) > 80:
                text_preview += "..."
            self.log(f"  {i+1:3d}. {speaker_display:8s}: {text_preview}")

        return AgentMessage(
            self.name,
            "Orchestrator",
            {"blocks": blocks},
            "analysis_result"
        )

    def _process_paragraph(self, paragraph: str, current_speaker: str, conversation_context: list) -> list:
        """Process a single paragraph with conversation tracking"""
        blocks = []

        # Extract all quoted dialogue with their positions
        dialogue_matches = list(re.finditer(r'"([^"]+)"', paragraph))

        if not dialogue_matches:
            # No dialogue, just narration
            clean_narration = self._clean_narration(paragraph)
            if clean_narration:
                blocks.append({'speaker': 'narrator', 'text': clean_narration})
            return blocks

        # Process the paragraph piece by piece
        last_end = 0

        for i, match in enumerate(dialogue_matches):
            dialogue_text = match.group(1)
            match_start = match.start()
            match_end = match.end()

            # Get text before this dialogue (narration)
            narration_before = paragraph[last_end:match_start].strip()
            if narration_before:
                clean_narration = self._clean_narration(narration_before)
                if clean_narration:
                    blocks.append({'speaker': 'narrator', 'text': clean_narration})

            # Determine speaker for this dialogue
            speaker = self._determine_speaker_for_dialogue(
                paragraph, match, dialogue_matches, i,
                current_speaker, conversation_context, dialogue_text
            )

            # Add dialogue block
            blocks.append({'speaker': speaker, 'text': dialogue_text})

            # Update for next iteration
            last_end = match_end

        # Get any narration after the last dialogue
        narration_after = paragraph[last_end:].strip()
        if narration_after:
            clean_narration = self._clean_narration(narration_after)
            if clean_narration:
                blocks.append({'speaker': 'narrator', 'text': clean_narration})

        return blocks

    def _determine_speaker_for_dialogue(self, paragraph: str, match: re.Match,
                                       all_matches: list, match_index: int,
                                       current_speaker: str, conversation_context: list,
                                       dialogue_text: str) -> str:
        """Intelligently determine speaker for a specific dialogue"""

        # STRONG CLUES FROM DIALOGUE CONTENT:

        # 1. Self-introduction patterns
        if re.search(r'\bmy name (is|s) jonah\b', dialogue_text.lower()):
            return 'jonah'
        if re.search(r'\bmy name (is|s) mira\b', dialogue_text.lower()):
            return 'mira'

        # 2. Direct address patterns
        if dialogue_text.lower().startswith('mira.') or 'mira,' in dialogue_text.lower():
            return 'jonah'  # Jonah addressing Mira

        # 3. Dialogue contains character's own name
        if 'jonah' in dialogue_text.lower() and 'mira' not in dialogue_text.lower():
            # If someone is talking about Jonah, it's probably Mira
            return 'mira'
        if 'mira' in dialogue_text.lower() and 'jonah' not in dialogue_text.lower():
            # If someone is talking about Mira, it's probably Jonah
            return 'jonah'

        # 4. Command patterns (Jonah tends to give commands)
        command_patterns = [
            r'^listen to me',
            r'^run',
            r'^close your eyes',
            r'^take it',
            r'^go up',
            r'^don\'t stop',
            r'^don\'t look',
            r'^do you understand'
        ]
        for pattern in command_patterns:
            if re.search(pattern, dialogue_text.lower()):
                return 'jonah'

        # 5. Question patterns (Mira asks more questions)
        question_words = ['who', 'what', 'where', 'why', 'when', 'how', 'do i', 'is that']
        if any(dialogue_text.lower().startswith(word) for word in question_words) or \
           dialogue_text.strip().endswith('?'):
            # But check context - Jonah also asks questions
            context_window = paragraph[max(0, match.start()-100):min(len(paragraph), match.end()+100)]
            if 'jonah' in context_window.lower() and 'mira' in context_window.lower():
                # If both names in context, use conversation flow
                if len(conversation_context) > 0:
                    last_speaker = conversation_context[-1] if conversation_context else None
                    return 'mira' if last_speaker == 'jonah' else 'jonah'

        # GET CONTEXT FROM NARRATION:
        context_start = max(0, match.start() - 150)
        context_end = min(len(paragraph), match.end() + 150)
        context = paragraph[context_start:context_end].lower()

        # Check for explicit speaker mentions
        jonah_mentioned = 'jonah' in context
        mira_mentioned = 'mira' in context

        # Look for speech verbs near the dialogue
        speech_verbs = ['said', 'asked', 'replied', 'answered', 'whispered',
                       'shouted', 'muttered', 'continued', 'added', 'explained']

        # Check 50 characters before and after the dialogue
        before_context = paragraph[max(0, match.start()-50):match.start()].lower()
        after_context = paragraph[match.end():min(len(paragraph), match.end()+50)].lower()

        # Check for "he/she [verb]" patterns
        for verb in speech_verbs:
            if re.search(rf'\bhe\s+{verb}\b', before_context) or \
               re.search(rf'\b{verb}\s+he\b', before_context) or \
               re.search(rf'\bhe\s+{verb}\b', after_context) or \
               re.search(rf'\b{verb}\s+he\b', after_context):
                return 'jonah'

            if re.search(rf'\bshe\s+{verb}\b', before_context) or \
               re.search(rf'\b{verb}\s+she\b', before_context) or \
               re.search(rf'\bshe\s+{verb}\b', after_context) or \
               re.search(rf'\b{verb}\s+she\b', after_context):
                return 'mira'

        # CONVERSATION FLOW LOGIC:
        if len(conversation_context) >= 2:
            # If we have conversation history, alternate speakers
            last_two = conversation_context[-2:] if len(conversation_context) >= 2 else []
            if len(last_two) == 2 and last_two[0] != last_two[1]:
                # Speakers were alternating, continue pattern
                return 'jonah' if conversation_context[-1] == 'mira' else 'mira'

        # FALLBACK: Use last known speaker if we have one
        if current_speaker:
            # Alternate in conversation
            return 'jonah' if current_speaker == 'mira' else 'mira'

        # FINAL FALLBACK: Guess based on name mentions
        if jonah_mentioned and not mira_mentioned:
            return 'jonah'
        elif mira_mentioned and not jonah_mentioned:
            return 'mira'
        elif jonah_mentioned and mira_mentioned:
            # Both mentioned - use conversation start pattern
            # First dialogue after Mira's name is usually Mira
            # First dialogue after Jonah's name is usually Jonah
            mira_pos = paragraph.lower().find('mira')
            jonah_pos = paragraph.lower().find('jonah')

            if mira_pos >= 0 and jonah_pos >= 0:
                # Both names appear, check which comes first near this dialogue
                if abs(match.start() - mira_pos) < abs(match.start() - jonah_pos):
                    return 'mira'
                else:
                    return 'jonah'

        # ULTIMATE FALLBACK: Narrator (should rarely happen)
        return 'narrator'

    def _clean_narration(self, text: str) -> str:
        """Clean narration text by removing dialogue attribution"""
        if not text.strip():
            return ""

        # Remove common dialogue attribution patterns
        patterns = [
            # "Jonah said" or "said Jonah"
            r'\b(jonah|mira)\s+\b(said|asked|replied|answered|whispered|shouted|exclaimed|muttered|continued|added|explained)\b',
            r'\b(said|asked|replied|answered|whispered|shouted|exclaimed|muttered|continued|added|explained)\s+\b(jonah|mira)\b',
            # "he said" or "said he"
            r'\b(he|she)\s+\b(said|asked|replied|answered|whispered|shouted|exclaimed|muttered|continued|added|explained)\b',
            r'\b(said|asked|replied|answered|whispered|shouted|exclaimed|muttered|continued|added|explained)\s+\b(he|she)\b',
            # Leading/trailing punctuation
            r'^\s*[,\-—:;]\s*',
            r'\s*[,\-—:;]\s*$',
            # Multiple spaces
            r'\s+'
        ]

        cleaned = text.strip()
        for pattern in patterns:
            cleaned = re.sub(pattern, ' ', cleaned, flags=re.IGNORECASE)

        # Final cleanup
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.strip(' ,.-—:;')

        return cleaned

class VoiceSynthesisAgent(Agent):
    def __init__(self, sample_rate: int = 24000):
        super().__init__("VoiceSynthesis", "Voice Generation")
        self.sample_rate = sample_rate
        self.models_loaded = False

    def load_models(self):
        """Lazy loading of models"""
        try:
            self.jonah_model = pipeline(
                "text-to-speech",
                model="facebook/mms-tts-eng",
                device=-1  # Use CPU; change to 0 for GPU if available
            )
            self.log("✅ MMS model loaded for Jonah")
            self.models_loaded = True
        except Exception as e:
            self.jonah_model = None
            self.log(f"⚠️ Could not load MMS model for Jonah: {e}")
            self.log("⚠️ Falling back to gTTS for Jonah")
            self.models_loaded = True

    def process(self, message: AgentMessage) -> AgentMessage:
        if not self.models_loaded:
            self.load_models()

        blocks = message.content.get("blocks", [])
        audio_segments = []

        self.log(f"\n🎵 Synthesizing voices for {len(blocks)} blocks...")

        for i, block in enumerate(blocks):
            speaker = block["speaker"]
            text = block["text"]

            if not text.strip():
                continue

            # Show which voice is being used
            if speaker == 'jonah':
                voice_source = "MMS-TTS" if self.jonah_model else "gTTS (Australian)"
            elif speaker == 'mira':
                voice_source = "gTTS (British)"
            else:
                voice_source = "gTTS (US)"

            self.log(f"  {i+1:3d}. {speaker.upper():6s} [{voice_source:20s}]: {text[:70]}...")

            # Generate audio
            audio = self._generate_voice(speaker, text)

            if audio is not None and len(audio) > 0:
                audio_segments.append(audio)

                # Add pause between blocks
                if i < len(blocks) - 1:
                    next_block = blocks[i + 1]

                    # Determine pause length based on context
                    if block['speaker'] == 'narrator' and next_block['speaker'] != 'narrator':
                        pause = 0.15  # Short pause before dialogue
                    elif block['speaker'] != 'narrator' and next_block['speaker'] == 'narrator':
                        pause = 0.2   # Pause after dialogue
                    elif block['speaker'] == next_block['speaker']:
                        pause = 0.1   # Short pause for same speaker
                    else:
                        pause = 0.25  # Normal pause for speaker change

                    audio_segments.append(np.zeros(int(self.sample_rate * pause)))

        # Combine audio
        if audio_segments:
            # Remove last pause if present
            if audio_segments and np.all(audio_segments[-1] == 0):
                audio_segments = audio_segments[:-1]

            if audio_segments:
                combined = np.concatenate(audio_segments)
            else:
                combined = np.array([])
        else:
            combined = np.array([])

        duration = len(combined) / self.sample_rate if len(combined) > 0 else 0
        self.log(f"\n✅ Synthesis complete. Duration: {duration:.2f}s")

        return AgentMessage(
            self.name,
            "Orchestrator",
            {"audio_data": combined, "sample_rate": self.sample_rate},
            "synthesis_result"
        )

    def _generate_voice(self, speaker: str, text: str) -> np.ndarray:
        """Generate voice for specific speaker"""
        try:
            if speaker == "jonah":
                return self._generate_jonah_voice(text)
            elif speaker == "mira":
                return self._generate_mira_voice(text)
            else:  # narrator
                return self._generate_narrator_voice(text)
        except Exception as e:
            self.log(f"❌ Error generating {speaker} voice: {e}")
            return np.zeros(int(self.sample_rate * 0.5))

    def _generate_jonah_voice(self, text: str) -> np.ndarray:
        """Generate Jonah's voice - prioritize MMS, fallback to gTTS"""
        clean_text = self._clean_text_for_tts(text)

        # Try MMS-TTS first
        if self.jonah_model is not None:
            try:
                result = self.jonah_model(clean_text)
                audio = result["audio"]

                # Convert to numpy array if needed
                if not isinstance(audio, np.ndarray):
                    audio = np.array(audio)

                # Convert to mono if stereo
                if audio.ndim == 2:
                    audio = audio.mean(axis=0)

                # Normalize
                if len(audio) > 0:
                    max_val = np.max(np.abs(audio))
                    if max_val > 0:
                        audio = audio / max_val * 0.7

                # Resample to target rate
                if self.sample_rate != 16000:
                    ratio = self.sample_rate / 16000
                    new_len = int(len(audio) * ratio)
                    audio = np.interp(
                        np.linspace(0, len(audio)-1, new_len),
                        np.arange(len(audio)),
                        audio
                    )

                self.log(f"    ✓ Jonah: MMS-TTS used")
                return audio

            except Exception as e:
                self.log(f"    ⚠️ Jonah MMS failed, using gTTS fallback: {e}")

        # Fallback to gTTS with Australian accent
        return self._generate_gtts_voice(clean_text, "com.au")

    def _generate_mira_voice(self, text: str) -> np.ndarray:
        """Generate Mira's voice with British accent"""
        clean_text = self._clean_text_for_tts(text)
        return self._generate_gtts_voice(clean_text, "co.uk")

    def _generate_narrator_voice(self, text: str) -> np.ndarray:
        """Generate narrator voice with US accent"""
        clean_text = self._clean_text_for_tts(text)
        return self._generate_gtts_voice(clean_text, "com")

    def _generate_gtts_voice(self, text: str, tld: str) -> np.ndarray:
        """Generate voice using gTTS"""
        try:
            # Create temp file
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = tmp.name

            # Generate TTS
            tts = gTTS(text=text, lang="en", tld=tld, slow=False)
            tts.save(tmp_path)

            # Load and process audio
            audio = AudioSegment.from_mp3(tmp_path)
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)

            # Convert to mono if stereo
            if audio.channels == 2:
                samples = samples.reshape(-1, 2).mean(axis=1)

            # Normalize
            samples = samples / 32768.0

            # Resample if needed
            if audio.frame_rate != self.sample_rate:
                ratio = self.sample_rate / audio.frame_rate
                new_len = int(len(samples) * ratio)
                samples = np.interp(
                    np.linspace(0, len(samples)-1, new_len),
                    np.arange(len(samples)),
                    samples
                )

            # Cleanup
            os.unlink(tmp_path)

            return samples

        except Exception as e:
            self.log(f"❌ gTTS failed (TLD: {tld}): {e}")
            return np.zeros(int(self.sample_rate * 1.0))

    def _clean_text_for_tts(self, text: str) -> str:
        """Clean text for better TTS output"""
        replacements = {
            "you're": "you are",
            "don't": "do not",
            "can't": "cannot",
            "won't": "will not",
            "it's": "it is",
            "I'm": "I am",
            "he's": "he is",
            "she's": "she is",
            "that's": "that is",
            "what's": "what is",
            "who's": "who is",
            "where's": "where is"
        }

        clean_text = text
        for orig, repl in replacements.items():
            clean_text = clean_text.replace(orig, repl)

        return clean_text

class EvaluatorAgent(Agent):
    """Evaluates the quality of audiobook generation and provides feedback"""

    def __init__(self):
        super().__init__("Evaluator", "Quality Assessment")
        self.evaluation_history = []
        self.quality_thresholds = {
            'speaker_accuracy': 0.95,  # 95% speaker identification accuracy
            'audio_duration_match': 0.90,  # 90% expected vs actual duration match
            'block_count_match': 0.95,  # 95% expected vs actual blocks
            'voice_consistency': 0.98,  # 98% consistent voice usage
            'pacing_quality': 0.85,  # Subjective pacing score
        }

    def process(self, message: AgentMessage) -> AgentMessage:
        """Evaluate the audiobook generation process"""
        self.log("🔍 Starting evaluation...")

        # Collect evaluation data from different stages
        evaluation_data = self._collect_evaluation_data(message)

        # Run comprehensive evaluation
        evaluation_results = self._run_evaluation(evaluation_data)

        # Generate detailed report
        evaluation_report = self._generate_report(evaluation_results)

        # Store in history
        self.evaluation_history.append({
            'timestamp': datetime.now().isoformat(),
            'results': evaluation_results,
            'report': evaluation_report
        })

        # Log summary
        self._log_evaluation_summary(evaluation_results)

        return AgentMessage(
            self.name,
            "Orchestrator",
            {
                "evaluation_results": evaluation_results,
                "evaluation_report": evaluation_report,
                "recommendations": evaluation_results['recommendations'],
                "overall_score": evaluation_results['overall_score'],
                "passed_quality_check": evaluation_results['passed_quality_check']
            },
            "evaluation_result"
        )

    def _collect_evaluation_data(self, message: AgentMessage) -> dict:
        """Collect data from all agents for evaluation"""
        evaluation_data = {
            'input_analysis': {},
            'text_analysis_results': {},
            'voice_synthesis_results': {},
            'final_output': {},
            'timing_data': {},
            'agent_logs': {}
        }

        content = message.content

        # Extract all data from the message
        # 1. Text analysis data
        if 'blocks' in content:
            blocks = content.get('blocks', [])
            evaluation_data['text_analysis_results'] = {
                'total_blocks': len(blocks),
                'speaker_distribution': self._analyze_speaker_distribution(blocks),
                'block_samples': blocks[:5] if blocks else []
            }

        # 2. Audio synthesis data
        if 'audio_data' in content:
            audio_data = content.get('audio_data')
            sample_rate = content.get('sample_rate', 24000)
            evaluation_data['voice_synthesis_results'] = {
                'audio_length_samples': len(audio_data) if audio_data is not None else 0,
                'audio_duration_seconds': len(audio_data)/sample_rate if audio_data is not None else 0,
                'sample_rate': sample_rate,
                'audio_metadata': self._analyze_audio_metadata(audio_data) if audio_data is not None else {}
            }

        # 3. Original text data
        if 'text' in content:
            text = content.get('text', '')
            evaluation_data['input_analysis'] = {
                'text_length': len(text),
                'paragraph_count': text.count('\n\n') + 1,
                'estimated_speaking_time': self._estimate_speaking_time(text)
            }

        # 4. Analysis details if provided
        if 'analysis_details' in content:
            analysis_details = content.get('analysis_details', {})
            if 'total_blocks' in analysis_details and not evaluation_data['text_analysis_results']:
                evaluation_data['text_analysis_results'] = {
                    'total_blocks': analysis_details.get('total_blocks', 0),
                    'speaker_distribution': analysis_details.get('speaker_distribution', {}),
                    'block_samples': []
                }

        # Collect timing data
        evaluation_data['timing_data'] = {
            'processing_start': message.timestamp,
            'evaluation_time': time.time()
        }

        return evaluation_data

    def _run_evaluation(self, data: dict) -> dict:
        """Run comprehensive evaluation"""
        results = {
            'metrics': {},
            'scores': {},
            'issues': [],
            'strengths': [],
            'recommendations': [],
            'passed_quality_check': True
        }

        # 1. Text Analysis Evaluation
        text_eval = self._evaluate_text_analysis(data.get('text_analysis_results', {}))
        results['metrics'].update(text_eval['metrics'])
        results['scores']['text_analysis_score'] = text_eval['score']
        results['issues'].extend(text_eval['issues'])
        results['strengths'].extend(text_eval['strengths'])

        # 2. Voice Synthesis Evaluation
        voice_eval = self._evaluate_voice_synthesis(data.get('voice_synthesis_results', {}))
        results['metrics'].update(voice_eval['metrics'])
        results['scores']['voice_synthesis_score'] = voice_eval['score']
        results['issues'].extend(voice_eval['issues'])
        results['strengths'].extend(voice_eval['strengths'])

        # 3. Overall Quality Evaluation
        quality_eval = self._evaluate_overall_quality(data)
        results['metrics'].update(quality_eval['metrics'])
        results['scores']['overall_quality_score'] = quality_eval['score']
        results['issues'].extend(quality_eval['issues'])
        results['strengths'].extend(quality_eval['strengths'])

        # 4. Calculate overall score
        weights = {
            'text_analysis_score': 0.4,
            'voice_synthesis_score': 0.4,
            'overall_quality_score': 0.2
        }

        overall_score = sum(
            results['scores'].get(key, 0) * weight
            for key, weight in weights.items()
        )
        results['overall_score'] = overall_score

        # 5. Check against quality thresholds
        results['passed_quality_check'] = self._check_quality_thresholds(results)

        # 6. Generate recommendations
        results['recommendations'] = self._generate_recommendations(results)

        return results

    def _evaluate_text_analysis(self, text_data: dict) -> dict:
        """Evaluate text analysis quality"""
        evaluation = {
            'metrics': {},
            'score': 0.0,
            'issues': [],
            'strengths': []
        }

        total_blocks = text_data.get('total_blocks', 0)
        speaker_dist = text_data.get('speaker_distribution', {})

        # Calculate metrics
        if total_blocks > 0:
            # Block distribution analysis
            narrator_blocks = speaker_dist.get('narrator', 0)
            dialogue_blocks = sum(v for k, v in speaker_dist.items() if k != 'narrator')

            evaluation['metrics']['total_blocks'] = total_blocks
            evaluation['metrics']['narrator_blocks'] = narrator_blocks
            evaluation['metrics']['dialogue_blocks'] = dialogue_blocks
            evaluation['metrics']['dialogue_ratio'] = dialogue_blocks / total_blocks if total_blocks > 0 else 0

            # Score based on dialogue presence (good stories have dialogue)
            dialogue_score = min(evaluation['metrics']['dialogue_ratio'] * 2, 1.0)

            # Check for reasonable block count (not too fragmented)
            if total_blocks > 200:
                evaluation['issues'].append(f"High block count ({total_blocks}): Text may be over-fragmented")
                block_score = 0.7
            elif total_blocks < 50:
                evaluation['issues'].append(f"Low block count ({total_blocks}): Text may be under-segmented")
                block_score = 0.7
            else:
                evaluation['strengths'].append(f"Good block count: {total_blocks}")
                block_score = 1.0

            # Speaker distribution check
            if len(speaker_dist) >= 2:  # At least narrator + one character
                evaluation['strengths'].append(f"Multiple speakers detected: {list(speaker_dist.keys())}")
                speaker_score = 1.0
            else:
                evaluation['issues'].append("Limited speaker variety detected")
                speaker_score = 0.6

            evaluation['score'] = (dialogue_score * 0.4 + block_score * 0.3 + speaker_score * 0.3)
        else:
            evaluation['issues'].append("No text blocks analyzed")
            evaluation['score'] = 0.0

        return evaluation

    def _evaluate_voice_synthesis(self, voice_data: dict) -> dict:
        """Evaluate voice synthesis quality"""
        evaluation = {
            'metrics': {},
            'score': 0.0,
            'issues': [],
            'strengths': []
        }

        audio_duration = voice_data.get('audio_duration_seconds', 0)
        sample_rate = voice_data.get('sample_rate', 24000)
        audio_metadata = voice_data.get('audio_metadata', {})

        evaluation['metrics']['audio_duration'] = audio_duration
        evaluation['metrics']['sample_rate'] = sample_rate

        # Duration quality check
        if audio_duration > 0:
            evaluation['strengths'].append(f"Audio generated: {audio_duration:.2f} seconds")
            duration_score = 1.0

            # Check for reasonable duration (not too short/long for typical chapter)
            if audio_duration < 60:  # Less than 1 minute
                evaluation['issues'].append(f"Very short audio: {audio_duration:.1f}s (might be incomplete)")
                duration_score = 0.5
            elif audio_duration > 3600:  # More than 1 hour
                evaluation['issues'].append(f"Very long audio: {audio_duration:.1f}s (might need chapter splitting)")
                duration_score = 0.8
        else:
            evaluation['issues'].append("No audio generated")
            duration_score = 0.0

        # Audio quality metrics
        if audio_metadata:
            max_amplitude = audio_metadata.get('max_amplitude', 0)
            silence_ratio = audio_metadata.get('silence_ratio', 0)

            evaluation['metrics']['max_amplitude'] = max_amplitude
            evaluation['metrics']['silence_ratio'] = silence_ratio

            # Volume check
            if 0.1 < max_amplitude < 0.9:
                evaluation['strengths'].append(f"Good audio level: {max_amplitude:.3f}")
                volume_score = 1.0
            elif max_amplitude <= 0.1:
                evaluation['issues'].append(f"Audio too quiet: {max_amplitude:.3f}")
                volume_score = 0.3
            else:
                evaluation['issues'].append(f"Audio might clip: {max_amplitude:.3f}")
                volume_score = 0.7

            # Silence check (appropriate pauses)
            if 0.05 < silence_ratio < 0.3:
                evaluation['strengths'].append(f"Good pause ratio: {silence_ratio:.3f}")
                pause_score = 1.0
            elif silence_ratio <= 0.05:
                evaluation['issues'].append(f"Too little pause time: {silence_ratio:.3f}")
                pause_score = 0.6
            else:
                evaluation['issues'].append(f"Too much silence: {silence_ratio:.3f}")
                pause_score = 0.6
        else:
            volume_score = 0.5
            pause_score = 0.5

        # Sample rate check
        if sample_rate >= 22050:
            evaluation['strengths'].append(f"Good sample rate: {sample_rate}Hz")
            sample_rate_score = 1.0
        else:
            evaluation['issues'].append(f"Low sample rate: {sample_rate}Hz (quality may be reduced)")
            sample_rate_score = 0.7

        evaluation['score'] = (
            duration_score * 0.3 +
            volume_score * 0.3 +
            pause_score * 0.2 +
            sample_rate_score * 0.2
        )

        return evaluation

    def _evaluate_overall_quality(self, data: dict) -> dict:
        """Evaluate overall system quality"""
        evaluation = {
            'metrics': {},
            'score': 0.0,
            'issues': [],
            'strengths': []
        }

        input_data = data.get('input_analysis', {})
        text_results = data.get('text_analysis_results', {})
        voice_results = data.get('voice_synthesis_results', {})

        # Check input-output consistency
        estimated_time = input_data.get('estimated_speaking_time', 0)
        actual_time = voice_results.get('audio_duration_seconds', 0)

        if estimated_time > 0 and actual_time > 0:
            time_ratio = actual_time / estimated_time
            evaluation['metrics']['time_ratio'] = time_ratio

            if 0.7 < time_ratio < 1.3:
                evaluation['strengths'].append(
                    f"Good timing match: estimated {estimated_time:.1f}s, actual {actual_time:.1f}s"
                )
                timing_score = 1.0
            else:
                evaluation['issues'].append(
                    f"Timing mismatch: estimated {estimated_time:.1f}s, actual {actual_time:.1f}s (ratio: {time_ratio:.2f})"
                )
                timing_score = 0.6
        else:
            timing_score = 0.5

        # Check data completeness
        completeness_score = 1.0
        missing_data = []

        if not text_results:
            missing_data.append("text analysis")
            completeness_score -= 0.3
        if not voice_results:
            missing_data.append("voice synthesis")
            completeness_score -= 0.4
        if not input_data:
            missing_data.append("input analysis")
            completeness_score -= 0.3

        if missing_data:
            evaluation['issues'].append(f"Missing data: {', '.join(missing_data)}")

        # Overall coherence check
        text_blocks = text_results.get('total_blocks', 0)
        if text_blocks > 0 and actual_time > 0:
            blocks_per_minute = (text_blocks / actual_time) * 60
            evaluation['metrics']['blocks_per_minute'] = blocks_per_minute

            if 20 < blocks_per_minute < 100:
                evaluation['strengths'].append(f"Good pacing: {blocks_per_minute:.1f} blocks/minute")
                pacing_score = 1.0
            elif blocks_per_minute <= 20:
                evaluation['issues'].append(f"Slow pacing: {blocks_per_minute:.1f} blocks/minute")
                pacing_score = 0.7
            else:
                evaluation['issues'].append(f"Fast pacing: {blocks_per_minute:.1f} blocks/minute")
                pacing_score = 0.7
        else:
            pacing_score = 0.5

        evaluation['score'] = (timing_score * 0.4 + completeness_score * 0.3 + pacing_score * 0.3)

        return evaluation

    def _analyze_speaker_distribution(self, blocks: list) -> dict:
        """Analyze distribution of speakers in text blocks"""
        distribution = {}
        for block in blocks:
            speaker = block.get('speaker', 'unknown')
            distribution[speaker] = distribution.get(speaker, 0) + 1
        return distribution

    def _analyze_audio_metadata(self, audio_data: np.ndarray) -> dict:
        """Analyze audio data for quality metrics"""
        if audio_data is None or len(audio_data) == 0:
            return {}

        # Calculate max amplitude (volume level)
        max_amplitude = np.max(np.abs(audio_data))

        # Calculate silence ratio (samples below threshold)
        silence_threshold = 0.01
        silent_samples = np.sum(np.abs(audio_data) < silence_threshold)
        silence_ratio = silent_samples / len(audio_data)

        # Calculate dynamic range
        if np.std(audio_data) > 0:
            dynamic_range = 20 * np.log10(max_amplitude / (np.std(audio_data) + 1e-10))
        else:
            dynamic_range = 0

        return {
            'max_amplitude': float(max_amplitude),
            'silence_ratio': float(silence_ratio),
            'dynamic_range': float(dynamic_range),
            'total_samples': len(audio_data)
        }

    def _estimate_speaking_time(self, text: str) -> float:
        """Estimate speaking time based on text length"""
        # Average speaking rate: 150 words per minute
        words = len(text.split())
        minutes = words / 150
        return minutes * 60  # Convert to seconds

    def _check_quality_thresholds(self, results: dict) -> bool:
        """Check if results meet quality thresholds"""
        overall_score = results.get('overall_score', 0)
        text_score = results.get('scores', {}).get('text_analysis_score', 0)
        voice_score = results.get('scores', {}).get('voice_synthesis_score', 0)

        # All scores must be above thresholds
        thresholds_met = (
            overall_score >= 0.7 and
            text_score >= 0.6 and
            voice_score >= 0.6
        )

        # Check for critical issues
        critical_issues = [
            issue for issue in results.get('issues', [])
            if any(keyword in issue.lower() for keyword in [
                'no audio', 'failed', 'error', 'missing', 'corrupt'
            ])
        ]

        return thresholds_met and len(critical_issues) == 0

    def _generate_recommendations(self, results: dict) -> list:
        """Generate improvement recommendations"""
        recommendations = []
        issues = results.get('issues', [])

        # Text analysis recommendations
        text_score = results.get('scores', {}).get('text_analysis_score', 0)
        if text_score < 0.8:
            recommendations.append("Consider improving text segmentation for better pacing")

        # Voice synthesis recommendations
        voice_score = results.get('scores', {}).get('voice_synthesis_score', 0)
        if voice_score < 0.8:
            recommendations.append("Adjust voice synthesis parameters for better audio quality")

        # Specific issue-based recommendations
        for issue in issues:
            if 'too quiet' in issue.lower():
                recommendations.append("Increase audio gain during synthesis")
            elif 'clip' in issue.lower():
                recommendations.append("Reduce audio gain to prevent clipping")
            elif 'silence' in issue.lower():
                recommendations.append("Adjust pause durations between speech segments")
            elif 'pacing' in issue.lower():
                recommendations.append("Review text segmentation for better rhythm")

        # Add general recommendations if few specific ones
        if len(recommendations) < 2 and results.get('overall_score', 0) < 0.9:
            recommendations.append("Run detailed diagnostics for performance optimization")
            recommendations.append("Consider using higher-quality voice models")

        return recommendations[:5]  # Limit to 5 recommendations

    def _generate_report(self, results: dict) -> str:
        """Generate detailed evaluation report"""
        report = []
        report.append("=" * 60)
        report.append("AUDIOBOOK GENERATION EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")

        # Overall Score
        overall_score = results.get('overall_score', 0) * 100
        passed = results.get('passed_quality_check', False)
        report.append(f"📊 OVERALL SCORE: {overall_score:.1f}/100")
        report.append(f"✅ QUALITY CHECK: {'PASSED' if passed else 'FAILED'}")
        report.append("")

        # Component Scores
        report.append("📈 COMPONENT SCORES:")
        scores = results.get('scores', {})
        for component, score in scores.items():
            component_name = component.replace('_', ' ').title()
            report.append(f"  {component_name}: {score*100:.1f}/100")
        report.append("")

        # Metrics
        report.append("📋 KEY METRICS:")
        metrics = results.get('metrics', {})
        for metric, value in metrics.items():
            metric_name = metric.replace('_', ' ').title()
            if isinstance(value, float):
                report.append(f"  {metric_name}: {value:.3f}")
            else:
                report.append(f"  {metric_name}: {value}")
        report.append("")

        # Strengths
        strengths = results.get('strengths', [])
        if strengths:
            report.append("✅ STRENGTHS:")
            for strength in strengths[:5]:  # Top 5 strengths
                report.append(f"  • {strength}")
            report.append("")

        # Issues
        issues = results.get('issues', [])
        if issues:
            report.append("⚠️ ISSUES DETECTED:")
            for issue in issues[:5]:  # Top 5 issues
                report.append(f"  • {issue}")
            report.append("")

        # Recommendations
        recommendations = results.get('recommendations', [])
        if recommendations:
            report.append("💡 RECOMMENDATIONS:")
            for rec in recommendations:
                report.append(f"  • {rec}")
            report.append("")

        report.append("=" * 60)
        report.append(f"Evaluation completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 60)

        return "\n".join(report)

    def _log_evaluation_summary(self, results: dict):
        """Log a summary of evaluation results"""
        overall_score = results.get('overall_score', 0) * 100
        issues = len(results.get('issues', []))
        strengths = len(results.get('strengths', []))

        self.log(f"📊 Evaluation Complete:")
        self.log(f"  Overall Score: {overall_score:.1f}/100")
        self.log(f"  Issues Found: {issues}")
        self.log(f"  Strengths Identified: {strengths}")
        self.log(f"  Quality Check: {'✅ PASSED' if results.get('passed_quality_check') else '❌ FAILED'}")

        # Log top recommendation if any
        recommendations = results.get('recommendations', [])
        if recommendations:
            self.log(f"  Top Recommendation: {recommendations[0]}")

# ====== YOUR TEXT GOES HERE ======
# (Paste your full chapter text here - removed for brevity in this example)
your_text = """Title: The Copper Key
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

Mira tightened her grip on the key and the bag and stepped into the streetlights, knowing she had just crossed into a story that wouldn't let her go."""  # Your full text here

# ====== MAIN EXECUTION ======
if __name__ == "__main__":
    # Create and configure agents
    orchestrator = OrchestratorAgent()
    analyzer = TextAnalyzerAgent()
    synthesizer = VoiceSynthesisAgent(sample_rate=24000)
    evaluator = EvaluatorAgent()

    # Register agents
    orchestrator.register_agent(analyzer)
    orchestrator.register_agent(synthesizer)
    orchestrator.register_agent(evaluator)

    initial_msg = AgentMessage(
        "User",
        "Orchestrator",
        {"text": your_text},
        "audio_generation_request"
    )

    # Process the text
    print("=" * 60)
    print("AUDIOBOOK AGENT SYSTEM")
    print("=" * 60)
    final_result = orchestrator.receive(initial_msg)

    # Play and save audio
    if final_result and final_result.content:
        audio_data = final_result.content.get("audio_data")
        sample_rate = final_result.content.get("sample_rate")

        if audio_data is not None and len(audio_data) > 0:
            duration = len(audio_data) / sample_rate
            print(f"\n{'='*60}")
            print(f"🎵 AUDIO READY: {duration:.2f} seconds")
            print(f"{'='*60}")

            # Display evaluation report
            if "evaluation_report" in final_result.content:
                print("\n" + final_result.content["evaluation_report"])

            # Play audio
            display(Audio(audio_data, rate=sample_rate))

            # Save option
            save_option = input("\n💾 Save audio file? (y/n): ").lower()
            if save_option == 'y':
                filename = "audiobook_chapter.wav"
                sf.write(filename, audio_data, sample_rate)
                print(f"✅ Saved to: {filename}")
                print(f"📊 File size: {os.path.getsize(filename) / 1024 / 1024:.1f} MB")

                # Save evaluation report
                report_filename = "evaluation_report.txt"
                with open(report_filename, 'w') as f:
                    f.write(final_result.content.get("evaluation_report", "No evaluation report available"))
                print(f"📝 Evaluation report saved to: {report_filename}")
        else:
            print("❌ No audio generated")
    else:
        print("❌ Processing failed")
