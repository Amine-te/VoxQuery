"""
Text-to-speech helper using XTTS v2 and pyttsx3 fallback.
Handles WAV files directly, converts MP3→WAV if needed.
Updated for VoxQuery integration with multi-language support.
"""

import os
import time
import tempfile
import subprocess
import logging
from pathlib import Path
from typing import Tuple

log = logging.getLogger(__name__)


def get_speaker_wav(speaker_path: str) -> Tuple[bool, str, str]:
    """
    Check speaker file type:
    - If WAV: return it as-is
    - If MP3: convert to WAV and return temp path
    - If missing: return error
    
    Returns: (success, error_msg, speaker_wav_path)
    """
    if not speaker_path:
        return False, "No speaker path provided", None
    
    path = Path(speaker_path)
    if not path.exists():
        return False, f"Speaker file not found: {speaker_path}", None
    
    suffix = path.suffix.lower()
    
    # Already WAV - use directly
    if suffix == ".wav":
        log.info(f"Using WAV speaker file: {speaker_path}")
        return True, None, speaker_path
    
    # MP3 - convert to WAV
    if suffix == ".mp3":
        log.info(f"Converting MP3 → WAV: {speaker_path}")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
        
        # Try pydub first (preferred)
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(str(speaker_path))
            audio.export(wav_path, format="wav")
            log.info(f"Converted via pydub to: {wav_path}")
            return True, None, wav_path
        except Exception as e:
            log.debug(f"pydub conversion failed: {e}")
        
        # Fallback: ffmpeg CLI
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(speaker_path), wav_path],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30
            )
            log.info(f"Converted via ffmpeg to: {wav_path}")
            return True, None, wav_path
        except Exception as e:
            return False, f"MP3→WAV conversion failed (pydub and ffmpeg both unavailable): {e}", None
    
    return False, f"Unsupported audio format: {suffix} (expected .wav or .mp3)", None


def register_safe_globals():
    """Register known TTS classes with torch.serialization for PyTorch 2.6+ compatibility."""
    try:
        import torch
        try:
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
            from TTS.config.shared_configs import BaseDatasetConfig
            torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig])
            log.info("Registered TTS classes as safe globals (PyTorch 2.6+ compatibility)")
        except Exception:
            try:
                from TTS.tts.configs.xtts_config import XttsConfig
                torch.serialization.add_safe_globals([XttsConfig])
                log.info("Registered XttsConfig as safe global")
            except Exception:
                log.debug("Could not register TTS classes; continuing anyway")
    except Exception as e:
        log.debug(f"torch registration skipped: {e}")


def detect_language(text: str) -> str:
    """
    Detect language of input text.
    Returns ISO 639-1 code (e.g., 'en', 'fr', 'ar')
    """
    try:
        from langdetect import detect
        lang = detect(text)
        # Map common languages
        lang_map = {
            'fr': 'fr',  # French
            'en': 'en',  # English
            'ar': 'ar',  # Arabic
            'es': 'es',  # Spanish
        }
        return lang_map.get(lang, 'en')  # Default to English
    except Exception as e:
        log.warning(f"Language detection failed: {e}. Defaulting to English.")
        return 'en'


def generate_audio_xtts(
    text: str,
    output_path: str = "response_audio.wav",
    language: str = None,
    speaker_wav: str = None
) -> Tuple[bool, str, float, str]:
    """
    Generate audio using XTTS v2 with optional voice cloning.
    
    - If speaker_wav is a WAV: use directly for voice cloning
    - If speaker_wav is an MP3: convert to WAV first
    - If no speaker_wav: use XTTS default voice (NOT pyttsx3)
    - If language is None: auto-detect from text
    
    Returns (success, error_msg, elapsed_ms, output_path)
    """
    start_time = time.time()
    
    # Auto-detect language if not provided
    if language is None:
        language = detect_language(text)
        log.info(f"Auto-detected language: {language}")
    
    try:
        register_safe_globals()
        from TTS.api import TTS
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"Loading XTTS v2 model (device={device})...")
        tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False)
        tts.to(device)

        # Prepare speaker reference (convert MP3→WAV if needed)
        speaker_to_use = None
        if speaker_wav:
            ok, err, speaker_to_use = get_speaker_wav(speaker_wav)
            if not ok:
                log.warning(f"Speaker preparation failed: {err}. Using default XTTS voice instead.")
        
        # Synthesize with XTTS
        if speaker_to_use:
            log.info(f"Synthesizing with voice cloning from {speaker_to_use}...")
            tts.tts_to_file(
                text=text, 
                file_path=output_path, 
                speaker_wav=speaker_to_use, 
                language=language
            )
        else:
            # Use XTTS default voice (no voice cloning)
            log.info(f"Synthesizing with XTTS default voice (no speaker reference provided)...")
            
            # For XTTS, we need to provide either speaker_wav OR speaker name
            # Since we don't have speaker_wav, use default speaker
            # Note: Different XTTS versions handle this differently
            try:
                # Try with speaker parameter (some XTTS versions)
                tts.tts_to_file(
                    text=text,
                    file_path=output_path,
                    language=language,
                    speaker="Claribel Dervla"  # XTTS default speaker
                )
            except TypeError:
                # If speaker parameter fails, try without it
                # This forces XTTS to use its default voice
                tts.tts_to_file(
                    text=text,
                    file_path=output_path,
                    language=language
                )

        elapsed_ms = (time.time() - start_time) * 1000
        log.info(f"XTTS synthesis completed in {elapsed_ms:.0f}ms")
        return True, None, elapsed_ms, output_path

    except ImportError as e:
        elapsed_ms = (time.time() - start_time) * 1000
        log.warning(f"XTTS unavailable: {e}. Falling back to pyttsx3...")
        # Fallback to pyttsx3 (lightweight, offline)
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.save_to_file(text, output_path)
            engine.runAndWait()
            log.info("pyttsx3 synthesis completed")
            return True, None, (time.time() - start_time) * 1000, output_path
        except Exception as e2:
            return False, f"XTTS import error: {e}; pyttsx3 fallback failed: {e2}", elapsed_ms, None

    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        log.error(f"XTTS generation failed: {e}")
        # Try pyttsx3 fallback
        log.warning("Attempting pyttsx3 fallback...")
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.save_to_file(text, output_path)
            engine.runAndWait()
            log.info("pyttsx3 synthesis completed")
            return True, None, (time.time() - start_time) * 1000, output_path
        except Exception as e2:
            return False, f"XTTS failed: {e}; pyttsx3 also failed: {e2}", elapsed_ms, None