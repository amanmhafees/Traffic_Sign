from gtts import gTTS  # Import Google Text-to-Speech
from langcodes import Language  # For handling language codes
from pydub import AudioSegment  # For audio playback
from pydub.playback import play  # For playing audio
import logging
from pathlib import Path
from typing import List, Optional
import streamlit as st  # Import Streamlit for visual notifications
import time  # For controlling the display duration of the notification
import os
import sys
import signal
import shutil  # For ffmpeg detection
try:
    import pyttsx3
except Exception:
    pyttsx3 = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NotificationHandler:
    def __init__(self, output_path: str = "output"):
        """
        Initialize the Notification Handler
        
        Args:
            output_path: Path for storing audio alerts
        """
        self.output_path = Path(output_path)
        self.audio_dir = self.output_path / "audio_alerts"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.available_languages = [
            "hi",  # Hindi
            "en",  # English
            "ta",  # Tamil
            "te",  # Telugu
            "kn",  # Kannada
            "ml",  # Malayalam
            "mr",  # Marathi
            "gu",  # Gujarati
            "bn",  # Bengali
            "pa",  # Punjabi
            "or",  # Odia
            "as",  # Assamese
            "ur",  # Urdu
            "sd",  # Sindhi
            "sa",  # Sanskrit
            "ne",  # Nepali
            "si",  # Sinhala
        ]
        self.notification_display_time = 3  # Time in seconds to display visual notifications
        # ffmpeg / playback capability detection
        self.use_pydub = self._check_ffmpeg()
        self.warned_no_ffmpeg = False
        # simple in-memory cache to skip regenerating same audio this session
        self._audio_cache = {}  # (sign, lang) -> Path
        self.offline_tts_engine = None
        if pyttsx3:
            try:
                self.offline_tts_engine = pyttsx3.init()
                self.offline_tts_engine.setProperty("rate", 170)
            except Exception:
                self.offline_tts_engine = None
    
    def _check_ffmpeg(self) -> bool:
        """
        Return True if ffmpeg (or avconv) is available for pydub playback.
        """
        return bool(shutil.which("ffmpeg") or shutil.which("avconv"))

    def _speak_offline(self, text: str):
        if self.offline_tts_engine:
            try:
                self.offline_tts_engine.say(text)
                self.offline_tts_engine.runAndWait()
            except Exception as e:
                logger.warning(f"Offline TTS failed: {e}")

    def notify_traffic_sign(self, detected_sign: str, languages: Optional[List[str]] = None):
        """
        Notify the user of the detected traffic sign and generate audio alerts in multiple languages.
        
        Args:
            detected_sign: The name of the detected traffic sign.
            languages: List of language codes for audio alerts. Defaults to all available languages.
        """
        if languages is None:
            languages = self.available_languages
        logger.info(f"Detected traffic sign: {detected_sign}")
        self._show_visual_notification(detected_sign)
        first_spoken = False
        for lang in languages:
            try:
                cache_key = (detected_sign, lang)
                if cache_key in self._audio_cache and self._audio_cache[cache_key].exists():
                    audio_file = self._audio_cache[cache_key]
                else:
                    audio_file = self.audio_dir / f"{detected_sign}_{lang}.mp3"
                    if not audio_file.exists():
                        # Speak only the sign name (clean underscores)
                        simple_text = detected_sign.replace("_", " ")
                        tts = gTTS(text=simple_text, lang=lang)
                        tts.save(audio_file)
                    self._audio_cache[cache_key] = audio_file

                if not first_spoken:
                    # Try online mp3 playback else offline fallback
                    if self.use_pydub:
                        try:
                            audio = AudioSegment.from_file(audio_file)
                            logger.info(f"Playing (pydub) {detected_sign} in {Language.get(lang).display_name()}")
                            play(audio)
                            first_spoken = True
                            continue
                        except Exception as e:
                            logger.warning(f"pydub playback failed: {e}")
                    # Streamlit fallback or offline
                    if not self.use_pydub and not self.warned_no_ffmpeg:
                        st.warning("ffmpeg not found. Using fallback audio method. Install ffmpeg for better playback.")
                        logger.warning("ffmpeg missing – fallback playback in use.")
                        self.warned_no_ffmpeg = True
                    # Provide Streamlit embedded audio
                    try:
                        with open(audio_file, "rb") as f:
                            st.audio(f.read(), format="audio/mp3")
                        first_spoken = True
                        continue
                    except Exception as e:
                        logger.warning(f"Streamlit audio fallback failed: {e}")
                    # Offline last resort
                    self._speak_offline(detected_sign.replace("_", " "))
                    first_spoken = True
                # Do not auto-play additional languages (avoid long delays); still cache them
            except Exception as e:
                logger.error(f"Failed audio generation/playback for '{detected_sign}' lang '{lang}': {e}")

    def _show_visual_notification(self, message: str) -> None:
        """
        Display a visual notification (toast if available, else fixed positioned HTML).
        Non-blocking: no sleep inside.
        """
        clean_msg = message.replace("_", " ")
        # Prefer Streamlit native toast if version supports it
        try:
            st.toast(f"🚦 {clean_msg}")
            return
        except Exception:
            pass
        # Fallback custom HTML (persistent one render cycle)
        placeholder = st.empty()
        placeholder.markdown(
            f"""
            <div style="
                position: fixed;
                top: 12px;
                right: 12px;
                background: linear-gradient(135deg,#222,#444);
                color:#fff;
                padding:12px 18px;
                border-radius:8px;
                font-size:15px;
                font-family:system-ui,Arial,sans-serif;
                box-shadow:0 4px 10px rgba(0,0,0,0.25);
                z-index:1000;
                animation: fadeSlide 0.35s ease-out;
            ">
                🚦 <strong>{clean_msg}</strong>
            </div>
            <style>
            @keyframes fadeSlide {{
              from {{ opacity:0; transform:translateY(-10px); }}
              to   {{ opacity:1; transform:translateY(0); }}
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
        # Auto-clear after display time using a lightweight timer (simulate via rerun cycle)
        # We store timestamp in session_state to clear on next run
        key = f"notif_{clean_msg}"
        st.session_state[f"{key}_ts"] = time.time()

    def stop_app(self):
        """
        Gracefully stop current Streamlit script execution (does NOT shut down the server).
        """
        st.warning("Stopping current Streamlit script run.")
        st.stop()

    def kill_server(self, force: bool = False):
        """
        Terminate the Streamlit server process.
        force=True uses os._exit(0); otherwise raises SystemExit.
        """
        logger.info("Terminating Streamlit server...")
        if force:
            os._exit(0)
        raise SystemExit("Streamlit server terminated.")
