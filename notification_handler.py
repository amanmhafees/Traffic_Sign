from gtts import gTTS  # Import Google Text-to-Speech
from langcodes import Language  # For handling language codes
from pydub import AudioSegment  # For audio playback
from pydub.playback import play  # For playing audio
import logging
from pathlib import Path
from typing import List, Optional

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
        
        for lang in languages:
            try:
                # Generate audio alert
                tts = gTTS(text=f"Detected traffic sign: {detected_sign}", lang=lang)
                audio_file = self.audio_dir / f"{detected_sign}_{lang}.mp3"
                tts.save(audio_file)
                
                # Play the audio alert using pydub
                audio = AudioSegment.from_file(audio_file)
                logger.info(f"Playing audio alert in {Language.get(lang).display_name()}")
                play(audio)
            except Exception as e:
                logger.error(f"Failed to generate or play audio for language '{lang}': {e}")
