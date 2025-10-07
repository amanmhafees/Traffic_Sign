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
import shutil  # For ffmpeg detection
try:
    import pyttsx3
except Exception:
    pyttsx3 = None
try:
    import winsound  # Windows beep
except Exception:
    winsound = None

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
        # Add missing flag for popup sound (prevents AttributeError)
        self.enable_popup_sound = True  # set False to disable popup chime
        self.audio_dir = self.output_path / "audio_alerts"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.available_languages = [
            "en",  # English
            "hi",  # Hindi
            "ta",  # Tamil
            "te",  # Telugu
            "kn",  # Kannada
            "ml",  # Malayalam
            "mr",  # Marathi
            "gu",  # Gujarati
            "bn",  # Bengali
            "pa",  # Punjabi (uses Hindi TTS fallback)
        ]
        # Keep alias map only for Punjabi fallback
        self.language_alias_map = {
            'pa': 'hi',
        }
        self.notification_display_time = 3  # Time in seconds to display visual notifications
        # Notification theme (customizable)
        self.notification_theme = {
            "bg_gradient": "linear-gradient(135deg,#1e3c72,#2a5298)",
            "border": "1px solid rgba(255,255,255,0.15)",
            "text_color": "#ffffff",
            "accent": "#ffcf33",
            "shadow": "0 8px 22px -6px rgba(0,0,0,0.55)"
        }
        # ffmpeg / playback capability detection
        self.use_pydub = self._check_ffmpeg()
        self.warned_no_ffmpeg = False
        # simple in-memory cache to skip regenerating same audio this session
        self._audio_cache = {}  # (sign, lang) -> Path
        self._audio_text_cache = {}  # (sign, lang) -> exact text used for TTS (to ensure description reuse)
        self._translation_cache = {}  # (sign, lang) -> translated description
        self.offline_tts_engine = None
        if pyttsx3:
            try:
                self.offline_tts_engine = pyttsx3.init()
                self.offline_tts_engine.setProperty("rate", 170)
            except Exception:
                self.offline_tts_engine = None

        self.sign_descriptions = {
            # Cautionary / Warning
            "BARRIER_AHEAD": "Warning: Barrier ahead. Slow down and prepare to stop.",
            "CATTLE": "Caution: Cattle crossing ahead. Drive carefully.",
            "CROSS_ROAD": "Warning: Crossroad ahead. Watch for traffic from other directions.",
            "DANGEROUS_DIP": "Warning: Dangerous dip ahead. Reduce speed immediately.",
            "FALLING_ROCKS": "Caution: Falling rocks ahead. Drive with extra care.",
            "FERRY": "Warning: Ferry service ahead. Prepare to stop if required.",
            "GAP_IN_MEDIAN": "Caution: Gap in road median ahead. Stay alert.",
            "GUARDED_LEVEL_CROSSING": "Warning: Guarded railway crossing ahead. Slow down and prepare to stop.",
            "HUMP_OR_ROUGH_ROAD": "Warning: Speed bump or rough road ahead. Reduce speed.",
            "LEFT_HAIR_PIN_BEND": "Sharp left hairpin bend ahead. Drive cautiously.",
            "LEFT_HAND_CURVE": "Left-hand curve ahead. Slow down.",
            "LEFT_REVERSE_BEND": "Left reverse bend ahead. Follow the road carefully.",
            "LOOSE_GRAVEL": "Warning: Loose gravel on the road. Reduce speed to avoid skidding.",
            "MEN_AT_WORK": "Caution: Road work in progress. Drive slowly and follow instructions.",
            "NARROW_BRIDGE": "Warning: Narrow bridge ahead. Yield if necessary.",
            "NARROW_ROAD_AHEAD": "Road narrows ahead. Maintain lane discipline.",
            "PEDESTRIAN_CROSSING": "Pedestrian crossing ahead. Slow down and yield to pedestrians.",
            "RIGHT_HAIR_PIN_BEND": "Sharp right hairpin bend ahead. Drive carefully.",
            "RIGHT_HAND_CURVE": "Right-hand curve ahead. Reduce speed.",
            "RIGHT_REVERSE_BEND": "Right reverse bend ahead. Follow the curve carefully.",
            "ROAD_WIDENS_AHEAD": "Road widens ahead. Stay in your lane.",
            "ROUNDABOUT": "Roundabout ahead. Slow down and give way to traffic on the right.",
            "SCHOOL_AHEAD": "School zone ahead. Drive slowly and watch for children.",
            "SIDE_ROAD_LEFT": "Side road joining from the left. Watch for merging vehicles.",
            "SIDE_ROAD_RIGHT": "Side road joining from the right. Stay alert for merging traffic.",
            "SLIPPERY_ROAD": "Slippery road ahead. Reduce speed and avoid sudden braking.",
            "STAGGERED_INTERSECTION": "Staggered intersection ahead. Proceed with caution.",
            "STEEP_ASCENT": "Steep ascent ahead. Shift to a lower gear.",
            "STEEP_DESCENT": "Steep descent ahead. Use a lower gear and control your speed.",
            "T_INTERSECTION": "T-intersection ahead. Prepare to stop or turn.",
            "UNGUARDED_LEVEL_CROSSING": "Warning: Unguarded railway crossing ahead. Stop, look, and proceed cautiously.",
            "Y_INTERSECTION": "Y-intersection ahead. Choose your direction carefully.",
            # Informatory
            "Destination_Sign": "Indicates directions to major destinations or towns ahead.",
            "Direction_Sign": "Provides information on direction and distance to nearby places.",
            "Eating_Place": "Food or restaurant available ahead.",
            "First_Aid_Post": "First aid or medical assistance ahead.",
            "Flood_Gauge": "Flood water level indicator ahead.",
            "Hospital": "Hospital or medical facility ahead.",
            "Light_Refreshment": "Refreshment stall or café ahead.",
            "No_Thorough_Road": "Dead-end road ahead. No through passage.",
            "No_Thorough_Side_Road": "Side road ends ahead and does not continue through.",
            "Park_This_Side": "Designated parking on this side of the road.",
            "Parking_Lot_Cars": "Parking area available for cars.",
            "Parking_Lot_Cycle": "Parking area available for bicycles.",
            "Parking_Lot_Scooter_and_Motorcycle": "Parking area available for two-wheelers.",
            "Petrol_Pump": "Fuel station ahead.",
            "Place_Identification": "Location or landmark identification ahead.",
            "Public_Telephone": "Public telephone available ahead.",
            "Re-assurance_Sign": "You are on the correct route.",
            "Resting_Place": "Rest area or stopping place ahead.",

            # --- Mandatory / Regulatory Signs (Added) ---
            "ALL_MOTOR_VEHICLE_PROHIBITED": "No motor vehicles beyond this point.",
            "AXLE_LOAD_LIMIT": "Axle load limit ahead. Do not exceed the specified axle load.",
            "BULLOCK_AND_HANDCART_PROHIBITED": "Bullock carts and handcarts are not allowed beyond this point.",
            "BULLOCK_PROHIBITED": "Bullock carts are prohibited on this road.",
            "COMPULSARY_AHEAD": "Proceed straight ahead only.",
            "COMPULSARY_AHEAD_OR_TURN_LEFT": "Go straight or turn left only.",
            "COMPULSARY_AHEAD_OR_TURN_RIGHT": "Go straight or turn right only.",
            "COMPULSARY_CYCLE_TRACK": "Cyclists must use the designated cycle track.",
            "COMPULSARY_KEEP_LEFT": "Keep to the left side of the road.",
            "COMPULSARY_KEEP_RIGHT": "Keep to the right side of the road.",
            "COMPULSARY_MINIMUM_SPEED": "Maintain at least the posted minimum speed.",
            "COMPULSARY_SOUND_HORN": "Sound horn where necessary.",
            "COMPULSARY_TURN_LEFT": "Turn left only.",
            "COMPULSARY_TURN_LEFT_AHEAD": "Proceed ahead or turn left.",
            "COMPULSARY_TURN_RIGHT": "Turn right only.",
            "COMPULSARY_TURN_RIGHT_AHEAD": "Proceed ahead or turn right.",
            "CYCLE_PROHIBITED": "Cycles are prohibited on this road.",
            "GIVE_WAY": "Give way to traffic on the main road.",
            "HANDCART_PROHIBITED": "Handcarts are not allowed on this road.",
            "HEIGHT_LIMIT": "Height restricted ahead. Do not exceed the posted limit.",
            "HORN_PROHIBITED": "Use of horn is prohibited in this area.",
            "LEFT_TURN_PROHIBITED": "Left turn is not allowed ahead.",
            "LENGTH_LIMIT": "Length restricted. Do not exceed the vehicle length limit.",
            "LOAD_LIMIT": "Maximum permitted vehicle load. Do not exceed.",
            "NO_ENTRY": "Do not enter. Entry is prohibited.",
            "NO_PARKING": "Parking is not allowed in this area.",
            "NO_STOPPING_OR_STANDING": "Stopping or standing is prohibited.",
            "OVERTAKING_PROHIBITED": "No overtaking allowed beyond this point.",
            "PEDESTRIAN_PROHIBITED": "Pedestrians are not allowed on this road.",
            "PRIORITY_FOR_ONCOMING_VEHICLES": "Oncoming vehicles have the right of way. Yield.",
            "RESTRICTION_ENDS": "All previous restrictions end here.",
            "RIGHT_TURN_PROHIBITED": "Right turn is not allowed ahead.",
            "SPEED_LIMIT_5": "Maximum speed limit is 5 kilometers per hour.",
            "SPEED_LIMIT_15": "Maximum speed limit is 15 kilometers per hour.",
            "SPEED_LIMIT_20": "Maximum speed limit is 20 kilometers per hour.",
            "SPEED_LIMIT_30": "Maximum speed limit is 30 kilometers per hour.",
            "SPEED_LIMIT_40": "Maximum speed limit is 40 kilometers per hour.",
            "SPEED_LIMIT_50": "Maximum speed limit is 50 kilometers per hour.",
            "SPEED_LIMIT_60": "Maximum speed limit is 60 kilometers per hour.",
            "SPEED_LIMIT_70": "Maximum speed limit is 70 kilometers per hour.",
            "SPEED_LIMIT_80": "Maximum speed limit is 80 kilometers per hour.",
            "STOP": "Stop completely and proceed only when safe.",
            "STRAIGHT_PROHIBITED": "Going straight is prohibited. Turn as directed.",
            "TONGA_PROHIBITED": "Horse-drawn tonga vehicles are prohibited."
        }
        # gTTS supported language codes reference (subset relevant to project)
        self.gtts_supported = {
            'af','ar','bn','bs','ca','cs','cy','da','de','el','en','en-us','en-uk','eo','es','et','fi','fr',
            'gu','hi','hr','hu','id','is','it','ja','jw','km','kn','ko','la','lv','mk','ml','mr','my','ne',
            'nl','no','pl','pt','ro','ru','si','sk','sq','sr','su','sv','sw','ta','te','th','tl','tr','uk',
            'ur','vi','zh-cn','zh-tw'
        }
        # Map unsupported requested codes to closest supported TTS codes
        # (Only affects TTS voice selection; translation still uses original requested code when possible)
        self.language_alias_map = {
            'as': 'hi',   # Assamese -> Hindi fallback (gTTS no 'as')
            'sd': 'hi',   # Sindhi -> Hindi
            'sa': 'hi',   # Sanskrit -> Hindi (best available)
            'si': 'si',   # Sinhala supported
            'ne': 'ne',   # Nepali supported
            # already supported directly: hi, en, ta, te, kn, ml, mr, gu, bn, pa (Punjabi not in gTTS, map)
            'pa': 'hi',   # Punjabi -> Hindi (gTTS lacks 'pa')
            # ensure base languages map to themselves
        }
        self._unsupported_warned = set()
        self._external_marker = "<external>"  # marker to note manually supplied audio
        # Track whether we've already warned about missing ffmpeg to reduce log noise
        self._shown_ffmpeg_embed_warning = False

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

    def _format_sign_phrase(self, raw_name: str) -> str:
        """
        Convert a raw class name like 'NO_OVERTAKING' into a natural phrase 'No Overtaking'.
        Underscores become spaces; words are title-cased.
        """
        return raw_name.replace("_", " ").strip().title()

    def _get_description(self, raw_name: str) -> str:
        """
        Return mapped description if available else a formatted phrase.
        """
        return self.sign_descriptions.get(raw_name, self._format_sign_phrase(raw_name))

    def _get_localized_description(self, raw_name: str, lang: str) -> str:
        """
        Return a localized (translated) description for the sign if possible.
        Falls back to English description if translation library or language unsupported.
        """
        base_desc = self._get_description(raw_name)
        if lang.startswith("en"):
            return base_desc
        key = (raw_name, lang)
        if key in self._translation_cache:
            return self._translation_cache[key]
        translated = base_desc
        try:
            from googletrans import Translator
            tr = Translator()
            translated = tr.translate(base_desc, dest=lang).text
        except Exception as e:
            logger.debug(f"Translation fallback to English for {raw_name} ({lang}): {e}")
            translated = base_desc
        self._translation_cache[key] = translated
        return translated

    def _resolve_tts_lang(self, requested_lang: str) -> str:
        """
        Return a gTTS-supported language code for synthesis.
        Uses alias map; if still unsupported, default to English.
        """
        # Normalize
        base = requested_lang.lower()
        # Direct support
        if base in self.gtts_supported:
            return base
        # Alias mapping
        alias = self.language_alias_map.get(base)
        if alias and alias in self.gtts_supported:
            if base not in self._unsupported_warned:
                logger.warning(f"gTTS does not support '{base}', using '{alias}' voice fallback.")
                self._unsupported_warned.add(base)
            return alias
        # Final fallback
        if base not in self._unsupported_warned:
            logger.warning(f"gTTS does not support '{base}', falling back to 'en'.")
            self._unsupported_warned.add(base)
        return 'en'

    def _build_tts(self, text: str, lang: str, fallback_lang: str = "en"):
        """
        Build gTTS with resolved language alias; fallback to English on failure.
        """
        resolved = self._resolve_tts_lang(lang)
        try:
            return gTTS(text=text, lang=resolved)
        except Exception as e:
            logger.debug(f"gTTS build failed for '{resolved}' ({lang} requested): {e}; fallback to {fallback_lang}")
            return gTTS(text=text, lang=fallback_lang)

    def notify_traffic_sign(self, detected_sign: str, languages: Optional[List[str]] = None):
        """
        Notify user; generate / play audio (translated per selected language).
        Shows English + selected language players.
        """
        if languages is None:
            languages = self.available_languages
        phrase = self._format_sign_phrase(detected_sign)
        base_description = self._get_description(detected_sign)
        logger.info(f"Detected traffic sign: {phrase} -> Multilingual descriptions")
        self._show_visual_notification(f"{phrase}|||{base_description}")

        # English canonical audio generation (unchanged logic except now uses _build_tts)
        english_key = (detected_sign, "en")
        english_audio_path = self.audio_dir / f"{detected_sign}_en.mp3"
        # Accept externally provided English file (if any) without regenerating
        if english_audio_path.exists() and english_key not in self._audio_cache:
            self._audio_cache[english_key] = english_audio_path
            self._audio_text_cache[english_key] = self._external_marker  # do not force regeneration
        if english_key not in self._audio_cache or not self._audio_cache[english_key].exists() or (
            self._audio_text_cache.get(english_key) not in (self._external_marker, base_description)
        ):
            try:
                tts_en = self._build_tts(base_description, "en")
                tts_en.save(english_audio_path)
                self._audio_cache[english_key] = english_audio_path
                self._audio_text_cache[english_key] = base_description
            except Exception as e:
                logger.error(f"Failed to generate English audio for {detected_sign}: {e}")
        else:
            english_audio_path = self._audio_cache[english_key]

        first_spoken = False
        ui_audio_entries = [("English", english_audio_path, base_description, "en")]

        for lang in languages:
            try:
                localized_desc = self._get_localized_description(detected_sign, lang)
                desired_text = localized_desc
                cache_key = (detected_sign, lang)
                audio_file = self.audio_dir / f"{detected_sign}_{lang}.mp3"

                # NEW: if an external file is present but not cached, register it and skip regeneration
                if audio_file.exists() and cache_key not in self._audio_cache:
                    self._audio_cache[cache_key] = audio_file
                    self._audio_text_cache[cache_key] = self._external_marker

                regenerate = True
                if cache_key in self._audio_cache and self._audio_cache[cache_key].exists():
                    # If it was marked external, never overwrite
                    existing_marker = self._audio_text_cache.get(cache_key)
                    if existing_marker == self._external_marker or existing_marker == desired_text:
                        regenerate = False

                if regenerate:
                    try:
                        tts = self._build_tts(desired_text, lang)
                        tts.save(audio_file)
                        self._audio_cache[cache_key] = audio_file
                        self._audio_text_cache[cache_key] = desired_text
                        logger.debug(f"Generated TTS ({lang}) for {detected_sign} using voice '{self._resolve_tts_lang(lang)}'")
                    except Exception as e:
                        logger.error(f"TTS generation failed ({detected_sign}, {lang}): {e}")
                        if not first_spoken:
                            self._speak_offline(desired_text)
                            first_spoken = True
                        continue
                if lang != "en":
                    ui_audio_entries.append((lang, audio_file, desired_text, lang))
                if not first_spoken:
                    if self.use_pydub:
                        try:
                            seg = AudioSegment.from_file(audio_file)
                            play(seg)
                            first_spoken = True
                            continue
                        except Exception as e:
                            logger.warning(f"pydub playback failed ({lang}): {e}")
                    if not self.use_pydub and not self.warned_no_ffmpeg:
                        st.warning("ffmpeg not found. Using embedded audio fallback.")
                        self.warned_no_ffmpeg = True
                    # Avoid repeating embed warning every call
                    try:
                        with open(audio_file, "rb") as f:
                            st.audio(f.read(), format="audio/mp3")
                        first_spoken = True
                        continue
                    except Exception as e:
                        logger.warning(f"Embedded audio playback failed ({lang}): {e}")
                    self._speak_offline(desired_text)
                    first_spoken = True
            except Exception as e:
                logger.error(f"Multilingual handling failed for {detected_sign} ({lang}): {e}")

        # Render audio players (unchanged structure)
        st.markdown(f"**Audio Descriptions ({phrase}):**")
        cols = st.columns(min(4, len(ui_audio_entries)))
        for i, (label, path, _text_used, lang_code) in enumerate(ui_audio_entries):
            try:
                if path and path.exists():
                    with open(path, "rb") as f:
                        with cols[i % len(cols)]:
                            st.audio(f.read(), format="audio/mp3")
                            st.caption(f"{label} ({lang_code})")
                else:
                    with cols[i % len(cols)]:
                        st.warning(f"Audio missing for {label}")
            except Exception as e:
                logger.warning(f"Failed showing audio player for {label}: {e}")
        # End notify

    def set_notification_theme(self, **kwargs):
        """
        Override notification theme colors.
        Example: set_notification_theme(bg_gradient='linear-gradient(...)', accent='#FF6600')
        """
        self.notification_theme.update({k: v for k, v in kwargs.items() if k in self.notification_theme})

    def _play_notification_sound(self):
        """
        Play a short notification chime (best-effort).
        Safe against missing attribute or unsupported environment.
        """
        # Guard if older instance without attribute
        if not hasattr(self, "enable_popup_sound") or not self.enable_popup_sound:
            return
        try:
            if winsound:
                winsound.Beep(1200, 140)  # (freq, ms)
            else:
                # Terminal bell fallback (may be ignored in some environments)
                print("\a", end="")
        except Exception as e:
            logger.debug(f"Popup sound failed: {e}")

    def _show_visual_notification(self, message: str) -> None:
        """
        Display a visual notification (toast if available, else styled HTML popup).
        """
        # play sound first
        self._play_notification_sound()
        # message may include "Phrase|||Description"
        parts = str(message).split("|||", 1)
        phrase = self._format_sign_phrase(parts[0])
        desc = parts[1] if len(parts) > 1 else self._get_description(parts[0])
        try:
            st.toast(f"🚦 {phrase} – {desc}")
            return
        except Exception:
            pass
        t = self.notification_display_time
        theme = self.notification_theme
        placeholder = st.empty()
        placeholder.markdown(
            f"""
            <div role="status" aria-live="polite"
                 style="
                    position: fixed; top:14px; right:14px;
                    min-width:320px; max-width:380px;
                    background:{theme['bg_gradient']};
                    color:{theme['text_color']};
                    padding:16px 18px 20px 18px;
                    border-radius:16px;
                    font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
                    box-shadow:{theme['shadow']};
                    border:{theme['border']};
                    backdrop-filter:blur(6px) saturate(140%);
                    -webkit-backdrop-filter:blur(6px) saturate(140%);
                    z-index:1000; overflow:hidden;
                    animation:notifSlideIn .45s cubic-bezier(.16,.8,.24,1);
                 ">
                <div style="display:flex;align-items:flex-start;gap:14px;">
                    <div style="
                        flex-shrink:0;width:50px;height:50px;border-radius:14px;
                        background:radial-gradient(circle at 35% 35%, {theme['accent']} 0%, #ff9d00 60%, #d97800 100%);
                        box-shadow:0 6px 14px -2px rgba(0,0,0,0.45);
                        display:flex;align-items:center;justify-content:center;
                        font-size:22px;font-weight:600;color:#2d2d2d;
                        text-shadow:0 1px 2px rgba(255,255,255,0.45);
                    ">⚠</div>
                    <div style="flex:1;line-height:1.35;">
                        <div style="font-size:16px;font-weight:600;letter-spacing:.35px;margin-bottom:4px;">
                            {phrase}
                        </div>
                        <div style="font-size:13.5px;font-weight:500;opacity:.92;">
                            {desc}
                        </div>
                    </div>
                </div>
                <div style="
                    position:absolute;left:0;top:0;height:4px;width:100%;
                    background:linear-gradient(90deg,{theme['accent']},#ffe68a);
                    animation:notifProgress {t}s linear forwards;
                    box-shadow:0 0 0 1px rgba(0,0,0,0.18) inset;
                "></div>
                <div style="
                    position:absolute;inset:0;pointer-events:none;
                    background:
                        radial-gradient(circle at 85% 18%,rgba(255,255,255,0.28) 0%,rgba(255,255,255,0) 60%),
                        radial-gradient(circle at 18% 85%,rgba(255,255,255,0.18) 0%,rgba(255,255,255,0) 65%);
                    mix-blend-mode:overlay;
                "></div>
            </div>
            <style>
                @keyframes notifSlideIn {{
                    0% {{ transform:translateY(-20px) scale(.95); opacity:0; filter:blur(6px); }}
                    55% {{ transform:translateY(5px) scale(1.015); }}
                    100% {{ transform:translateY(0) scale(1); opacity:1; filter:blur(0); }}
                }}
                @keyframes notifProgress {{
                    0% {{ width:100%; opacity:1; }}
                    90% {{ width:0%; opacity:1; }}
                    100% {{ width:0%; opacity:.12; }}
                }}
            </style>
            """,
            unsafe_allow_html=True
        )
        st.session_state[f"notif_{phrase}_ts"] = time.time()

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
        if force:
            os._exit(0)
        raise SystemExit("Streamlit server terminated.")
