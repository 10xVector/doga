from dotenv import load_dotenv
load_dotenv()
import os
import openai
import json
from moviepy.editor import *
from PIL import Image
from google.cloud import texttospeech

# --- CONFIG ---
openai.api_key = os.getenv("OPENAI_API_KEY")
OPEN_MOUTH = "avatar_open_mouth.png"
CLOSE_MOUTH = "avatar_close_mouth.png"
BACKGROUND = "background.png"
FONT_PATH = "/Library/Fonts/Arial.ttf"  # Update this path for your OS

# --- Step 1: Get lesson content from GPT ---
def generate_lesson():
    prompt = """
    Create a short Japanese lesson. Return a JSON object with exactly these keys:
    - japanese: a Japanese sentence
    - english: the English translation
    - explanation: a one-line explanation
    Example:
    {"japanese": "...", "english": "...", "explanation": "..."}
    """
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    content = response.choices[0].message.content
    print("GPT Output:", content)  # Debug print
    try:
        lesson = json.loads(content)
    except Exception as e:
        print("JSON decode error:", e)
        raise

    # If nested, extract the first value
    if isinstance(lesson, dict) and len(lesson) == 1 and isinstance(list(lesson.values())[0], dict):
        lesson = list(lesson.values())[0]

    # Normalize keys: lowercase, remove underscores and spaces, etc.
    norm = {k.lower().replace("_", "").replace(" ", ""): v for k, v in lesson.items()}

    # Map to expected keys, with fallbacks
    return {
        "japanese": norm.get("japanese", norm.get("japanesesentence", norm.get("sentence", ""))),
        "english": norm.get("english", norm.get("englishtranslation", norm.get("translation", ""))),
        "explanation": norm.get("explanation", "")
    }

# --- Step 2: Generate TTS audio ---
def generate_audio(text, lang, filename):
    client = texttospeech.TextToSpeechClient()
    
    # Set the text input to be synthesized
    synthesis_input = texttospeech.SynthesisInput(text=text)
    
    # Build the voice request
    if lang == "ja":
        voice = texttospeech.VoiceSelectionParams(
            language_code="ja-JP",
            name="ja-JP-Chirp3-HD-Leda",  # High-definition female voice
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )
    else:  # English
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Neural2-F",  # Female voice
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )
    
    # Select the type of audio file
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )
    
    # Perform the text-to-speech request
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )
    
    # Write the response to the output file
    with open(filename, "wb") as out:
        out.write(response.audio_content)

# --- Helper: Create talking animation ---
def make_talking_animation(start, duration, height=720, interval=0.25):
    clips = []
    t = 0
    toggle = True
    height = int(height)
    # Get the original image size to calculate width
    from PIL import Image as PILImage
    img_sample = PILImage.open(OPEN_MOUTH)
    orig_w, orig_h = img_sample.size
    width = int(orig_w * (height / orig_h))
    size = (width, height)
    while t < duration:
        img = OPEN_MOUTH if toggle else CLOSE_MOUTH
        seg_duration = min(interval, duration - t)
        clips.append(
            ImageClip(img)
            .set_start(start + t)
            .set_duration(seg_duration)
            .resize(newsize=size)
            .set_position("center")
        )
        t += seg_duration
        toggle = not toggle
    return clips

# --- Step 3: Create the video ---
def create_video(jp_text, en_text, explanation):
    generate_audio(jp_text, 'ja', "jp.mp3")
    generate_audio(en_text, 'en', "en.mp3")

    audio_jp = AudioFileClip("jp.mp3")
    audio_en = AudioFileClip("en.mp3")
    total_duration = audio_jp.duration + audio_en.duration + 2

    background = ImageClip(BACKGROUND).set_duration(total_duration)

    # Avatar talking animation
    avatar_jp_clips = make_talking_animation(0, audio_jp.duration)
    avatar_en_clips = make_talking_animation(audio_jp.duration, audio_en.duration)
    # Ensure integer size for neutral avatar
    from PIL import Image as PILImage
    img_sample = PILImage.open(CLOSE_MOUTH)
    orig_w, orig_h = img_sample.size
    height = 720
    width = int(orig_w * (height / orig_h))
    size = (width, height)
    avatar_neutral = ImageClip(CLOSE_MOUTH).set_start(audio_jp.duration + audio_en.duration).set_duration(2).resize(newsize=size).set_position("center")

    txt_jp = TextClip(jp_text, fontsize=60, color='white', font=FONT_PATH).set_position(("center", 600)).set_duration(audio_jp.duration)
    txt_en = TextClip(en_text, fontsize=50, color='white', font=FONT_PATH).set_position(("center", 680)).set_start(audio_jp.duration).set_duration(audio_en.duration)
    txt_ex = TextClip(explanation, fontsize=40, color='yellow', font=FONT_PATH).set_position(("center", 50)).set_duration(total_duration)

    full_audio = concatenate_audioclips([audio_jp, audio_en])
    video = CompositeVideoClip([
        background,
        *avatar_jp_clips,
        *avatar_en_clips,
        avatar_neutral,
        txt_jp,
        txt_en,
        txt_ex
    ]).set_audio(full_audio)
    video.write_videofile("lesson_output.mp4", fps=24)

# --- Run the Pipeline ---
if __name__ == "__main__":
    lesson = generate_lesson()
    create_video(lesson['japanese'], lesson['english'], lesson['explanation']) 