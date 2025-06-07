from dotenv import load_dotenv
load_dotenv()
import os
import json
import moviepy.video.io.ImageSequenceClip
from moviepy.video.VideoClip import ImageClip, TextClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from PIL import Image
import requests
import openai
import re
import wave
import base64
import subprocess
import numpy as np

# --- CONFIG ---
OPEN_MOUTH = "avatar_open_mouth.png"
CLOSE_MOUTH = "avatar_close_mouth.png"
BACKGROUND = "background.png"
FONT_PATH = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"  # Updated to Arial Unicode for Japanese support

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# --- Step 1: Get lesson content from GPT ---
def generate_lesson():
    prompt = """
    Create a Japanese 'Grammar of the Day' lesson. Return a JSON object with exactly these keys:
    - grammar: the name of the grammar point in Japanese characters (use only Japanese script: hiragana, katakana, kanji; do not use romaji)
    - explanation: a concise explanation of the grammar point in English (1-2 sentences)
    - japanese: an example sentence in Japanese using the grammar (use only Japanese script: hiragana, katakana, kanji; do not use romaji)
    - english: the English translation of the example sentence (full sentence, no abbreviations)
    Example:
    {\"grammar\": \"...\", \"explanation\": \"...\", \"japanese\": \"...\", \"english\": \"...\"}
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    content = response.choices[0].message.content
    print("GPT Output:", content)  # Debug print
    # Remove Markdown code block if present
    content = re.sub(r"^```json\s*|```$", "", content.strip(), flags=re.MULTILINE)
    try:
        lesson = json.loads(content)
    except Exception as e:
        print("JSON decode error:", e)
        raise
    # Normalize keys: lowercase, remove underscores and spaces, etc.
    norm = {k.lower().replace("_", "").replace(" ", ""): v for k, v in lesson.items()}
    return {
        "grammar": norm.get("grammar", ""),
        "explanation": norm.get("explanation", ""),
        "japanese": norm.get("japanese", norm.get("japanesesentence", norm.get("sentence", ""))),
        "english": norm.get("english", norm.get("englishtranslation", norm.get("translation", "")))
    }

# --- Step 2: Generate TTS audio ---
def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)

def generate_audio(text, lang, filename, voice_name=None):
    import sys
    import wave
    import subprocess
    API_KEY = os.environ["GEMINI_API_KEY"]
    ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-tts:generateContent?key={API_KEY}"
    headers = {"Content-Type": "application/json"}
    if lang == 'ja':
        voice = voice_name or 'Leda'
    else:
        voice = voice_name or 'Kore'
    data = {
        "contents": [
            {"parts": [{"text": text}]}
        ],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {
                        "voiceName": voice
                    }
                }
            }
        },
        "model": "gemini-2.5-flash-preview-tts"
    }
    response = requests.post(ENDPOINT, headers=headers, json=data)
    if not response.ok:
        print("Request data:", data)
        print("Error response:", response.text)
        with open("gemini_tts_error.txt", "w") as f:
            f.write("Request data:\n")
            f.write(str(data) + "\n")
            f.write("Error response:\n")
            f.write(response.text)
        sys.stdout.flush()
        response.raise_for_status()
    audio_b64 = response.json()["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]
    pcm_data = base64.b64decode(audio_b64)
    with wave.open(filename, "wb") as wavfile:
        wavfile.setnchannels(1)
        wavfile.setsampwidth(2)  # 16-bit = 2 bytes
        wavfile.setframerate(24000)
        wavfile.writeframes(pcm_data)
    print(f"Audio saved as {filename}")
    # Convert to mp3
    mp3_filename = filename.replace('.wav', '.mp3')
    try:
        subprocess.run(["ffmpeg", "-y", "-i", filename, mp3_filename], check=True)
        print(f"Audio also saved as {mp3_filename}")
    except Exception as e:
        print("ffmpeg conversion failed:", e)

# --- Helper: Create talking animation ---
def make_talking_animation(start, duration, height=720, audio_file="en.wav"):
    clips = []
    height = int(height)
    # Get the original image size to calculate width
    from PIL import Image as PILImage
    img_sample = PILImage.open(OPEN_MOUTH)
    orig_w, orig_h = img_sample.size
    width = int(orig_w * (height / orig_h))
    size = (width, height)
    x_offset = int(width * 0.2)

    # Analyze audio for speech patterns
    from scipy.io import wavfile

    # Load audio file
    sample_rate, audio_data = wavfile.read(audio_file)
    
    # Convert to mono if stereo
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    
    # Calculate audio energy
    frame_length = int(sample_rate * 0.01)  # 10ms frames
    energy = np.array([np.sum(np.square(audio_data[i:i+frame_length])) 
                      for i in range(0, len(audio_data), frame_length)])
    
    # Normalize energy
    energy = energy / np.max(energy)
    
    # Detect speech using energy threshold
    threshold = 0.01
    speech_frames = energy > threshold

    frame_duration = 0.1  # 100ms per frame
    # Ensure number of frames matches audio duration
    audio_duration = len(audio_data) / sample_rate
    num_frames = int(audio_duration / frame_duration)
    # Pad or truncate speech_frames to match num_frames
    if len(speech_frames) < num_frames:
        speech_frames = np.pad(speech_frames, (0, num_frames - len(speech_frames)), 'constant')
    elif len(speech_frames) > num_frames:
        speech_frames = speech_frames[:num_frames]

    for i in range(num_frames):
        t = i * frame_duration
        if speech_frames[i]:
            # Open mouth during speech
            clips.append(
                ImageClip(OPEN_MOUTH)
                .with_start(start + t)
                .with_duration(frame_duration)
                .resized(width=width, height=height)
                .with_position((x_offset, "bottom"))
            )
        else:
            # Close mouth during silence
            clips.append(
                ImageClip(CLOSE_MOUTH)
                .with_start(start + t)
                .with_duration(frame_duration)
                .resized(width=width, height=height)
                .with_position((x_offset, "bottom"))
            )
    
    return clips

# --- Step 3: Create the video ---
def create_video(lesson):
    # Build teacher-style narration: include explanation
    english_narration = (
        f"Today's grammar is: {lesson['grammar']}. {lesson['explanation']} "
        f"Example: {lesson['japanese']}, which means {lesson['english']}."
    )
    # English narration (Gemini TTS)
    generate_audio(english_narration, 'en', "en.wav", voice_name='Kore')
    audio = AudioFileClip("en.wav")
    total_duration = audio.duration + 2

    background = ImageClip(BACKGROUND).with_duration(total_duration)

    # Avatar talking animation with audio sync
    avatar_clips = make_talking_animation(0, audio.duration, audio_file="en.wav")
    from PIL import Image as PILImage
    img_sample = PILImage.open(CLOSE_MOUTH)
    orig_w, orig_h = img_sample.size
    height = 720
    width = int(orig_w * (height / orig_h))
    size = (width, height)
    x_offset = int(width * 0.2)
    avatar_neutral = ImageClip(CLOSE_MOUTH).with_start(audio.duration).with_duration(2).resized(width=width, height=height).with_position((x_offset, "bottom"))

    # Japanese TTS (Gemini API)
    generate_audio(lesson['japanese'], 'ja', "ja.wav", voice_name='Leda')
    japanese_audio = AudioFileClip("ja.wav")

    # Add grammar point text to the right of the avatar
    grammar_text = TextClip(
        text=lesson['grammar'],
        font="/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        font_size=70,
        color="black",
        method='caption',
        size=(500, None)
    ).with_position((600, 200)).with_duration(audio.duration)

    video = CompositeVideoClip([
        background,
        *avatar_clips,
        avatar_neutral,
        grammar_text
    ]).with_audio(audio)

    print(f"audio.duration: {audio.duration}")
    print(f"total_duration: {total_duration}")
    print(f"background duration: {background.duration}")
    if avatar_clips:
        print(f"first avatar_clip duration: {avatar_clips[0].duration}")
        print(f"last avatar_clip duration: {avatar_clips[-1].duration}")
        print(f"first avatar_clip start: {avatar_clips[0].start}")
        print(f"last avatar_clip start: {avatar_clips[-1].start}")
    print(f"avatar_neutral duration: {avatar_neutral.duration}")
    print(f"grammar_text duration: {grammar_text.duration}")

    print(f"final video duration: {video.duration}")
    video.write_videofile("lesson_output.mp4", fps=24)

# --- Run the Pipeline ---
if __name__ == "__main__":
    lesson = generate_lesson()
    create_video(lesson) 