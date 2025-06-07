from dotenv import load_dotenv
load_dotenv()
import os
import json
import moviepy.video.io.ImageSequenceClip
from moviepy.video.VideoClip import ImageClip, TextClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.video.VideoClip import ColorClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.audio.AudioClip import CompositeAudioClip
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
    - extra: additional information such as common mistakes, related grammar points, or usage tips (1-2 sentences, in English, after the examples)

    Please encourage variety and novelty in your choice of grammar point. If a common grammar point (like 〜ながら, 〜たい, 〜ている) has not been used yet, it is okay to use it, but otherwise try to pick a less common, more casual, or more conversational grammar point or word. Avoid repeating grammar points within the same session. Include casual or conversational Japanese when possible.

    Example:
    {"grammar": "...", "explanation": "...", "japanese": "...", "english": "...", "extra": "..."}
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
        "english": norm.get("english", norm.get("englishtranslation", norm.get("translation", ""))),
        "extra": norm.get("extra", "")
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
    response_json = response.json()
    if "candidates" not in response_json or "content" not in response_json["candidates"][0]:
        print("Gemini TTS API error or unexpected response:")
        print(json.dumps(response_json, indent=2))
        raise RuntimeError("Gemini TTS API did not return expected audio content.")
    audio_b64 = response_json["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]
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

def concatenate_videoclips(clips):
    t = 0
    new_clips = []
    for clip in clips:
        new_clip = clip.with_start(t)
        new_clips.append(new_clip)
        t += clip.duration
    return CompositeVideoClip(new_clips, size=clips[0].size)

# --- Step 3: Create the video ---
def create_video(lesson):
    # --- INTRO CLIP ---
    # Generate TTS for the avatar's intro line
    intro_line = "Grammar tip!"
    generate_audio(intro_line, 'en', "intro_tts.wav", voice_name='Kore')
    intro_voice = AudioFileClip("intro_tts.wav")
    intro_sfx = AudioFileClip("intro_sound.mp3").subclipped(0, intro_voice.duration)
    intro_audio = CompositeAudioClip([intro_voice, intro_sfx]).with_duration(intro_voice.duration)
    intro_duration = intro_voice.duration
    intro_text = TextClip(
        text="日本語文法のヒント！",  # "Here's a Japanese grammar tip!"
        font="/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        font_size=100,
        color="white",
        size=(1280, 200),
        method="caption"
    ).with_position(("center", "center")).with_duration(intro_duration)
    intro_bg = ColorClip(size=(1280, 720), color=(30, 144, 255)).with_duration(intro_duration)  # Dodger blue
    intro_avatar = ImageClip(CLOSE_MOUTH).with_duration(intro_duration).with_position(("center", 400)).resized(height=300)
    intro_clip = CompositeVideoClip([intro_bg, intro_avatar, intro_text]).with_audio(intro_audio)

    # --- MAIN LESSON CLIP ---
    # Build teacher-style narration: include explanation
    cta_line = "Try making your own sentence with this grammar in the comments, and follow Sekai Meetup for more Japanese tips!"
    english_narration = (
        f"Here's a Japanese grammar tip: {lesson['grammar']}. {lesson['explanation']} "
        f"Example: {lesson['japanese']}, which means {lesson['english']}. "
        f"{lesson.get('extra', '')} "
        f"{cta_line}"
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
        text=lesson['grammar'],  # Using Japanese characters
        font="/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        font_size=60,
        color="black",
        method='caption',
        size=(500, None)
    ).with_position((600, 200)).with_duration(audio.duration)

    main_clip = CompositeVideoClip([
        background,
        *avatar_clips,
        avatar_neutral,
        grammar_text
    ]).with_audio(audio)

    # --- OUTRO CLIP ---
    # (Optional) Overlay CTA text visually at the end of the main lesson
    # To do this, you could add a TextClip with .with_start(audio.duration - X) and .with_duration(Y)

    # --- CONCATENATE INTRO AND MAIN CLIP ---
    # Export intro and main clips as temporary files
    intro_clip.write_videofile("intro_temp.mp4", fps=24)
    main_clip.write_videofile("main_temp.mp4", fps=24)

    from moviepy.video.io.VideoFileClip import VideoFileClip
    intro_video = VideoFileClip("intro_temp.mp4")
    main_video = VideoFileClip("main_temp.mp4")

    # --- CONCATENATE ALL CLIPS ---
    final_video = CompositeVideoClip([
        intro_video,
        main_video.with_start(intro_video.duration)
    ], size=intro_video.size)
    final_video.write_videofile("lesson_output.mp4", fps=24)

# --- Run the Pipeline ---
if __name__ == "__main__":
    lesson = generate_lesson()
    create_video(lesson) 