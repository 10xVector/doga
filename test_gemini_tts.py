from dotenv import load_dotenv
load_dotenv()
import requests
import os
import base64
import json
import subprocess
import wave

API_KEY = os.environ.get("GEMINI_API_KEY")
ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-tts:generateContent?key={API_KEY}"
headers = {"Content-Type": "application/json"}
data = {
    "contents": [
        {"parts": [{"text": "こんにちは、今日は日本語の文法を学びましょう！"}]}
    ],
    "generationConfig": {
        "responseModalities": ["AUDIO"],
        "speechConfig": {
            "voiceConfig": {
                "prebuiltVoiceConfig": {
                    "voiceName": "Leda"
                }
            }
        }
    },
    "model": "gemini-2.5-flash-preview-tts"
}

response = requests.post(ENDPOINT, headers=headers, json=data)
print("Status code:", response.status_code)
print("Response text:", response.text)
try:
    print("Raw JSON:", json.dumps(response.json(), indent=2, ensure_ascii=False))
except Exception as e:
    print("Could not parse JSON:", e)

# If successful, save the audio as a valid WAV and convert to mp3
if response.ok:
    audio_b64 = response.json()["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]
    pcm_data = base64.b64decode(audio_b64)
    with wave.open("output.wav", "wb") as wavfile:
        wavfile.setnchannels(1)
        wavfile.setsampwidth(2)  # 16-bit = 2 bytes
        wavfile.setframerate(24000)
        wavfile.writeframes(pcm_data)
    print("Audio saved as output.wav")
    # Convert to mp3
    try:
        subprocess.run(["ffmpeg", "-y", "-i", "output.wav", "output.mp3"], check=True)
        print("Audio also saved as output.mp3")
    except Exception as e:
        print("ffmpeg conversion failed:", e) 