# Japanese Lesson Video Generator

This project automatically generates short-form Japanese language learning videos with synchronized audio, text, and avatar animations. The videos are designed to be engaging and educational, perfect for social media platforms like YouTube, Instagram, and TikTok.

## Features

- AI-generated Japanese language lessons using GPT-4
- High-quality text-to-speech for both Japanese and English
- Customizable VTuber-style avatar animations
- Automatic video generation with synchronized audio and text
- Support for subtitles and explanations

## Prerequisites

- Python 3.11 or later
- Google Cloud account with Text-to-Speech API enabled
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/japanese-lesson-generator.git
cd japanese-lesson-generator
```

2. Create and activate a virtual environment:
```bash
python -m venv venv311
source venv311/bin/activate  # On Windows: venv311\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the project root with:
```
OPENAI_API_KEY=your_openai_api_key
GOOGLE_APPLICATION_CREDENTIALS=path/to/your/google_credentials.json
```

## Required Assets

Place the following image files in the project root:
- `avatar_open_mouth.png` - Avatar image with open mouth
- `avatar_close_mouth.png` - Avatar image with closed mouth
- `background.png` - Background image for the video

## Usage

Run the main script:
```bash
python lesson_video_generator.py
```

This will:
1. Generate a Japanese lesson using GPT-4
2. Create audio files for both Japanese and English
3. Generate a video with synchronized avatar animations and text
4. Save the output as `lesson_output.mp4`

## Customization

You can customize various aspects of the video:
- Change the avatar images
- Modify the background
- Adjust text positions and styles
- Change the TTS voices

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for GPT-4 and TTS capabilities
- Google Cloud for high-quality Japanese TTS
- MoviePy for video processing 