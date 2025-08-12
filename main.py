import shutil

import instaloader
import subprocess
import os
import glob
import cv2
import base64
import json
import io
from PIL import Image
import imagehash
from dotenv import load_dotenv
import google.generativeai as genai
from flask import Flask, request, jsonify

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY не найден в .env")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.5-pro")

app = Flask(__name__)

def download_reel(url, target_folder='reels'):
    L = instaloader.Instaloader(dirname_pattern=target_folder)
    shortcode = url.split("/")[-2]
    post = instaloader.Post.from_shortcode(L.context, shortcode)
    L.download_post(post, target=target_folder)
    files = glob.glob(os.path.join(target_folder, '*.mp4'))
    if not files:
        raise FileNotFoundError("Видео не найдено")
    return max(files, key=os.path.getctime)

def extract_audio(video_path, audio_path='audio.wav'):
    subprocess.call(['ffmpeg', '-i', video_path, '-ar', '16000', '-ac', '1', audio_path, '-y'])

def frame_to_bytes(frame):
    is_success, buffer = cv2.imencode(".jpg", frame)
    if not is_success:
        raise RuntimeError("Ошибка кодирования кадра")
    return io.BytesIO(buffer)

def extract_frames_every_second(video_path, fps=1):
    cap = cv2.VideoCapture(video_path)
    frames = []
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0:
        raise RuntimeError("Не удалось получить FPS видео")
    frame_interval = int(video_fps / fps)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frames.append(frame_to_bytes(frame))
        frame_count += 1

    cap.release()
    return frames

def remove_duplicate_frames_in_memory(frames_bytes, hash_size=8, similarity_threshold=5):
    unique_frames = []
    seen_hashes = []

    for frame_io in frames_bytes:
        frame_io.seek(0)
        img = Image.open(frame_io)
        h = imagehash.phash(img, hash_size=hash_size)

        if not any(abs(h - prev_h) <= similarity_threshold for prev_h in seen_hashes):
            unique_frames.append(frame_io)
            seen_hashes.append(h)

    return unique_frames

def base64_from_bytesio(bio):
    bio.seek(0)
    return base64.b64encode(bio.read()).decode()

def analyze_with_gemini_multiple(frames_bytes, audio_path):
    images_data = [{"mime_type": "image/jpeg", "data": base64_from_bytesio(frame_io)} for frame_io in frames_bytes]
    audio_data = base64_from_file(audio_path)

    prompt = (
        "Определи название фильмов, книг, сериалов, аниме или игр на основе следующих скриншотов и аудио. "
        "Указывай ВСЕ названия, которые найдешь и как они представлены в тексте или на скриншоте. "
        "Ответь кратко и укажи точное название. Если названий несколько, указывай их по порядку через запятую с пробелом. "
        "Также укажи тип: Anime, Book, Film, Series, Game. "
        "Отвечай в формате JSON, например [{ \"name\": \"название1\", \"type\": \"Book\" }]. "
        "Если ничего не найдено, ответь []"
    )

    response = model.generate_content([
        prompt,
        *images_data,
        {"mime_type": "audio/wav", "data": audio_data}
    ])

    raw_text = response.text.strip()

    if raw_text.startswith("```"):
        raw_text = "\n".join(raw_text.split("\n")[1:-1])

    try:
        return json.loads(raw_text)
    except Exception as e:
        raise ValueError(f"Ошибка парсинга JSON: {e}\nОтвет Gemini: {response.text}")

def base64_from_file(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def process_url(url):
    video_file = download_reel(url)
    extract_audio(video_file, 'audio.wav')

    frames_bytes = extract_frames_every_second(video_file, fps=1)
    unique_frames = remove_duplicate_frames_in_memory(frames_bytes, hash_size=8, similarity_threshold=5)

    # Для дебага сохраняем кадры
    # debug_dir = 'debug_frames'
    # os.makedirs(debug_dir, exist_ok=True)
    # for i, frame_io in enumerate(unique_frames):
    #     frame_io.seek(0)
    #     with open(os.path.join(debug_dir, f'unique_frame_{i}.jpg'), 'wb') as f:
    #         f.write(frame_io.read())

    # Удаляем аудио
    if os.path.exists('audio.wav'):
        os.remove('audio.wav')

    # Удаляем папку reels, если пустая
    reels_folder = os.path.dirname(video_file)
    if os.path.exists(reels_folder):
        shutil.rmtree(reels_folder)

    results = analyze_with_gemini_multiple(unique_frames, 'audio.wav')
    return {"results": results, "source": url}

@app.route('/parse', methods=['POST'])
def parse():
    data = request.json
    url = data.get('url')
    if not url:
        return jsonify({"error": "URL is required"}), 400
    try:
        result = process_url(url)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True, threaded=True)
