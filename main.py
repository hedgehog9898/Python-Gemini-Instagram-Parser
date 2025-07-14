import instaloader
import subprocess
import os
import glob
import cv2
import base64
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

def extract_screenshot(video_path, out_image='frame.jpg', frame_no=30):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Не удалось извлечь кадр")
    cv2.imwrite(out_image, frame)
    cap.release()
    return out_image

def base64_from_file(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def analyze_with_gemini(image_path, audio_path):
    image_data = base64_from_file(image_path)
    audio_data = base64_from_file(audio_path)
    prompt = (
        "Определи название фильмов, книг, сериалов, аниме или игр на основе следующего скриншота и аудио. "
        "Указывай ВСЕ названия, которые найдешь и как они представлены в тексте или на скриншоте. Не меняй оригинальных авторов и названия. "
        "Ответь кратко и укажи точное название. Если названий несколько, то указывай их по порядку через запятую с пробелом. "
        "Отвечай по шаблону: название1, название2, название3. "
        "Если не уверен — скажи 'Не удалось определить'."
    )
    response = model.generate_content([
        prompt,
        {"mime_type": "image/jpeg", "data": image_data},
        {"mime_type": "audio/wav", "data": audio_data}
    ])
    return response.text

def process_url(url):
    video_file = download_reel(url)
    extract_audio(video_file, 'audio.wav')
    extract_screenshot(video_file, 'frame.jpg')
    result = analyze_with_gemini('frame.jpg', 'audio.wav')
    return {"detected": result.strip(), "source": url}

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
