# preprocess.py

import os,cv2,subprocess,tempfile
import nltk,numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from nltk.tokenize import word_tokenize,sent_tokenize
from pydub import AudioSegment
import pytesseract
from PIL import Image


nltk.download('punkt_tab')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class Preprocessor:

    def preprocess_text(self, text_data: dict):
        text = text_data["text"]

        # 1. Lowercase
        text = text.lower()

        # 2. Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))

        # 3. Tokenize
        tokens = word_tokenize(text)

        # 4. Remove stopwords
        stop_words = set(stopwords.words("english"))
        tokens = [word for word in tokens if word not in stop_words]

        # 5. Lemmatization
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

        # 6. Create final dictionary
        processed_text = {
            "raw_text": text_data["text"],
            "char_count": len(text_data["text"]),
            "clean_text": " ".join(tokens),
            "tokens": tokens,
            "token_count": len(tokens),
            "sentences": sent_tokenize(text_data["text"]),
            "source_type": text_data["source_type"]
        }

        return processed_text

    def preprocess_image(self, image_data: dict):
        try:
            image_path = image_data["path"]

            img = cv2.imread(image_path)
            if img is None:
                raise Exception("Could not load image")

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(img_rgb)

            resized = cv2.resize(img_rgb, (224, 224))
            normalized = resized.astype("float32") / 255.0

            avg_color = np.mean(normalized, axis=(0, 1)).tolist()
            brightness = float(np.mean(normalized))

            # ---------- OCR (safe) ----------
            try:
                gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
                ocr_text = pytesseract.image_to_string(gray).strip()
            except:
                ocr_text = ""

            return {
                "path": image_path,
                "image_arr":img_rgb,
                "original_resolution": image_data["resolution"],
                "format": image_data["format"],
                "size_kb": image_data["size_kb"],
                "pil_image":pil_image,
                "avg_color": avg_color,
                "brightness": brightness,
                "ocr_text": ocr_text,
            }

        except Exception as e:
            print("Error preprocessing image:", e)
            return {
                "status": "error",
                "message": str(e)
            }

        
    def preprocess_audio(self, audio_data: dict):
        try:
            audio: AudioSegment = audio_data["audio"]

            # Convert pydub AudioSegment to raw samples (numpy array)
            samples = np.array(audio.get_array_of_samples()).astype(np.float32)
            if audio.channels > 1:
                samples = samples.reshape((-1, audio.channels))
                samples = np.mean(samples, axis=1)
            
            
            # Normalize waveform (-1 to 1)
            max_val = np.max(np.abs(samples))
            if max_val==0:
                normalized = np.zeros_like(samples, dtype=np.float32)
            else:
                normalized = samples / max_val

            # RMS loudness
            rms = float(np.sqrt(np.mean(normalized**2)))

            # Zero Crossing Rate
            zero_crossings = np.mean(
                np.abs(np.diff(np.sign(normalized)))
            )

            return {
                "duration_sec": audio_data["duration_sec"],
                "sample_rate": audio_data["sample_rate"],
                "format": audio_data["format"],

                "rms_loudness": rms,
                "zero_crossing_rate": float(zero_crossings),

                # waveform for ML
                "waveform": normalized[:16000]  # first 1 sec only
            }

        except Exception as e:
            print("Error preprocessing audio:", e)
            return {"status": "error", "message": str(e)}

    def segment_audio_silence(self, waveform, sample_rate, rms_threshold=0.01, window_ms=100):
        """
        Split audio into sound/silence segments.
        - waveform: 1D np.array normalized (-1 to 1)
        - sample_rate: audio sampling rate
        - rms_threshold: RMS below this is silence
        - window_ms: window size for RMS computation
        """
        window_size = int(sample_rate * window_ms / 1000)
        segments = []

        current_label = None
        segment_start = 0

        for i in range(0, len(waveform), window_size):
            window = waveform[i:i + window_size]
            if len(window) == 0:
                continue

            rms = np.sqrt(np.mean(window ** 2))
            label = "silence" if rms < rms_threshold else "sound"

            if label != current_label:
                if current_label is not None:
                    segments.append({
                        "start_sec": segment_start,
                        "end_sec": i / sample_rate,
                        "type": current_label
                    })
                segment_start = i / sample_rate
                current_label = label

        # Close last segment
        segments.append({
            "start_sec": segment_start,
            "end_sec": len(waveform) / sample_rate,
            "type": current_label
        })

        return segments
    
    from pydub import AudioSegment

    def extract_audio(self,video_path):
        # Temporary audio file
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        # Use ffmpeg to extract audio as wav
        command = f'ffmpeg -i "{video_path}" -vn -acodec pcm_s16le -ar 16000 -ac 1 "{temp_path}" -y'
        os.system(command)
        
        audio = AudioSegment.from_wav(temp_path)
        return {
            "audio": audio,
            "duration_sec": len(audio)/1000,
            "sample_rate": audio.frame_rate,
            "channels": audio.channels,
            "format": "wav"
        }

    def preprocess_video(self, video_data: dict, frame_sample_sec: int = 1):
        # --- Open video ---
        cap = cv2.VideoCapture(video_data["path"])
        if not cap.isOpened():
            return {"status": "error", "message": "Cannot open video"}

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        frames = []
        motion_curve = []
        prev_gray = None
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % int(fps * frame_sample_sec) != 0:
                frame_idx += 1
                continue

            # --- RGB frame ---
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(frame_rgb, (224, 224))
            normalized = resized.astype(np.float32) / 255.0

            # --- Gray for motion ---
            gray = cv2.cvtColor((resized * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                motion_curve.append(float(np.mean(diff)))
            prev_gray = gray

            # --- OCR ---
            gray_thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]
            ocr_text = pytesseract.image_to_string(gray_thresh, config="--psm 6").strip()

            # --- Convert to PIL for embeddings ---
            from PIL import Image
            pil_frame = Image.fromarray(resized)  # 0-255 integers

            # --- Store frame info ---
            avg_color = np.mean(normalized, axis=(0, 1)).tolist()
            brightness = float(np.mean(normalized))
            frames.append({
                "timestamp_sec": frame_idx / fps,
                "frame_index": frame_idx,
                "frame": pil_frame,       # PIL Image for CLIP embedding
                "array": normalized,      # optional, for motion/analysis
                "avg_color": avg_color,
                "brightness": brightness,
                "ocr_text": ocr_text
            })

            frame_idx += 1

        cap.release()

        # --- Motion-based segmentation ---
        segments = []
        if motion_curve:
            threshold = np.mean(motion_curve) + np.std(motion_curve)
            start_sec = 0
            for i, m in enumerate(motion_curve):
                if m > threshold:
                    segments.append({
                        "start_sec": start_sec,
                        "end_sec": i,
                        "frame_indices": list(range(start_sec, i + 1))
                    })
                    start_sec = i + 1
            segments.append({
                "start_sec": start_sec,
                "end_sec": len(motion_curve),
                "frame_indices": list(range(start_sec, len(motion_curve)))
            })

        # --- Extract audio using ffmpeg ---
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            audio_path = tmp.name

        cmd = [
            "ffmpeg",
            "-i", video_data["path"],
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            audio_path,
            "-y"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        audio = None
        if result.returncode == 0 and os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:   
            audio = AudioSegment.from_wav(audio_path)

        if audio is not None:
            waveform = np.array(audio.get_array_of_samples()).astype(np.float32)
            waveform /= np.max(np.abs(waveform)) if np.max(np.abs(waveform)) != 0 else 1.0
            sample_rate = audio.frame_rate
        else:
            waveform = np.zeros(16000)
            sample_rate = 16000

        os.remove(audio_path)

        return {
            "fps": fps,
            "duration_sec": duration,
            "frame_count": total_frames,
            "frames": frames,
            "motion_curve": motion_curve,
            "segments": segments,
            "audio": {
                "waveform": waveform,
                "sample_rate": sample_rate
            }
        }

if __name__ == "__main__":
    from preprocess import Preprocessor
    from Input import HandleInput
    handler = HandleInput()
    prepro = Preprocessor()
    video_data = handler.video_input(r"C:\Users\VAIBHAV\Desktop\TY sem6 Project\Insightify\data\sample.mp4")
    video_processed = prepro.preprocess_video(video_data)
    print(video_processed["frames"][0].keys())