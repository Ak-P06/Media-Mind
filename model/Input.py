# Input.py

from PIL import Image
import cv2,numpy as np,logging
from pydub import AudioSegment
import requests,os,mimetypes
from PyPDF2 import PdfReader
from docx import Document

# Set up logging (optional, good for Flask apps)
logging.basicConfig(level=logging.INFO)

class HandleInput:

    def text_input(self, text_source: str):
        try:
            if os.path.exists(text_source):
                ext = os.path.splitext(text_source)[-1].lower()

                # TXT file
                if ext == ".txt":
                    with open(text_source, "r", encoding="utf-8") as f:
                        text = f.read()

                # PDF file
                elif ext == ".pdf":
                    reader = PdfReader(text_source)
                    text = " ".join([page.extract_text() or "" for page in reader.pages])

                # DOCX file
                elif ext == ".docx":
                    doc = Document(text_source)
                    text = " ".join([p.text for p in doc.paragraphs])

                else:
                    raise ValueError("Unsupported text file format")

                source_type = "file"
            else:
                text = text_source
                source_type = "text"

            tokens = text.split()
            word_count = len(tokens)

            return {
                "text": text,
                "tokens": tokens,
                "word_count": word_count,
                "source_type": source_type
            }

        except Exception as e:
            logging.error(f"Text Input Error: {e}")
            return {
                "status": "error",
                "message": str(e)
            }


    def image_input(self, image_path: str):
        try:
            # Step 1: Validate file type
            if not image_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                raise ValueError("Unsupported image format")

            # Step 2: Read image using PIL (safer for different formats)
            image = Image.open(image_path).convert("RGB")

            # Step 3: Convert to OpenCV format (numpy array)
            image_cv = np.array(image)
            image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

            # Step 4: Get basic metadata
            height, width, channels = image_cv.shape
            size_kb = round(os.path.getsize(image_path) / 1024, 2)

            return {
                "path": image_path,
                "resolution": (width, height),
                "channels": channels,
                "size_kb": size_kb,
                "format": image.format or image_path.split('.')[-1].upper()
            }

        except Exception as e:
            print("Error loading image:", e)
            return None
        
    def video_input(self, video_path: str):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("Unable to open video")

            # Basic metadata
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            cap.release()

            return {
                "path": video_path,
                "fps": fps,
                "frame_count": frame_count,
                "duration_sec": duration,
                "resolution": (width, height)
            }

        except Exception as e:
            print("Error reading video:", e)
            return None

    def audio_input(self, audio_path: str):
        try:
            # Load the audio file (supports mp3, wav, m4a, etc.)
            audio = AudioSegment.from_file(audio_path)

            # Standardize format: mono channel, 16kHz sample rate
            audio = audio.set_frame_rate(16000).set_channels(1)

            # Extract metadata
            duration = len(audio) / 1000.0  # milliseconds → seconds
            channels = audio.channels
            sample_rate = audio.frame_rate
            format_ = audio_path.split('.')[-1].lower()

            return {
                "audio": audio,
                "duration_sec": duration,
                "channels": channels,
                "sample_rate": sample_rate,
                "format": format_
            }

        except Exception as e:
            print("Error loading audio:", e)
            return None