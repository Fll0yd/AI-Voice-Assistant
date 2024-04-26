import pandas as pd
import os
import json
import librosa
from PIL import Image
from bs4 import BeautifulSoup
import cv2
import fitz
from concurrent.futures import ThreadPoolExecutor

def load_file(file_path, file_handlers):
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext in file_handlers:
        handler = file_handlers[ext]
        try:
            return handler(file_path)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    else:
        print(f"Unsupported file format: {file_path}")
        return None

def extract_image_features(image_path):
    try:
        image = cv2.imread(image_path)
        if image is not None:
            return {"image_path": image_path, "image_shape": image.shape}
        else:
            print(f"Error loading image: {image_path}")
            return None
    except Exception as e:
        print(f"Error extracting image features: {e}")
        return None

def extract_video_features(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_sec = frame_count / cap.get(cv2.CAP_PROP_FPS)
            return {"video_path": video_path, "frame_count": frame_count, "duration_sec": duration_sec}
        else:
            print(f"Error opening video: {video_path}")
            return None
    except Exception as e:
        print(f"Error extracting video features: {e}")
        return None

def extract_pdf_text(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None

def load_data_from_sources(data_directory, file_handlers, max_workers=1):
    combined_data = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for root, dirs, files in os.walk(data_directory):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.isfile(file_path):
                    futures.append(executor.submit(load_file, file_path, file_handlers))

        for future in concurrent.futures.as_completed(futures):
            data = future.result()
            if data is not None:
                combined_data.append(data)

    return combined_data

def preprocess_data(data):
    # Placeholder for data preprocessing steps
    # Modify as per your specific preprocessing requirements
    return data

def format_data_for_training(data):
    # Placeholder for data formatting steps
    # Modify as per your specific data formatting requirements
    return data

def save_formatted_data(data, output_file_path):
    with open(output_file_path, 'w') as outfile:
        json.dump(data, outfile, indent=4)

def extract_sti_features(sti_path):
    try:
        with open(sti_path, 'r', encoding='utf-8') as sti_file:
            sti_data = sti_file.read()
            # Process STI file data
            return {"sti_path": sti_path, "sti_data": sti_data}
    except Exception as e:
        print(f"Error extracting features from .STI file: {e}")
        return None

def main():
    FILE_HANDLERS = {
        ".csv": lambda file_path: pd.read_csv(file_path, delimiter=',', encoding='utf-8', error_bad_lines=False),
        ".json": lambda file_path: json.load(open(file_path, 'r', encoding='utf-8-sig')),
        ".xlsx": pd.read_excel,
        ".wav": lambda file_path: librosa.load(file_path, sr=None),
        ".mp3": lambda file_path: librosa.load(file_path, sr=None),
        ".html": lambda file_path: BeautifulSoup(open(file_path, 'r', encoding='utf-8').read(), 'html.parser'),
        ".txt": lambda file_path: open(file_path, 'r', encoding='utf-8').read(),
        ".jpg": extract_image_features,
        ".jpeg": extract_image_features,
        ".png": extract_image_features,
        ".mp4": extract_video_features,
        ".avi": extract_video_features,
        ".pdf": extract_pdf_text,
        ".gif": lambda file_path: Image.open(file_path),
        ".eml": lambda file_path: open(file_path, 'r', encoding='utf-8').read(),
        ".sti": extract_sti_features
    }

    data_directory = r"E:\Code\Flloyd\Data"
    max_workers = 8
    combined_data = load_data_from_sources(data_directory, FILE_HANDLERS, max_workers=max_workers)

    if not combined_data:
        print("No data loaded. Check if supported files are present in the specified directory.")
        return

    preprocessed_data = preprocess_data(combined_data)
    formatted_data = format_data_for_training(preprocessed_data)

    output_file_path = r"E:\Code\Flloyd\ProcessedData\formatted_data.json"
    save_formatted_data(formatted_data, output_file_path)

    print("Data preprocessing and formatting completed. Formatted data saved to:", output_file_path)

if __name__ == "__main__":
    main()
