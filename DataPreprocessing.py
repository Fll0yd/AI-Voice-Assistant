import pandas as pd
import os
import json
import librosa
from PIL import Image
from bs4 import BeautifulSoup
import cv2  # OpenCV for video processing
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import numpy as np

# Define supported file extensions and corresponding data handlers
FILE_HANDLERS = {
    ".csv": pd.read_csv,
    ".json": lambda file_path: json.load(open(file_path, 'r')),
    ".xlsx": pd.read_excel,
    ".wav": librosa.load,
    ".mp3": librosa.load,
    ".html": lambda file_path: parse_html(open(file_path, 'r', encoding='utf-8').read()),
    ".htm": lambda file_path: parse_html(open(file_path, 'r', encoding='utf-8').read()),
    ".txt": lambda file_path: open(file_path, 'r', encoding='utf-8').read(),
    ".eml": lambda file_path: open(file_path, 'r', encoding='utf-8').read(),
    ".jpg": lambda file_path: extract_image_features(Image.open(file_path)),
    ".jpeg": lambda file_path: extract_image_features(Image.open(file_path)),
    ".png": lambda file_path: extract_image_features(Image.open(file_path)),
    ".mp4": lambda file_path: extract_video_features(file_path),
    ".avi": lambda file_path: extract_video_features(file_path)
}

def load_data_from_sources(data_directory):
    data_frames = []

    with ThreadPoolExecutor() as executor:
        future_to_file = {executor.submit(load_file, os.path.join(root, file)): os.path.join(root, file)
                          for root, _, files in os.walk(data_directory) for file in files}

        for future in concurrent.futures.as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                result = future.result()
                if result is not None:
                    data_frames.append(result)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    return data_frames

def load_file(file_path):
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext in FILE_HANDLERS:
        handler = FILE_HANDLERS[ext]
        try:
            if callable(handler):
                return handler(file_path)
            else:
                return handler(file_path)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    else:
        print(f"Unsupported file format: {file_path}")

def preprocess_data(data_frames):
    # Perform data preprocessing (e.g., remove NaN values)
    preprocessed_data = [df.dropna() for df in data_frames if isinstance(df, pd.DataFrame)]
    return preprocessed_data

def format_data_for_training(data_frames):
    # Example: Format data for training (customize based on use case)
    formatted_data = []

    for df in data_frames:
        if isinstance(df, pd.DataFrame):
            formatted_entry = {
                "input_features": df.get("input_column_name"),
                "output_features": df.get("output_column_name")
            }
            formatted_data.append(formatted_entry)

    return formatted_data

def save_formatted_data(formatted_data, output_path):
    # Save formatted data to JSON file
    with open(output_path, 'w') as f:
        json.dump(formatted_data, f, indent=4)

def extract_audio_features(audio_data, sample_rate):
    # Example: Extract MFCC features from audio
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate)
    return mfcc.tolist()

def parse_html(html_content):
    # Parse HTML content and extract text
    soup = BeautifulSoup(html_content, 'html.parser')
    text_content = soup.get_text()
    return text_content

def extract_image_features(image):
    # Example: Extract image features (e.g., color histogram)
    resized_image = image.resize((224, 224))
    histogram = resized_image.histogram()
    return histogram

def extract_video_features(video_path):
    # Example: Extract video features (e.g., frame-level color histograms)
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    histograms = []

    for _ in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        histogram = cv2.calcHist([frame], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
        histograms.append(histogram.flatten().tolist())

    cap.release()
    return histograms

def main():
    data_directory = r"E:\Code\Flloyd\Data"
    combined_data = load_data_from_sources(data_directory)

    if not combined_data:
        print("No data loaded. Check if supported files are present in the specified directory.")
        return

    preprocessed_data = preprocess_data(combined_data)
    formatted_data = format_data_for_training(preprocessed_data)

    output_file_path = "formatted_data.json"
    save_formatted_data(formatted_data, output_file_path)

    print("Data preprocessing and formatting completed. Formatted data saved to:", output_file_path)

if __name__ == "__main__":
    main()
