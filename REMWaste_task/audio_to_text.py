import os
from datetime import datetime
from langchain_community.document_loaders.blob_loaders.youtube_audio import (
    YoutubeAudioLoader,
)
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import OpenAIWhisperParser


def extract_audio_and_convert_to_text(
    url: str,
    audio_save_path: str = "./youtube_audios/",
    text_save_path: str = "./converted_texts/",
) -> str:
    """
    Extracts audio from a YouTube video, converts it to text using Whisper,
    and saves the result as a .txt file in the specified directory.

    Parameters:
        url (str): The URL of the YouTube video
        audio_save_path (str): Directory where the downloaded audio will be saved (default: ./youtube_audios/)
        text_save_path (str): Directory where the converted text file will be saved (default: ./converted_texts/)

    Returns:
        str: The full transcribed text (returns an empty string if an error occurs)
    """
    try:
        # Create directories if they don't exist
        os.makedirs(audio_save_path, exist_ok=True)
        os.makedirs(text_save_path, exist_ok=True)

        # Prepare URL list
        urls = [url]

        # Download audio and transcribe to text
        loader = GenericLoader(
            YoutubeAudioLoader(urls, audio_save_path), OpenAIWhisperParser()
        )
        docs = loader.load()

        # Combine text content from all documents
        combined_docs = [doc.page_content for doc in docs]
        text = " ".join(combined_docs)

        # Generate filename using timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = f"result_{timestamp}.txt"
        result_path = os.path.join(text_save_path, result_filename)

        # Save transcribed text to file
        with open(result_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"[Done] Text has been saved to '{result_path}'.")

        return text

    except Exception as e:
        print(f"[Error] {e}")

        return ""
