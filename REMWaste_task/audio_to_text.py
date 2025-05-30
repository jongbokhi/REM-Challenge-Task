from langchain_community.document_loaders.blob_loaders.youtube_audio import (
    YoutubeAudioLoader,
)
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import OpenAIWhisperParser


def extract_audio_and_covert_to_text(url: str, audio_save_path="./youtube_audios/"):

    # YouTube URL
    urls = [url]

    # Set directory
    save_dir = audio_save_path

    # convert to text
    loader = GenericLoader(YoutubeAudioLoader(urls, save_dir), OpenAIWhisperParser())
    docs = loader.load()

    # Combine documents
    combined_docs = [doc.page_content for doc in docs]
    text = " ".join(combined_docs)

    return text
