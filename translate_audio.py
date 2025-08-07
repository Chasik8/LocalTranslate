import fitz  # PyMuPDF
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os
import torch
import soundfile as sf
from tqdm import tqdm
import traceback
import spacy  # For intonation processing

# Load spaCy model for Russian language
nlp = spacy.load("ru_core_news_sm")

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        return f"Error processing PDF: {e}"

def translate_text_nllb(text_to_translate, model_name, src_lang_code, tgt_lang_code):
    """Translates text using the NLLB multilingual model."""
    try:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Using device for translation: {device}")

        tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=src_lang_code)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

        chunk_size = 350
        chunks = [text_to_translate[i:i + chunk_size] for i in range(0, len(text_to_translate), chunk_size)]

        translated_chunks = []
        print("Starting text translation...")
        for chunk in tqdm(chunks, desc="Translation"):
            inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            target_lang_id = tokenizer.convert_tokens_to_ids(tgt_lang_code)
            translated_ids = model.generate(**inputs, forced_bos_token_id=target_lang_id, max_length=512)
            translated_text = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
            translated_chunks.append(translated_text)

        return " ".join(translated_chunks)
    except Exception as e:
        print(f"Translation error: {e}")
        traceback.print_exc()
        return None

def synthesize_speech_silero(text_to_synthesize, output_audio_path, speaker='xenia', model_id='v4_ru'):
    """Synthesizes speech from text using Silero models and saves it as a WAV file."""
    try:
        print("\n3. Starting speech synthesis process...")
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            print('cuda')
        else:
            device = torch.device('cpu')
            torch.set_num_threads(4)

        model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                  model='silero_tts',
                                  language='ru',
                                  speaker=model_id,
                                  trust_repo=True)
        model.to(device)

        sample_rate = 48000
        text_parts = []
        max_len = 950
        current_text = text_to_synthesize
        while len(current_text) > 0:
            if len(current_text) <= max_len:
                text_parts.append(current_text)
                break
            split_at = current_text.rfind('.', 0, max_len)
            if split_at == -1:
                split_at = current_text.rfind(' ', 0, max_len)
            if split_at == -1:
                split_at = max_len
            text_parts.append(current_text[:split_at + 1])
            current_text = current_text[split_at + 1:].lstrip()

        audio_chunks = []
        for part in tqdm(text_parts, desc="Synthesizing chunks"):
            if not part.strip():
                continue
            audio = model.apply_tts(text=part, speaker=speaker, sample_rate=sample_rate)
            audio_chunks.append(audio)

        if not audio_chunks:
            print("No text to synthesize.")
            return

        full_audio = torch.cat(audio_chunks).cpu()
        sf.write(output_audio_path, full_audio, sample_rate)
        print(f"Audio successfully saved to: {output_audio_path}")
    except Exception as e:
        print(f"\nCritical error during speech synthesis: {e}")
        traceback.print_exc()

def save_translated_text(translated_text, output_path):
    """Saves translated text to a file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(translated_text)
        print(f"Translation successfully saved to: {output_path}")
    except Exception as e:
        print(f"Error saving file: {e}")

def add_intonation_marks(text):
    """Adds intonation markers using spaCy for better speech synthesis."""
    doc = nlp(text)
    marked_text = ""
    for token in doc:
        marked_text += token.text
        if token.text in [',', '.', '!', '?']:
            marked_text += " "  # Add space for pause
    return marked_text

if __name__ == "__main__":
    # --- SETTINGS ---
    input_pdf_path_folder = r'D:\Project\ISP\CountNN\literatur\orig'
    input_pdf_path = input_pdf_path_folder + r"\2408.05446v1.pdf"
    translation_model = "facebook/nllb-200-distilled-600M"
    source_language = "eng_Latn"
    target_language = "rus_Cyrl"
    output_txt_path_folder_txt = r"D:\Project\ISP\CountNN\literatur\translate"
    output_txt_path = output_txt_path_folder_txt + r"\2408.05446v1.txt"
    output_txt_path_folder_audio = r"D:\Project\ISP\CountNN\literatur\audio"
    output_audio_path = output_txt_path_folder_audio + r"\2408.05446v1.wav"
    silero_speaker = 'aidar'  # aidar  xenia

    # --- EXECUTION ---
    if not os.path.exists(input_pdf_path):
        print(f"Error: File '{input_pdf_path}' not found.")
    else:
        print("1. Extracting text from PDF...")
        original_text = extract_text_from_pdf(pdf_path=input_pdf_path)

        if "Error" in original_text or not original_text.strip():
            print(original_text or "Error: PDF file is empty or lacks a text layer.")
        else:
            print("Text successfully extracted.")

            # Check if translation file already exists
            if os.path.exists(output_txt_path):
                print("Translation file already exists. Using it for synthesis.")
                with open(output_txt_path, 'r', encoding='utf-8') as f:
                    translated_content = f.read()
            else:
                print("\n2. Starting translation with NLLB...")

                # Split text into paragraphs to preserve structure
                paragraphs = original_text.split('\n\n')
                translated_paragraphs = []
                for paragraph in paragraphs:
                    translated_paragraph = translate_text_nllb(
                        text_to_translate=paragraph,
                        model_name=translation_model,
                        src_lang_code=source_language,
                        tgt_lang_code=target_language
                    )
                    if translated_paragraph:
                        translated_paragraphs.append(translated_paragraph)
                    else:
                        print("Error translating paragraph. Skipping.")
                translated_content = '\n\n'.join(translated_paragraphs)

                if translated_content:
                    print("Translation completed.")
                    save_translated_text(translated_content, output_txt_path)
                else:
                    print("Translation failed. Synthesis will not proceed.")
                    exit()

            # Add intonation markers
            marked_text = add_intonation_marks(translated_content)

            # Start speech synthesis
            synthesize_speech_silero(
                text_to_synthesize=marked_text,
                output_audio_path=output_audio_path,
                speaker=silero_speaker
            )