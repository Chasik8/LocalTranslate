import fitz  # PyMuPDF
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os
import torch
import soundfile as sf
from tqdm import tqdm
import traceback  # <-- Импортируем модуль для детальной отладки


def extract_text_from_pdf(pdf_path):
    """Извлекает текст из PDF-файла."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        return f"Ошибка при обработке PDF: {e}"


def translate_text_nllb(text_to_translate, model_name, src_lang_code, tgt_lang_code):
    """Переводит текст с использованием многоязычной модели NLLB."""
    try:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Используемое устройство для перевода: {device}")

        tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=src_lang_code)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

        chunk_size = 350
        chunks = [text_to_translate[i:i + chunk_size] for i in range(0, len(text_to_translate), chunk_size)]

        translated_chunks = []
        print("Начинаю перевод текста...")
        for chunk in tqdm(chunks, desc="Перевод"):
            inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            target_lang_id = tokenizer.convert_tokens_to_ids(tgt_lang_code)
            translated_ids = model.generate(**inputs, forced_bos_token_id=target_lang_id, max_length=512)
            translated_text = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
            translated_chunks.append(translated_text)

        return " ".join(translated_chunks)

    except Exception as e:
        print(f"Ошибка при переводе: {e}")
        traceback.print_exc()
        return None


def synthesize_speech_silero(text_to_synthesize, output_audio_path, speaker='xenia', model_id='v4_ru'):
    """Синтезирует речь из текста с помощью моделей Silero и сохраняет в WAV файл."""
    try:
        print("\n3. Начало процесса синтеза речи (озвучивания)...")
        device = torch.device('cpu')
        torch.set_num_threads(4)

        # Добавляем trust_repo=True, чтобы избежать предупреждения и подтвердить доверие к репозиторию
        model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                  model='silero_tts',
                                  language='ru',
                                  speaker=model_id,
                                  trust_repo=True)  # <-- ИЗМЕНЕНИЕ ЗДЕСЬ
        model.to(device)

        sample_rate = 48000

        # !!! ИЗМЕНЕНИЕ ЗДЕСЬ: Надежная разбивка текста на чанки по длине !!!
        # Вместо split('.'), который может создавать слишком длинные фрагменты.
        text_parts = []
        max_len = 950  # Максимальная длина чанка в символах (эмпирически подобранный безопасный лимит)
        current_text = text_to_synthesize
        while len(current_text) > 0:
            if len(current_text) <= max_len:
                text_parts.append(current_text)
                break
            # Ищем последнюю точку или пробел, чтобы не резать слова
            split_at = current_text.rfind('.', 0, max_len)
            if split_at == -1:
                split_at = current_text.rfind(' ', 0, max_len)
            if split_at == -1:  # Если не нашли ни точки, ни пробела, режем как есть
                split_at = max_len

            text_parts.append(current_text[:split_at + 1])
            current_text = current_text[split_at + 1:].lstrip()

        audio_chunks = []
        for part in tqdm(text_parts, desc="Озвучивание фрагментов"):
            if not part.strip():
                continue
            audio = model.apply_tts(text=part,
                                    speaker=speaker,
                                    sample_rate=sample_rate)
            audio_chunks.append(audio)

        if not audio_chunks:
            print("Нет текста для озвучивания.")
            return

        full_audio = torch.cat(audio_chunks).cpu()
        sf.write(output_audio_path, full_audio, sample_rate)

        print(f"Озвучка успешно сохранена в: {output_audio_path}")

    except Exception as e:
        # !!! ИЗМЕНЕНИЕ ЗДЕСЬ: Улучшенный вывод ошибок !!!
        print(f"\nКритическая ошибка при синтезе речи: {e}")
        print("----------------------------------------------------")
        traceback.print_exc()
        print("----------------------------------------------------")


def save_translated_text(translated_text, output_path):
    """Сохраняет переведенный текст в файл."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(translated_text)
        print(f"Перевод успешно сохранен в: {output_path}")
    except Exception as e:
        print(f"Ошибка при сохранении файла: {e}")


if __name__ == "__main__":
    # --- НАСТРОЙКИ ---

    # 1. Путь к вашему PDF-файлу
    input_pdf_path_folder=r'D:\Project\ISP\CountNN\literatur\orig'
    input_pdf_path = input_pdf_path_folder+r"\2408.05446v1.pdf"  # <-- ИЗМЕНИТЬ ЗДЕСЬ

    # 2. Настройки перевода
    translation_model = "facebook/nllb-200-distilled-600M"
    source_language = "eng_Latn"
    target_language = "rus_Cyrl"
    output_txt_path_folder_txt = r"D:\Project\ISP\CountNN\literatur\translate"  # <-- ИЗМЕНИТЬ ЗДЕСЬ
    output_txt_path = output_txt_path_folder_txt+r"\2408.05446v1.txt"  # <-- ИЗМЕНИТЬ ЗДЕСЬ

    # 3. Настройки озвучивания
    output_txt_path_folder_audio = r"D:\Project\ISP\CountNN\literatur\audio"  # <-- ИЗМЕНИТЬ ЗДЕСЬ
    output_audio_path = output_txt_path_folder_audio+r"\2408.05446v1.wav"  # <-- ИЗМЕНИТЬ ЗДЕСЬТакже добавь проверку, что файл с переводом уже создан, чтобы он прочел его. Также хотелось бы, чтобы при переводе сохранялась структура текста. Еще возможно стоит добавить автоматические преобразования для расстовления интонации, если это поможет синтезатору.
    silero_speaker = 'xenia'

    # --- ВЫПОЛНЕНИЕ ---
    if not os.path.exists(input_pdf_path):
        print(f"Ошибка: Файл '{input_pdf_path}' не найден.")
    else:
        print("1. Извлечение текста из PDF...")
        original_text = extract_text_from_pdf(pdf_path=input_pdf_path)

        if "Ошибка" in original_text or not original_text.strip():
            print(original_text or "Ошибка: PDF-файл пуст или не содержит текстового слоя.")
        else:
            print("Текст успешно извлечен.")
            print("\n2. Начало процесса перевода с NLLB...")

            translated_content = translate_text_nllb(
                text_to_translate=original_text,
                model_name=translation_model,
                src_lang_code=source_language,
                tgt_lang_code=target_language
            )

            if translated_content:
                print("Перевод завершен.")
                save_translated_text(translated_content, output_txt_path)

                # Запуск озвучивания после успешного перевода
                synthesize_speech_silero(
                    text_to_synthesize=translated_content,
                    output_audio_path=output_audio_path,
                    speaker=silero_speaker
                )