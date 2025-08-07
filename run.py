import fitz  # PyMuPDF
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os
import torch


def extract_text_from_pdf(pdf_path):
    """
    Извлекает текст из PDF-файла.
    """
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
    """
    Переводит текст с использованием многоязычной модели NLLB.
    (Версия, совместимая с разными версиями библиотеки transformers)
    """
    try:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Используемое устройство: {device}")

        tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=src_lang_code)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

        chunk_size = 350
        chunks = [text_to_translate[i:i + chunk_size] for i in range(0, len(text_to_translate), chunk_size)]

        translated_chunks = []
        for i, chunk in enumerate(chunks):
            print(f"Перевод части {i + 1}/{len(chunks)}...")

            inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

            # !!! ИЗМЕНЕНИЕ ЗДЕСЬ !!!
            # Используем метод .convert_tokens_to_ids() для обратной совместимости
            target_lang_id = tokenizer.convert_tokens_to_ids(tgt_lang_code)

            translated_ids = model.generate(
                **inputs,
                forced_bos_token_id=target_lang_id,
                max_length=512
            )

            translated_text = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
            translated_chunks.append(translated_text)

        return " ".join(translated_chunks)

    except Exception as e:
        return f"Ошибка при переводе: {e}"


def save_translated_text(translated_text, output_path):
    """
    Сохраняет переведенный текст в файл.
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(translated_text)
        print(f"Перевод успешно сохранен в: {output_path}")
    except Exception as e:
        print(f"Ошибка при сохранении файла: {e}")


if __name__ == "__main__":
    # --- НАСТРОЙКИ ---

    # 1. Укажите путь к вашему PDF-файлу
    input_pdf_path_folder=r'D:\Project\ISP\CountNN\literatur\orig'
    input_pdf_path = input_pdf_path_folder+r"\2408.05446v1.pdf"  # <-- ИЗМЕНИТЬ ЗДЕСЬ

    # 2. Укажите модель для перевода
    #    Используем NLLB. При первом запуске будет скачано ~2.4 ГБ.
    translation_model = "facebook/nllb-200-distilled-600M"  # <-- МОДЕЛЬ NLLB

    # 3. Укажите языки, используя официальные коды NLLB
    #    Полный список кодов: https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200
    source_language = "eng_Latn"  # <-- ИСХОДНЫЙ ЯЗЫК (английский)
    target_language = "rus_Cyrl"  # <-- ЦЕЛЕВОЙ ЯЗЫК (русский)

    # 3. Укажите имя файла для сохранения перевода
    output_txt_path_folder = r"D:\Project\ISP\CountNN\literatur\translate"  # <-- ИЗМЕНИТЬ ЗДЕСЬ
    output_txt_path = output_txt_path_folder+r"\d2408.05446v1.txt"  # <-- ИЗМЕНИТЬ ЗДЕСЬ

    # --- ВЫПОЛНЕНИЕ ---

    if not os.path.exists(input_pdf_path):
        print(f"Ошибка: Файл '{input_pdf_path}' не найден.")
    else:
        print("1. Извлечение текста из PDF...")
        original_text = extract_text_from_pdf(input_pdf_path)

        if "Ошибка" in original_text:
            print(original_text)
        elif not original_text.strip():
            print("Ошибка: PDF-файл пуст или не содержит текстового слоя.")
        else:
            print("Текст успешно извлечен.")
            print(
                "\n2. Начало процесса перевода с NLLB (это может занять много времени и потребует значительных ресурсов)...")

            translated_content = translate_text_nllb(original_text, translation_model, source_language, target_language)

            if "Ошибка" in translated_content:
                print(translated_content)
            else:
                print("Перевод завершен.")
                print("\n3. Сохранение переведенного текста...")
                save_translated_text(translated_content, output_txt_path)