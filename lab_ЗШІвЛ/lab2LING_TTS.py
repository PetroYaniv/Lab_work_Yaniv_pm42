import os
import random
from gtts import gTTS
from pydub import AudioSegment

# =============================
# СЛОВА
# =============================

nato_letters = [
    "Alpha","Bravo","Charlie","Delta","Echo","Foxtrot","Golf","Hotel",
    "India","Juliett","Kilo","Lima","Mike","November","Oscar","Papa",
    "Quebec","Romeo","Sierra","Tango","Uniform","Victor","Whiskey",
    "Xray","Yankee","Zulu"
]

digits = ["Zero","One","Two","Three","Four","Five","Six","Seven","Eight","Nine"]

words = nato_letters + digits
variations = 5
dataset_path = "dataset"
os.makedirs(dataset_path, exist_ok=True)

total_files = 0

for word in words:
    word_path = os.path.join(dataset_path, word)
    os.makedirs(word_path, exist_ok=True)

    for i in range(variations):
        try:
            # Створюємо аудіо
            tts = gTTS(text=word, lang='en')
            temp_mp3 = os.path.join(word_path, f"{word}_{i}.mp3")
            tts.save(temp_mp3)

            # Конвертація в WAV через pydub
            sound = AudioSegment.from_mp3(temp_mp3)
            wav_file = os.path.join(word_path, f"{word}_{i}.wav")
            sound.export(wav_file, format="wav")

            os.remove(temp_mp3)  # видаляємо тимчасовий MP3

            total_files += 1
            print(f"✓ Saved: {wav_file}")

        except Exception as e:
            print(f"❌ Error with {word}_{i}: {e}")

print("\n==============================")
print(f"✅ Dataset generated successfully!")
print(f"Total files created: {total_files}")
print("==============================")