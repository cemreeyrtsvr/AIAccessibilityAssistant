"""Native SAPI TextToSpeechService test script.

Verifies consecutive speech requests without sound drops or pyttsx3 deadlocks.
"""

import time

from speech.tts import TextToSpeechService

if __name__ == "__main__":
    print("Testing native SAPI TextToSpeechService...")
    tts = TextToSpeechService()

    tts.speak("Birinci cümle test ediliyor.")
    time.sleep(2)

    tts.speak("İkinci cümle başarıyla seslendiriliyor.")
    time.sleep(2)

    tts.speak("Üçüncü cümle kesintisiz çalışıyor.")
    time.sleep(5)

    tts.stop()
    print("TTS Test Finished.")