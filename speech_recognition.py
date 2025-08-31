import speech_recognition as sr
import pyttsx3
import datetime
import wikipedia
import pywhatkit
import sys
import os

# Initialize text-to-speech engine
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)  # male voice, change index for female
engine.setProperty('rate', 170)  # words per minute

def speak(text):
    print("Assistant:", text)
    engine.say(text)
    engine.runAndWait()

def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.pause_threshold = 1
        audio = r.listen(source)
    try:
        query = r.recognize_google(audio, language='en-in')
        print(f"You said: {query}")
        return query.lower()
    except sr.UnknownValueError:
        speak("Sorry, I didn’t catch that.")
        return ""
    except sr.RequestError:
        speak("Network issue.")
        return ""

def process_command(query):
    if "time" in query:
        time = datetime.datetime.now().strftime("%I:%M %p")
        speak(f"The time is {time}")

    elif "search" in query:
        topic = query.replace("search", "").strip()
        if topic:
            results = wikipedia.summary(topic, sentences=2)
            speak(results)
        else:
            speak("What do you want me to search?")

    elif "play" in query:
        song = query.replace("play", "").strip()
        if song:
            speak(f"Playing {song} on YouTube")
            pywhatkit.playonyt(song)
        else:
            speak("Which song should I play?")

    elif "open notepad" in query:
        os.system("notepad.exe")
        speak("Opening Notepad")

    elif "exit" in query or "quit" in query:
        speak("Goodbye!")
        sys.exit()

    else:
        speak("I don’t know that command yet.")

def main():
    speak("Hello Sam, I am your voice assistant.")
    while True:
        query = listen()
        if query:
            process_command(query)

if __name__ == "__main__":
    main()
