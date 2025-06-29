# voice_output.py

import pyttsx3

def initialize_voice_engine():
    engine = pyttsx3.init()
    return engine

def speak_message(engine, message):
    engine.say(message)
    engine.runAndWait()
 
