import speech_recognition as sr
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import pyttsx3

class PersonalAssistant:
    def __init__(self):
        # Initialize the tokenizer and model
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.model.eval()

        # Initialize the speech recognizer and text-to-speech engine
        self.recognizer = sr.Recognizer()
        self.engine = pyttsx3.init()

    def listen(self):
        with sr.Microphone() as source:
            print("Listening...")
            audio = self.recognizer.listen(source)

        try:
            # Recognize speech using Google Web Speech API
            text = self.recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
            return None

    def respond(self, text):
        input_ids = self.tokenizer.encode(text, return_tensors='pt')
        with torch.no_grad():
            output = self.model.generate(input_ids, max_length=150)
        response_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        print(response_text)
        self.engine.say(response_text)
        self.engine.runAndWait()
        return response_text

# Create an instance of the assistant
assistant = PersonalAssistant()

# Example of continuous listening and responding
while True:
    text = assistant.listen()
    if text:
        assistant.respond(text)
