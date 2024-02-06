import openai
import speech_recognition as sr
import pyttsx3
import pygame
from PyQt5.QtWidgets import QApplication, QLabel, QLineEdit, QVBoxLayout, QWidget

# set up OpenAI API client
openai.api_key = "YOUR_API_KEY"

# set up text-to-speech engine
engine = pyttsx3.init()

# set up speech recognition engine
r = sr.Recognizer()
mic = sr.Microphone()

# initialize pygame
pygame.init()

# set up display
screen = pygame.display.set_mode((500, 500))
pygame.display.set_caption("Voice Assistant")
avatar_image = pygame.image.load("avatar.png")
avatar_rect = avatar_image.get_rect(center=(250, 250))

# initialize memory
memory = []

# define function to send user input to ChatGPT and return model's response
def get_response(user_input):
    # prepend memory to user input
    input_with_memory = ' '.join(memory + [user_input])
    response = openai.Completion.create(
        engine="davinci", prompt=input_with_memory, max_tokens=1024, n=1, stop=None, temperature=0.7
    )
    return response.choices[0].text.strip()

# define function to convert text to speech
def speak(text):
    engine.say(text)
    engine.runAndWait()

# main loop to listen for user input and respond
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    screen.fill((255, 255, 255))
    screen.blit(avatar_image, avatar_rect)
    pygame.display.flip()

    with mic as source:
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
    try:
        user_input = r.recognize_google(audio)
        print("You said:", user_input)
        response = get_response(user_input)
        print("ChatGPT:", response)
        speak(response)

        # append user input and response to memory
        memory.append(user_input)
        memory.append(response)
    except sr.UnknownValueError:
        print("Sorry, I didn't catch that.")
