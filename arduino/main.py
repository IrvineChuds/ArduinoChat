import requests
from arduino.app_utils import App

response = requests.get("https://www.youtube.com")
print(response.text)
print("Hello world!")


def loop():
    """This function is called repeatedly by the App framework."""
    # You can replace this with any code you want your App to run repeatedly.

    # step 1: recognize face

    # step 2: get user input by implementing speech to text
    # perhaps have an activation keyword

    # QOL feature, make it so that you when talk, the AI stops speaking

    # step 3: send user string to the API with a POST request after formatting the input
    # to  


# See: https://docs.arduino.cc/software/app-lab/tutorials/getting-started/#app-run
App.run(user_loop=loop)
