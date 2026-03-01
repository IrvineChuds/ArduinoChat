import time
import requests
from arduino.app_utils import App

response = requests.get("https://www.youtube.com")
print(response.text)
print("Hello world!")


def loop():
    """This function is called repeatedly by the App framework."""
    # You can replace this with any code you want your App to run repeatedly.
    
        


# See: https://docs.arduino.cc/software/app-lab/tutorials/getting-started/#app-run
App.run(user_loop=loop)
