# WEB API: https://sv443.net/jokeapi/v2/#try-it
import requests

def get_jokes(lenguage):
    api_jokes_url = "https://v2.jokeapi.dev/joke/Any?lang=" + lenguage

    response = requests.get(api_jokes_url)
    jokes_array = []
    try:
        if response.status_code == 200:
            joke_data = response.json()
            
            if joke_data['type'] == 'single':
                joke = joke_data['joke']
                jokes_array.append(joke)
                
            elif joke_data['type'] == 'twopart':
                setup = joke_data['setup']
                delivery = joke_data['delivery']
                jokes_array.append(setup + " " + delivery)
                
            return jokes_array
        else:
            print(f"Error: {response.status_code}")
            return None
        
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None
