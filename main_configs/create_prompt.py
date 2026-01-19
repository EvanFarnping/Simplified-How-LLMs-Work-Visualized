from bs4 import BeautifulSoup
from pathlib import Path

import requests
import re

######################################## | TODO | ########################################
URL_TO_SCRAPE = "https://en.wikipedia.org/wiki/Tropical_Storm_Brenda_(1960)" # TODO EDIT ME! Find a LINK!
######################################## | TODO | ########################################

CURRENT_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = CURRENT_DIR / "prompts" / "generated_from_url"
PROMPTS_DIR.mkdir(parents=True, exist_ok=True)

def clean_filename(url):
    clean = re.sub(r'^https?://(www\.)?', '', url)
    clean = re.sub(r'[^a-zA-Z0-9]', '_', clean)
    return clean[:50] + ".txt"

def fetch_and_save_url(url):
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        for script in soup(["script", "style", "nav", "footer"]):
            script.extract()
            
        text = soup.get_text(separator=' ', strip=True)
        
        final_content = f"Very concisely summarize the information accurately:\n\n{text}"
        
        filename = clean_filename(url)
        output_path = PROMPTS_DIR / filename
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_content)
            
        print(f"File saved to:\n   {output_path}")
        return output_path

    except Exception as e:
        print(f"Error fetching URL: {e}")
        return None

if __name__ == "__main__":
    if URL_TO_SCRAPE and URL_TO_SCRAPE != "PASTE_LINK_HERE":
        fetch_and_save_url(URL_TO_SCRAPE)
    else:
        print("Please paste a valid link into the URL_TO_SCRAPE variable at the top of the file.")