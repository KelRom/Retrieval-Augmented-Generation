import requests
from bs4 import BeautifulSoup

def scrape_webpage():
    url = "https://en.wikipedia.org/wiki/Machine_learning"  # Hard-coded URL
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            content_div = soup.find('div', class_='mw-parser-output')
            if content_div:
                paragraphs = content_div.find_all('p')
                article_text = "\n\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
            else:
                article_text = ""
                print("Warning: <div class='mw-parser-output'> not found. No content extracted.")

            with open("Selected_Document.txt", "w", encoding="utf-8") as file:
                file.write(article_text)

            print("Success: Page fetched and content saved to Selected_Document.txt")
            return article_text
        else:
            print(f"Failure: HTTP status code {response.status_code}")
            return ""
    except Exception as e:
        print(f"An error occurred: {e}")
        return ""

def main():
    scrape_webpage()

if __name__ == '__main__':
    main()
