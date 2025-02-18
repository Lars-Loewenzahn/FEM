import requests
from bs4 import BeautifulSoup

# URL der Webseite
url = 'https://sdn.3qsdn.com/de/file/filesubtitle/11056943/'

# Webseite laden
response = requests.get(url)

# Überprüfen, ob die Anfrage erfolgreich war
if response.status_code == 200:
    # HTML-Inhalt der Seite parsen
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Alle Links auf der Seite finden
    links = soup.find_all('a')
    
    # Links zu SRT und TXT-Dateien filtern
    srt_link = None
    txt_link = None
    for link in links:
        href = link.get('href')
        if href.endswith('.srt'):
            srt_link = href
        elif href.endswith('.txt'):
            txt_link = href
    
    # Ergebnisse ausgeben
    print('SRT Link:', srt_link)
    print('TXT Link:', txt_link)
else:
    print('Fehler beim Laden der Webseite:', response.status_code)
