from urllib.request import urlopen
import time
from bs4 import BeautifulSoup
import re

class PageGrabber:
    def __init__(self, delay=0.5):
        self.last_pull_time = time.time()
        self.delay = delay

    def request_page(self, url):
        """Request the page from wikipedia"""
        while True:
            if time.time() - self.delay > self.last_pull_time:
                raw_html = urlopen(url).read()
                self.last_pull_time = time.time()
                break
        return raw_html

    def pull(self, url):
        """Pull the prunciation off of wiktionary"""
        raw_html = self.request_page(url)

        # Header document parse
        soup = BeautifulSoup(raw_html, 'html.parser')
        english_header = soup.find(id="English")
        siblings = english_header.parent.next_siblings
        pro_id = -3
        pro_ul = None

        for i, s in enumerate(siblings):
            if i == (pro_id + 2):
                # We've found the pronunciatian list.
                pro_ul = BeautifulSoup(str(s), 'html.parser')
            sib = BeautifulSoup(str(s), 'html.parser')
            if sib.find(id="Pronunciation"):
                pro_id = i

        ipa_elem_string = None
        american_match = re.compile(">american|>us|>general american")

        if pro_ul is not None:
            contains_american = False
            for list_elem in pro_ul.contents[0]:
                if american_match.search(str(list_elem).lower()):
                    ipa_elem_string = str(
                            BeautifulSoup(str(list_elem),
                                "html.parser").find(class_="IPA"))
                    contains_american = True
            if not contains_american:
                ipa_elem_string = str(pro_ul.find(class_="IPA"))
        else:
            # Could not find pronunciation header.
            print("Could not find pronunciation header")
            return None

        # Take out the IPA between the slashes.
        slash_indices = []
        for i, c in enumerate(ipa_elem_string):
            if c == "/" or c == "[" or c == "]":
                slash_indices.append(i)
        ipa = ipa_elem_string[slash_indices[0]+1:slash_indices[1]]

        # Return the ipa cleaned.
        return ipa


def get_wiktionary_page(word):
    url = "https://en.wiktionary.org/wiki/"+word
    return url

pg = PageGrabber()
print(pg.pull(get_wiktionary_page(str(input("word> ")))))
