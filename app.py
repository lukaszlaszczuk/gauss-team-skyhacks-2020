import numpy as np
import pandas as pd
import streamlit as st
import speech_recognition as sr
import matplotlib.pyplot as plt
from wordcloud import WordCloud

import spacy
from spacy import displacy
import glob
import os
import json
import time
from pydub import AudioSegment

translation = ['park rozrywki',
               'zwierzęta',
               'ławka',
               'budynek',
               'zamek',
                'jaskinia',
              'kościół',
              'miasto',
              'krzyż',
              'kultura',  # kultura
              'jedzenie',
               'chodnik',
              'las',
              'meble',
              'trawa',
              'cmentarz',
              'jezioro',
              'kopalnia',
              'pomnik',
              'pojazd',  # pojazd
               'góry',
               'muzeum',
               'skansen',
               'park',
               'osoba',
               'rośliny',
               'rezerwuar',
               'rzeka',
               'droga',
               'skały',
               'śnieg',
               'sport',
               'obiekt sportowy',
               'schody',
               'drzewa',
               'statek',
               'okna'
              ]


STOPWORDS_ = ['ach', 'aj', 'albo', 'bardzo', 'bez', 'bo', 'być', 'ci', 'cię', 'ciebie', 'co', 'czy', 'daleko', 'dla',
              'dlaczego', 'dlatego', 'do', 'dobrze', 'dokąd', 'dość', 'dużo', 'dwa', 'dwaj', 'dwie', 'dwoje', 'dziś',
              'dzisiaj', 'gdyby', 'gdzie', 'go', 'ich', 'ile', 'im', 'inny', 'ja', 'ją', 'jak', 'jakby', 'jaki', 'jaka',
              'jako', 'je', 'tym',
              'jeden', 'jedna', 'jedno', 'jego', 'jej', 'jemu', 'jeśli', 'jest', 'jestem', 'jeżeli ', 'już', 'każdy',
              'kiedy', 'kierunku', 'kto', 'ku', 'lub', 'ma', 'mają', 'mam', 'mi', 'mną', 'mnie', 'moi', 'mój', 'moja',
              'moje', 'może', 'mu', 'my', 'na', 'nam', 'nami', 'nas', 'nasi', 'nasz', 'nasza', 'nasze', 'natychmiast',
              'nią', 'nic', 'nich', 'nie', 'niego', 'niej', 'niemu', 'nigdy', 'nim', 'nimi', 'niż', 'obok', 'od',
              'około', 'on', 'ona', 'one', 'oni', 'ono', 'owszem', 'po', 'pod', 'ponieważ', 'przed', 'przedtem',
              'przez', 'są',
              'sam', 'sama', 'się', 'skąd', 'tak', 'taki', 'tam', 'ten', 'to', 'tobą', 'tobie', 'tu', 'tutaj', 'twoi',
              'twój', 'twoja', 'twoje', 'ty', 'wam', 'wami', 'was', 'wasi', 'wasz', 'wasza', 'wasze', 'we', 'więc',
              'wszystko', 'wtedy', 'wy', 'żaden', 'zawsze', 'że']

def get_dict_txt():
    with open('../data/opisy.json') as f:
        dict_txt = json.load(f)

def create_spacy_doc(txt, nlp):
    doc = nlp(txt)
    return doc

def get_topics_in_text(dict_txt):
    nlp = spacy.load("pl_core_news_lg")
    df = pd.DataFrame()
    for i in range(len(list(dict_txt.keys()))):
        for key, value in dict_txt[list(dict_txt.keys())[i]].items():
            doc = nlp(value)
            list_out = [(t.orth_, t.lemma_, key) for t in doc]
            df = df.append(pd.DataFrame(list_out, columns=['forma','lemat', 'timestamp']))
    df['is_in_topics'] = df['lemat'].apply(lambda x: x in translation)
    return df.reset_index(drop=True)


def main():
    st.title('Silesia Video/Audio')

    modes = ['Video', 'Audio']

    choice = st.sidebar.selectbox("Select Mode", modes)

    if choice == 'Video':
        st.subheader('Video detection')
        data = st.file_uploader("Upload Video file")

    elif choice == 'Audio':
        st.subheader('Audio detection')
        data = st.file_uploader("Upload Audio file")
        if data:
            audio_bytes = data.read()
            st.audio(audio_bytes, format='audio/wav')
            time.sleep(3.14)
            r = sr.Recognizer()
            with open("audio-save.wav", "wb") as f:
                f.write(audio_bytes)

            audio_file = sr.AudioFile('audio-save.wav')
            with audio_file as source:
                audio = r.record(source)
            duration_max = audio_file.FRAME_COUNT / audio_file.SAMPLE_RATE
            text_all = ""
            for start in np.arange(0, duration_max, 5):
                with audio_file as source:
                    audio = r.record(source, offset=start, duration=5)
                translation = r.recognize_google(audio, language='pl')
                print(translation)
                text_all += f' {f"[{int(start//60)}:{str(0) if start%60 < 10 else str()}{int(start%60)}]"} {translation}'
                print(text_all)
            nlp = spacy.load("pl_core_news_lg")
            spacy_doc = create_spacy_doc(text_all, nlp)
            st.markdown(displacy.render(spacy_doc, style='ent'), unsafe_allow_html=True)

            wc = WordCloud(stopwords=STOPWORDS_).generate_from_text(text_all)
            fig, ax = plt.subplots(figsize=(10, 5))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            st.write(fig)


if __name__ == '__main__':
    main()
