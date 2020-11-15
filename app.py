import pandas as pd
import streamlit as st
import speech_recognition as sr
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from pydub import AudioSegment

from spacy import load
from spacy import displacy

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
              'jako', 'je', 'tym', 'gdy', 'nad', 'ze',
              'jeden', 'jedna', 'jedno', 'jego', 'jej', 'jemu', 'jeśli', 'jest', 'jestem', 'jeżeli ', 'już', 'każdy',
              'kiedy', 'kierunku', 'kto', 'który', 'ku', 'lub', 'ma', 'mają', 'mam', 'mi', 'mną', 'mnie', 'moi', 'mój',
              'moja',
              'moje', 'może', 'mu', 'my', 'na', 'nam', 'nami', 'nas', 'nasi', 'nasz', 'nasza', 'nasze', 'natychmiast',
              'nią', 'nic', 'nich', 'nie', 'niego', 'niej', 'niemu', 'nigdy', 'nim', 'nimi', 'niż', 'obok', 'od',
              'około', 'on', 'ona', 'one', 'oni', 'ono', 'owszem', 'po', 'pod', 'ponieważ', 'przed', 'przedtem',
              'przez', 'są',
              'sam', 'sama', 'się', 'skąd', 'tak', 'taki', 'tam', 'ten', 'to', 'tobą', 'tobie', 'tu', 'tutaj', 'twoi',
              'twój', 'twoja', 'twoje', 'ty', 'wam', 'wami', 'was', 'wasi', 'wasz', 'wasza', 'wasze', 'we', 'więc',
              'wszystko', 'wtedy', 'wy', 'żaden', 'zawsze', 'że']


def create_spacy_doc(txt, nlp):
    doc = nlp(txt)
    return doc


def get_topics_in_text(dict_txt, nlp):
    df = pd.DataFrame()
    for key, value in dict_txt.items():
        doc = nlp(value)
        list_out = [(t.orth_, t.lemma_, key) for t in doc]
        df = df.append(pd.DataFrame(list_out, columns=['forma', 'Temat', 'Time (min:sec)']))
    df['is_in_topics'] = df['Temat'].apply(lambda x: x in translation)
    return df[df['is_in_topics']].loc[:, ['Temat', 'Time (min:sec)']].sort_values('Time (min:sec)').reset_index(drop=True)


def main():
    st.title('Silesia Video/Audio')

    modes = ['Audio', 'Video (currently not supported)']

    choice = st.sidebar.selectbox("Select Mode", modes)

    if choice == 'Video':
        st.subheader('Video detection')
        data = st.file_uploader("Upload Video file")

    elif choice == 'Audio':
        st.subheader('Audio detection')
        data = st.file_uploader("Upload Audio file")
        selectbox = st.sidebar.selectbox(
            "What is your audio file type?",
            ("mp3 (does not display sound)", "wav (displays sound)")
        )

        if data:
            audio_bytes = data.read()

            if selectbox.startswith("mp3"):
                with open("audio-save.mp3", "wb") as f:
                    f.write(audio_bytes)
                audio_file = AudioSegment.from_mp3("audio-save.mp3")
                audio_file.export('audio-save.wav', format='wav')

            else:
                st.audio(data, format='audio/wav')
                with open("audio-save.wav", "wb") as f:
                    f.write(audio_bytes)

            r = sr.Recognizer()
            audio_file = sr.AudioFile('audio-save.wav')

            with audio_file as source:
                audio = r.record(source)
            duration_max = audio_file.FRAME_COUNT / audio_file.SAMPLE_RATE
            text_all = ""
            dict_txt = dict()
            for start in range(0, int(duration_max), 5):
                with audio_file as source:
                    audio = r.record(source, offset=start, duration=5)
                translation = r.recognize_google(audio, language='pl')
                dict_txt[f"{int(start // 60)}:{str(0) if start % 60 < 10 else str()}{int(start % 60)}"] = translation
                print(translation)
                text_all += f' {f"[{int(start // 60)}:{str(0) if start % 60 < 10 else str()}{int(start % 60)}]"} {translation}'
            del audio
            del r
            del audio_bytes
            del audio_file
            print('Loading model')
            nlp = load("pl_core_news_sm", disable=["tagger", "parser"])
            print('creating spacy ners')
            spacy_doc = create_spacy_doc(text_all, nlp)

            st.markdown(displacy.render(spacy_doc, style='ent'), unsafe_allow_html=True)

            df = get_topics_in_text(dict_txt, nlp)
            del nlp
            del spacy_doc

            col1, col2 = st.beta_columns((5, 2))
            print('creating wordcloud')
            wc = WordCloud(stopwords=STOPWORDS_, width=1000, height=500).generate_from_text(text_all)
            print('plotting wordcloud')
            fig, ax = plt.subplots(figsize=(10, 5))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')

            # st.write(fig)
            col1.header('Generated WordCloud:')
            col1.write(fig)

            col2.header('Detected categories in time:')
            col2.write(df)


if __name__ == '__main__':
    main()
