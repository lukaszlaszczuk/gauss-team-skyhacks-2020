import logging
import streamlit as st
import speech_recognition as sr

from pydub import AudioSegment
from spacy import load
from spacy import displacy
from wordcloud import WordCloud

from utils.configuration import STOPWORDS
from utils.functions import (create_spacy_doc,
                             create_wordcloud_plot,
                             get_audio_duration,
                             get_categories_from_text,
                             audio_to_text)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def main():
    st.title('Silesia Video/Audio')

    modes = ['Audio', 'Video (currently not supported)']

    choice = st.sidebar.selectbox("Select Mode", modes)

    if choice == modes[1]:
        st.subheader('Video detection')
        data = st.file_uploader("Upload Video file")

    elif choice == modes[0]:
        st.subheader('Audio detection')
        data = st.file_uploader("Upload Audio file")
        selectbox = st.sidebar.selectbox(
            "What is your audio file type?",
            ("mp3 (does not display sound)", "wav (displays sound)")
        )

        if data:
            audio_bytes = data.read()

            if selectbox.startswith("mp3"):
                with open("data/audio-save.mp3", "wb") as f:
                    f.write(audio_bytes)
                audio_file = AudioSegment.from_mp3("data/audio-save.mp3")
                audio_file.export('data/audio-save.wav', format='wav')

            else:
                st.audio(data, format='audio/wav')
                with open("data/audio-save.wav", "wb") as f:
                    f.write(audio_bytes)

            # speech to text recognition
            r = sr.Recognizer()
            audio_file = sr.AudioFile('data/audio-save.wav')
            duration_max = get_audio_duration(audio_file, r)
            text_all, dict_txt = audio_to_text(audio_file, r, duration_max, 5)

            # NLP on text
            logger.debug('Loading model')
            nlp = load("pl_core_news_sm", disable=["tagger", "parser"])

            logger.debug('Creating spacy ners')
            spacy_doc = create_spacy_doc(text_all, nlp)
            st.markdown(displacy.render(spacy_doc, style='ent'), unsafe_allow_html=True)

            col1, col2 = st.beta_columns((5, 2))  # page layout

            logger.debug('Creating wordcloud')
            wc = WordCloud(stopwords=STOPWORDS, width=1000, height=500).generate_from_text(text_all)
            fig = create_wordcloud_plot(wc, (10, 5))
            col1.header('Generated WordCloud:')
            col1.write(fig)

            df = get_categories_from_text(dict_txt, nlp)
            col2.header('Detected categories in time:')
            col2.write(df)


if __name__ == '__main__':
    main()
