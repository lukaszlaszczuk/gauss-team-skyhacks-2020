import logging
import matplotlib.pyplot as plt
import pandas as pd

from utils.configuration import CATEGORY_TRANSLATION

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def create_spacy_doc(txt, nlp):
    doc = nlp(txt)
    return doc


def get_audio_duration(audio_file, recognizer):
    with audio_file as source:
        audio = recognizer.record(source)
    duration_max = audio_file.FRAME_COUNT / audio_file.SAMPLE_RATE
    return duration_max


def get_categories_from_text(dict_txt, nlp):
    df = pd.DataFrame()
    for key, value in dict_txt.items():
        doc = nlp(value)
        list_out = [(t.orth_, t.lemma_, key) for t in doc]
        df = df.append(pd.DataFrame(list_out, columns=['forma', 'Temat', 'Time (min:sec)']))
    df['is_in_topics'] = df['Temat'].apply(lambda x: x in CATEGORY_TRANSLATION)
    return df[df['is_in_topics']].loc[:, ['Temat', 'Time (min:sec)']].sort_values('Time (min:sec)').reset_index(
        drop=True)


def audio_to_text(audio_file, recognizer, duration_max, frame_length):
    text_all = ""
    dict_txt = dict()
    for start in range(0, int(duration_max), frame_length):
        with audio_file as source:
            audio = recognizer.record(source, offset=start, duration=frame_length)
        translation = recognizer.recognize_google(audio, language='pl')
        dict_txt[f"{int(start // 60)}:{str(0) if start % 60 < 10 else str()}{int(start % 60)}"] = translation
        logger.debug(translation)
        text_all += f' {f"[{int(start // 60)}:{str(0) if start % 60 < 10 else str()}{int(start % 60)}]"} {translation}'

    return text_all, dict_txt


def create_wordcloud_plot(wordcloud, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    return fig
