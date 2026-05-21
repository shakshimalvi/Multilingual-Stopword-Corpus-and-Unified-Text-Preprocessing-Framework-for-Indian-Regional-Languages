import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download(["punkt", "stopwords", "wordnet"], quiet=True)

lemmatizer = WordNetLemmatizer()

# English stopwords
EN_STOPWORDS = set(stopwords.words("english"))

# Combined custom stopwords (Hindi + Tamil + Kannada)
CUSTOM_STOPWORDS = {
    # Hindi
    "है","और","का","के","को","में","से","पर","यह","वह","भी","तक",
    "क्यों","क्या","जब","तब","तो","ही","था","थी","थे","बहुत"
# Additional Hindi stopwords
"मैं","तुम","आप","हम","वे","ये","इस","उस","इन","उन",
"मेरा","मेरी","मेरे","तेरा","तेरी","तेरे","आपका","आपकी","आपके",
"हमारा","हमारी","हमारे","उनका","उनकी","उनके",
"एक","दो","तीन","कुछ","कई","सभी","कोई","किसी","हर",
"अगर","लेकिन","मगर","या","अथवा","किंतु","परंतु","फिर","बाद",
"पहले","अब","तभी","अभी","यहाँ","वहाँ","कहाँ","कैसे","कब",
"कौन","कौनसा","किसका","किसकी","किसके","जिस","जिसका",
"जिसकी","जिसके","जो","सो","ताकि","इसीलिए","इसलिए",
"द्वारा","लिए","साथ","बिना","ऊपर","नीचे","अंदर","बाहर",
"पास","दूर","ज्यादा","कम","लगभग","सिर्फ","केवल","जरूर",
"काफी","बहुत","थोड़ा","थोड़ी","थोड़े",
"हो","हूँ","हैं","था","थी","थे","रहा","रही","रहे",
"होना","होते","कर","करना","किया","किए","किए गए","करते",
"करती","करेंगे","करोगे","गया","गई","गए","जाना","आना",
"देना","लेना","पाना","रखना","बनाना","चलना","देखना",
"सकता","सकती","सकते","चाहिए","चाहता","चाहती","चाहते",
"नहीं","मत","ना","हाँ","जी","अच्छा","ठीक","ओर","तरफ",
"वाला","वाली","वाले","वालीं","वगैरह","आदि"

    # Tamil
"இது","அது","மற்றும்","உள்ள","ஒரு","என்று","என","ஆக",
"இல்லை","நான்","நீ","அவன்","அவள்","அவர்கள்","நாம்","நீங்கள்",
"எங்கள்","உங்கள்","என்","உன்","இந்த","அந்த","இங்கே","அங்கே",
"எப்படி","எப்போது","எதற்கு","ஏன்","எது","யார்","யாரும்",
"ஆனால்","அல்லது","மேலும்","பின்னர்","முன்பு","பிறகு",
"மட்டும்","கூட","மிகவும்","பல","சில","எல்லாம்","ஒன்று",
"இருக்கும்","இருந்து","இருந்த","உண்டு","வேண்டும்","செய்ய",
"செய்த","செய்யும்","போல்","பற்றி","மூலம்","வரை","பின்",
"எனவே","ஆகவே","கொண்ட","கொண்டு","வந்து","சென்று",
"போது","அப்போது","இப்போது","இன்னும்","அதே","தான்",
"இவன்","இவள்","அவை","இவை","இதன்","அதன்","என்ன",
"என்னை","உன்னை","அவரை","அவர்களின்","எங்களின்",
"உங்களின்","தமது","தன்னுடைய"
}

ALL_STOPWORDS = EN_STOPWORDS.union(CUSTOM_STOPWORDS)


def is_pure_english(word):
    """Check if a word is purely English"""
    return bool(re.fullmatch(r"[a-zA-Z]+", word))


def data_prepare(text, custom_stopwords=None, remove_numbers=True):
    
    if pd.isna(text):
        return ""

    text = text.lower()

    # Choose regex pattern
    if remove_numbers:
        pattern = r"[^\u0900-\u097Fa-zA-Z\u0B80-\u0BFF\u0C80-\u0CFF\s]"
    else:
        pattern = r"[^\u0900-\u097Fa-zA-Z0-9\u0B80-\u0BFF\u0C80-\u0CFF\s]"

    text = re.sub(pattern, "", text)

    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()

    #  Merge stopwords
    final_stopwords = ALL_STOPWORDS.copy()
    if custom_stopwords:
        final_stopwords = final_stopwords.union(set(custom_stopwords))

    clean_tokens = []

    for word in tokens:

        if word in final_stopwords:
            continue

        if is_pure_english(word):
            word = lemmatizer.lemmatize(word, pos="v")

        clean_tokens.append(word)

    return " ".join(clean_tokens)