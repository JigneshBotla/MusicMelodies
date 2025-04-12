# import streamlit as st

# import pandas as pd
# import numpy as np
# import altair as alt

# import joblib

# pipe_lr = joblib.load(open("model/text_emotion.pkl", "rb"))

# emotions_emoji_dict = {"anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂", "neutral": "😐", "sad": "😔",
#                        "sadness": "😔", "shame": "😳", "surprise": "😮"}


# def predict_emotions(docx):
#     results = pipe_lr.predict([docx])
#     return results[0]


# def get_prediction_proba(docx):
#     results = pipe_lr.predict_proba([docx])
#     return results


# def main():
#     st.title("Mood Melodies -- Text to Emotion Detection")
#     st.subheader("Detect Emotions In Text")

#     with st.form(key='my_form'):
#         raw_text = st.text_area("Type Here")
#         submit_text = st.form_submit_button(label='Submit')

#     if submit_text:
#         col1, col2 = st.columns(2)

#         prediction = predict_emotions(raw_text)
#         probability = get_prediction_proba(raw_text)

#         with col1:
#             st.success("Original Text")
#             st.write(raw_text)

#             st.success("Prediction")
#             emoji_icon = emotions_emoji_dict[prediction]
#             st.write("{}:{}".format(prediction, emoji_icon))
#             st.write("Confidence:{}".format(np.max(probability)))

#         with col2:
#             st.success("Prediction Probability")
#             #st.write(probability)
#             proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
#             #st.write(proba_df.T)
#             proba_df_clean = proba_df.T.reset_index()
#             proba_df_clean.columns = ["emotions", "probability"]

#             fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
#             st.altair_chart(fig, use_container_width=True)






# if __name__ == '__main__':
#     main()



import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib

# Load the trained pipeline
pipe_svm = joblib.load(open("model/text_emotion.pkl", "rb"))

# Emotion-to-emoji mapping
emotions_emoji_dict = {
    "anger": "😠",
    "disgust": "🤮",
    "fear": "😨😱",
    "happy": "🤗",
    "joy": "😂",
    "neutral": "😐",
    "sad": "😔",
    "sadness": "😔",
    "shame": "😳",
    "surprise": "😮",
}

# Function to predict the emotion
def predict_emotions(docx):
    results = pipe_svm.predict([docx])
    return results[0]

# Function to get prediction probabilities
def get_prediction_proba(docx):
    # Check if the pipeline supports predict_proba
    if hasattr(pipe_svm, "predict_proba"):
        results = pipe_svm.predict_proba([docx])
    else:
        # Use decision_function and convert to probabilities via softmax
        decision_scores = pipe_svm.decision_function([docx])
        results = np.exp(decision_scores) / np.sum(np.exp(decision_scores), axis=1, keepdims=True)
    return results

# Streamlit App
def main():
    st.title("Mood Melodies -- Text to Emotion Detection")
    st.subheader("Detect Emotions In Text")

    with st.form(key="my_form"):
        raw_text = st.text_area("Type Here")
        submit_text = st.form_submit_button(label="Submit")

    if submit_text:
        col1, col2 = st.columns(2)

        # Prediction and probabilities
        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)

        with col1:
            st.success("Original Text")
            st.write(raw_text)

            st.success("Prediction")
            emoji_icon = emotions_emoji_dict[prediction]
            st.write(f"{prediction}: {emoji_icon}")
            st.write(f"Confidence: {np.max(probability):.2f}")

        with col2:
            st.success("Prediction Probability")
            proba_df = pd.DataFrame(probability, columns=pipe_svm.classes_)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotions", "probability"]

            fig = alt.Chart(proba_df_clean).mark_bar().encode(
                x="emotions", y="probability", color="emotions"
            )
            st.altair_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
