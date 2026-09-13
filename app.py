import streamlit as st
from googleapiclient.discovery import build
from transformers import pipeline
import re
import pandas as pd


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="YouTube Sentiment Analyzer",
    page_icon="📊",
    layout="wide"
)

st.title("📊 YouTube Sentiment Analyzer")

st.write(
    "Analyze YouTube comments using NLP, "
    "Hugging Face Transformers and AI-generated viewer insights."
)


# -----------------------------
# YouTube API Key
# -----------------------------

API_KEY = st.secrets["YT_APIKEY"]


# -----------------------------
# Extract YouTube Video ID
# -----------------------------

def extract_video_id(url):

    patterns = [
        r"(?:youtube\.com/watch\?v=)([^&]+)",
        r"(?:youtu\.be/)([^?&]+)",
        r"(?:youtube\.com/shorts/)([^?&]+)"
    ]

    for pattern in patterns:

        match = re.search(pattern, url)

        if match:
            return match.group(1)

    return None


# -----------------------------
# Fetch YouTube Comments
# -----------------------------

def get_comments(video_id, api_key):

    youtube = build(
        "youtube",
        "v3",
        developerKey=api_key
    )

    comments = []

    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100,
        textFormat="plainText"
    )

    while request:

        response = request.execute()

        for item in response["items"]:

            comment = (
                item["snippet"]
                ["topLevelComment"]
                ["snippet"]
                ["textDisplay"]
            )

            comments.append(comment)

        request = youtube.commentThreads().list_next(
            request,
            response
        )

    return comments


# -----------------------------
# Load Sentiment Model
# -----------------------------

@st.cache_resource
def load_sentiment_model():

    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )


# -----------------------------
# Load AI Summarization Model
# -----------------------------

@st.cache_resource
def load_summary_model():

    return pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-0.5B-Instruct"
    )

# -----------------------------
# Analyze Sentiments
# -----------------------------

def analyze_sentiments(comments, model):

    results = []

    for comment in comments:

        result = model(
            comment[:512]
        )[0]

        results.append({
            "Comment": comment,
            "Sentiment": result["label"],
            "Confidence": result["score"]
        })

    return pd.DataFrame(results)


# -----------------------------
# Generate AI Viewer Insights
# -----------------------------

def generate_ai_summary(df, summary_model):

    positive_comments = df[
        df["Sentiment"] == "POSITIVE"
    ]["Comment"].tolist()

    negative_comments = df[
        df["Sentiment"] == "NEGATIVE"
    ]["Comment"].tolist()

    # Select representative comments
    positive_samples = positive_comments[:8]
    negative_samples = negative_comments[:8]

    comments_text = ""

    comments_text += "\nPositive comments:\n"

    for comment in positive_samples:
        comments_text += "- " + comment[:300] + "\n"

    comments_text += "\nNegative comments:\n"

    for comment in negative_samples:
        comments_text += "- " + comment[:300] + "\n"

    prompt = f"""
Analyze these YouTube comments as an AI sentiment analyst.

{comments_text}

Return ONLY the following:

- Overall tone: [Positive/Negative/Mixed]
- What viewers liked: [main points]
- Main concerns: [main points]
- Negative examples: [brief examples]
- Conclusion: [one short sentence]

Do not repeat the instructions.
Do not add personal opinions.
Use only information found in the comments.
"""

    result = summary_model(
       prompt,
       max_new_tokens=180,
       do_sample=False
    )[0]["generated_text"]

    if result.startswith(prompt):
        result = result[len(prompt):].strip()

    # Remove the original prompt from the generated result
    if result.startswith(prompt):
        result = result[len(prompt):]

    return  result.strip()

# -----------------------------
# Initialize Session State
# -----------------------------

if "analysis_df" not in st.session_state:

    st.session_state.analysis_df = None


# -----------------------------
# YouTube URL
# -----------------------------

video_url = st.text_input(
    "Enter YouTube Video URL"
)


# -----------------------------
# Analyze Button
# -----------------------------

if st.button("🚀 Analyze Video"):

    if not video_url:

        st.warning(
            "Please enter a YouTube video URL."
        )

    else:

        video_id = extract_video_id(video_url)

        if video_id is None:

            st.error(
                "Invalid YouTube URL."
            )

        else:

            st.success(
                f"Video ID detected: {video_id}"
            )

            try:

                # -----------------------------
                # Fetch Comments
                # -----------------------------

                with st.spinner(
                    "Fetching YouTube comments..."
                ):

                    comments = get_comments(
                        video_id,
                        API_KEY
                    )

                st.success(
                    "Comments fetched successfully!"
                )


                if comments:

                    # -----------------------------
                    # Load Sentiment Model
                    # -----------------------------

                    with st.spinner(
                        "Loading AI sentiment model..."
                    ):

                        sentiment_model = (
                            load_sentiment_model()
                        )


                    # -----------------------------
                    # Sentiment Analysis
                    # -----------------------------

                    with st.spinner(
                        "Analyzing comments with AI..."
                    ):

                        df = analyze_sentiments(
                            comments,
                            sentiment_model
                        )


                    st.session_state.analysis_df = df

                    st.success(
                        "Sentiment analysis completed!"
                    )


                    # -----------------------------
                    # AI Summary Model
                    # -----------------------------

                    with st.spinner(
                        "Loading AI summarization model..."
                    ):

                        summary_model = (
                            load_summary_model()
                        )


                    # -----------------------------
                    # Generate Viewer Insights
                    # -----------------------------

                    with st.spinner(
                        "Generating AI viewer insights..."
                    ):

                        insights = generate_ai_summary(
                            df,
                            summary_model
                        )


                    st.session_state.insights = insights


                else:

                    st.warning(
                        "No comments were found for this video."
                    )


            except Exception as e:

                st.error(
                    f"Error: {e}"
                )


# ==================================================
# DISPLAY RESULTS
# ==================================================

if st.session_state.analysis_df is not None:

    df = st.session_state.analysis_df


    # -----------------------------
    # Basic Statistics
    # -----------------------------

    positive = len(
        df[df["Sentiment"] == "POSITIVE"]
    )

    negative = len(
        df[df["Sentiment"] == "NEGATIVE"]
    )

    total = len(df)


    positive_percentage = (
        positive / total
    ) * 100

    negative_percentage = (
        negative / total
    ) * 100


    # -----------------------------
    # Sentiment Overview
    # -----------------------------

    st.subheader(
        "📊 Sentiment Overview"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "💬 Total Comments",
            total
        )

    with col2:

        st.metric(
            "😊 Positive",
            positive
        )

    with col3:

        st.metric(
            "😠 Negative",
            negative
        )

    with col4:

        st.metric(
            "📈 Positive %",
            f"{positive_percentage:.1f}%"
        )


    # -----------------------------
    # Sentiment Distribution
    # -----------------------------

    st.subheader(
        "📈 Sentiment Distribution"
    )

    chart_data = pd.DataFrame(
        {
            "Comments": [
                positive,
                negative
            ]
        },
        index=[
            "Positive",
            "Negative"
        ]
    )

    st.bar_chart(chart_data)


    # -----------------------------
    # Sentiment Percentages
    # -----------------------------

    st.subheader(
        "📊 Sentiment Percentages"
    )

    percentage_data = pd.DataFrame(
        {
            "Percentage": [
                positive_percentage,
                negative_percentage
            ]
        },
        index=[
            "Positive",
            "Negative"
        ]
    )

    st.bar_chart(
        percentage_data
    )


    # ==================================================
    # AI VIEWER INSIGHTS
    # ==================================================

    if "insights" in st.session_state:

        st.subheader(
            "🤖 AI-Generated Viewer Insights"
        )

        st.info(
            "The following insights are generated "
            "from representative viewer comments using "
            "a Hugging Face AI summarization model."
        )


        st.markdown("### 🤖 AI Viewer Analysis")

        st.write(
           st.session_state.insights
        )



    # -----------------------------
    # Most Positive Comment
    # -----------------------------

    st.subheader(
        "😊 Most Positive Comment"
    )

    positive_df = df[
        df["Sentiment"] == "POSITIVE"
    ]

    if not positive_df.empty:

        best_comment = positive_df.loc[
            positive_df["Confidence"].idxmax()
        ]

        st.info(
            best_comment["Comment"]
        )

        st.write(
            "Confidence:",
            f"{best_comment['Confidence'] * 100:.2f}%"
        )


    # -----------------------------
    # Most Negative Comment
    # -----------------------------

    st.subheader(
        "😠 Most Negative Comment"
    )

    negative_df = df[
        df["Sentiment"] == "NEGATIVE"
    ]

    if not negative_df.empty:

        worst_comment = negative_df.loc[
            negative_df["Confidence"].idxmax()
        ]

        st.warning(
            worst_comment["Comment"]
        )

        st.write(
            "Confidence:",
            f"{worst_comment['Confidence'] * 100:.2f}%"
        )


    # -----------------------------
    # Search Comments
    # -----------------------------

    st.subheader(
        "🔎 Search Comments"
    )

    search_text = st.text_input(
        "Search for a word or phrase"
    )


    if search_text:

        filtered_df = df[
            df["Comment"].str.contains(
                search_text,
                case=False,
                na=False
            )
        ]

    else:

        filtered_df = df


    # -----------------------------
    # Results Table
    # -----------------------------

    st.subheader(
        "📋 Comment Analysis"
    )

    st.dataframe(
        filtered_df,
        use_container_width=True
    )


    # -----------------------------
    # Download CSV
    # -----------------------------

    csv = df.to_csv(
        index=False
    )

    st.download_button(
        label="⬇️ Download Complete Results",
        data=csv,
        file_name="youtube_sentiment_results.csv",
        mime="text/csv"
    )