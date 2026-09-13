\# YouTube Comment Sentiment Analyzer



An AI-powered Streamlit application that analyzes YouTube viewer comments using the YouTube Data API and Hugging Face Transformer models.



\## Features



\- Fetches YouTube comments using YouTube Data API v3

\- Automatically extracts YouTube video IDs

\- Performs Transformer-based sentiment analysis

\- Calculates positive and negative sentiment distribution

\- Displays sentiment confidence scores

\- Identifies highly confident positive and negative comments

\- Provides interactive comment search

\- Generates AI-based viewer insights

\- Allows analyzed results to be downloaded as CSV

\- Uses Streamlit for an interactive web interface



\## Technologies Used



\- Python

\- Streamlit

\- Pandas

\- YouTube Data API v3

\- Hugging Face Transformers

\- PyTorch

\- Natural Language Processing (NLP)

\- Sentiment Analysis

\- Large Language Model (LLM)



\## Project Workflow



YouTube Video URL  

↓  

YouTube Data API  

↓  

Comment Collection  

↓  

Transformer Sentiment Analysis  

↓  

Sentiment Classification  

↓  

Statistical Analysis  

↓  

AI Viewer Insights  

↓  

Interactive Dashboard



\## Output



The application provides:



\- Total number of comments

\- Positive and negative comment counts

\- Sentiment percentages

\- Sentiment visualization

\- Most positive comment

\- Most negative comment

\- AI-generated viewer insights

\- Searchable comment analysis table

\- CSV download



\## API Key Setup



Create:



`.streamlit/secrets.toml`



and add:



```toml

YT\_APIKEY = "YOUR\_YOUTUBE\_API\_KEY"

