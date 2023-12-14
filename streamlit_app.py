# streamlit_app.py

import streamlit as st
import pandas as pd  # Import pandas for data manipulation
import plotly.express as px  # Import Plotly for interactive visualizations
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from bs4 import BeautifulSoup
import requests
import re
from transformers import pipeline
import csv

# Setup Model
model_name = "human-centered-summarization/financial-summarization-pegasus"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)

# The Streamlit app
def main():
    st.title("Stock News Sentiment Analysis")

    # Get user input for tickers
tickers_input = st.text_input("Enter Tickers (comma-separated):", "AAPL,TSLA")

    # Split tickers by comma and create a list
monitored_tickers = [ticker.strip() for ticker in tickers_input.split(',')]
# Fetch stock news URLs using mboum-finance API
def search_for_stock_news_urls(ticker):
    url = "https://mboum-finance.p.rapidapi.com/v1/markets/news"

    querystring = {"tickers": ticker}

    headers = {
        "X-RapidAPI-Key": "3ef69b170emshd75a0378b325a57p151becjsn2b70213f10cd",
        "X-RapidAPI-Host": "mboum-finance.p.rapidapi.com",
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.1234.56 Safari/537.36'
    }

    response = requests.get(url, headers=headers, params=querystring)

    if response.status_code == 200:
        data = response.json()
        articles = data.get("body", [])

        if articles:
            return [article.get("link") for article in articles if article.get("link")]
        else:
            return []
    else:
        return []
    # Fetch stock news URLs using the mboum-finance API
print('Fetching stock news URLs for', monitored_tickers)
raw_urls = {ticker: search_for_stock_news_urls(ticker) for ticker in monitored_tickers}
    
# Strip out unwanted URLs
print('Cleaning URLs.')
exclude_list = ['maps', 'policies', 'preferences', 'accounts', 'support']
def strip_unwanted_urls(urls, exclude_list):
    val = []
    for url in urls:
        if 'https://' in url and not any(exc in url for exc in exclude_list):
            res = re.findall(r'(https?://\S+)', url)[0].split('&')[0]
            val.append(res)
    return list(set(val))

    # Strip out unwanted URLs
exclude_list = ['maps', 'policies', 'preferences', 'accounts', 'support']
cleaned_urls = {ticker: strip_unwanted_urls(raw_urls[ticker], exclude_list) for ticker in monitored_tickers}

# Scrape Cleaned URLs
def scrape_and_process(URLs):
    ARTICLES = []
    for url in URLs:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # Extract text content from the webpage
            article_text = ' '.join([p.get_text() for p in soup.find_all('p')])
            ARTICLE = article_text[:350]  # Take the first 350 characters as a sample
            ARTICLES.append(ARTICLE)
    return ARTICLES

    # Scrape Cleaned URLs
articles = {ticker: scrape_and_process(cleaned_urls[ticker]) for ticker in monitored_tickers}

# Summarize all Articles
def summarize(articles):
    summaries = []
    for article in articles:
        input_ids = tokenizer.encode(article, return_tensors="pt")
        output = model.generate(input_ids, max_length=95, num_beams=5, early_stopping=True)
        summary = tokenizer.decode(output[0], skip_special_tokens=True)
        summaries.append(summary)
    return summaries
    # Summarize all Articles
summaries = {ticker: summarize(articles[ticker]) for ticker in monitored_tickers}

# Adding Sentiment Analysis
print('Calculating sentiment.')
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", revision="af0f99b")
scores = {ticker: sentiment_analyzer(summaries[ticker]) for ticker in monitored_tickers}



# # 6. Exporting Results
print('Exporting results')
def create_output_array(summaries, scores, urls):
    output = []
    for ticker in monitored_tickers:
        for counter in range(len(summaries[ticker])):
            output_this = [
                            ticker, 
                            summaries[ticker][counter], 
                            scores[ticker][counter]['label'], 
                            scores[ticker][counter]['score'], 
                            urls[ticker][counter]
                          ]
            output.append(output_this)
    return output
final_output = create_output_array(summaries, scores, cleaned_urls)
final_output.insert(0, ['Ticker','Summary', 'Sentiment', 'Sentiment Score', 'URL'])

# Display the results in the Streamlit app
st.write("### Summaries")
st.table(final_output)

# Convert the summaries and scores to a DataFrame for visualization
df_visualization = pd.DataFrame(final_output[1:], columns=final_output[0])

# Interactive Sentiment Analysis Visualization
st.write("### Sentiment Analysis Visualization")

fig = make_subplots(rows=1, cols=len(monitored_tickers), subplot_titles=monitored_tickers)

for idx, ticker in enumerate(monitored_tickers):
    fig.add_trace(
        go.Box(
            y=df_visualization[df_visualization['Ticker'] == ticker]['Sentiment Score'],
            name=ticker
        ),
        row=1, col=idx + 1
    )

fig.update_layout(height=400, showlegend=False)
st.plotly_chart(fig)

with open('summaries.csv', mode='w', newline='') as f:
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerows(final_output)

if __name__ == "__main__":
    main()
