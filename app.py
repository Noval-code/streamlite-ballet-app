from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import pymongo
import schedule
import time
import plotly.express as px
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# MongoDB connection
try:
    mongo_uri = st.secrets["mongo"]["uri"]
    client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    client.admin.command('ping')
    st.success("✔️ Koneksi ke MongoDB Atlas berhasil")
except Exception as e:
    st.error(f"❌ Gagal koneksi ke MongoDB Atlas:\n{e}")
    st.stop()

db = client["BIGDATA"]
collection = db["ballet"]

# Custom ballet-related vocabulary (feel free to expand this)
ballet_vocabulary = {
    'ballet', 'dancer', 'dance', 'performance', 'rehearsal', 'choreography', 'balletic', 'pirouette',
    'ballerina', 'balletschool', 'balletcompany', 'pas', 'tendu', 'arabesque', 'pointe', 'pas de deux',
    'balletdance', 'balletperformance', 'balletclass', 'danceacademy', 'danceday', 'balletshow',
    'balet', 'penari', 'tari', 'pertunjukan', 'latihan', 'koreografi', 'gerakan balet', 'putaran',
    'ballerina', 'sekolah balet', 'perusahaan balet', 'langkah', 'tendu', 'arabesque', 'ujung jari',
    'pas de deux', 'tari balet', 'pertunjukan balet', 'kelas balet', 'akademi tari', 'hari tari', 'pertunjukan balet',
    'ballet klasik', 'pentas balet', 'gerak balet', 'tarian balet', 'komunitas balet', 'pembelajaran balet',
    'pas ballerina', 'karya balet'
}

# Additional stopwords (standard plus any other words you don't want to include)
stop_words = {
    'yang', 'di', 'ke', 'dari', 'pada', 'dalam', 'untuk', 'dengan', 'dan', 'atau',
    'ini', 'itu', 'juga', 'sudah', 'saya', 'anda', 'dia', 'mereka', 'kita', 'akan',
    'bisa', 'ada', 'tidak', 'saat', 'oleh', 'setelah', 'para', 'seperti', 'saat',
    'bagi', 'serta', 'tapi', 'lain', 'sebuah', 'karena', 'ketika', 'jika', 'apa',
    'seorang', 'tentang', 'dalam', 'bisa', 'sementara', 'dilakukan', 'setelah',
    'yakni', 'menurut', 'hampir', 'dimana', 'bagaimana', 'selama', 'sebelum', 
    'hingga', 'kepada', 'sebagai', 'masih', 'hal', 'sempat', 'sedang', 'selain',
    'sembari', 'mendapat', 'sedangkan', 'tetapi', 'membuat', 'namun', 'gimana'
}

def scrape_detik():
    articles = []
    base_url = "https://www.detik.com/tag/balet"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(base_url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        articles_container = soup.find_all('article')
        
        for article in articles_container:
            try:
                title_element = article.find('h2', class_='title')
                link_element = article.find('a')
                date_element = article.find('span', class_='date')
                category_element = article.find('span', class_='category')
                
                if all([title_element, link_element, date_element]):
                    title = title_element.text.strip()
                    link = link_element['href']
                    date = date_element.text.strip()
                    category = category_element.text.strip() if category_element else "Unknown"
                    
                    articles.append({
                        'source': 'Detik',
                        'title': title,
                        'date': date,
                        'link': link,
                        'category': category,
                        'scraped_at': datetime.now()
                    })
                    
            except Exception as e:
                st.write(f"Error parsing article: {str(e)}")
                continue
            
        st.success(f"Found {len(articles)} articles from Detik")
            
    except Exception as e:
        st.error(f"Error scraping Detik: {str(e)}")
    
    return articles

def save_to_mongodb(articles):
    try:
        if not articles:
            st.warning("No articles to save")
            return
            
        collection.insert_many(articles)
        st.success(f"Saved {len(articles)} articles to MongoDB")
    except Exception as e:
        st.error(f"Error saving to MongoDB: {str(e)}")

def visualize_data():
    # Fetch data from MongoDB
    data = list(collection.find())
    from dateutil import parser
    import pandas as pd
    import streamlit as st
    import plotly.express as px
    import re

    def extract_date(text):
        try:
            # Ambil tanggal dari format seperti "WolipopKamis, 06 Feb 2025 12:57 WIB"
            cleaned = re.search(r'(\d{1,2} \w+ \d{4}.*)', text)
            if cleaned:
                return parser.parse(cleaned.group(1), fuzzy=True)
            return None
        except:
            return None

    df = pd.DataFrame(data)

    if len(df) == 0:
        st.warning("No data available for visualization")
        return

    # Parsing tanggal
    df['parsed_date'] = df['date'].apply(extract_date)
    df = df.dropna(subset=['parsed_date'])

    # Ekstrak tahun dan bulan
    df['year'] = df['parsed_date'].dt.year
    df['month'] = df['parsed_date'].dt.month

    filtered_df = df

    # 1. Number of articles by source
    st.subheader("1. Number of Articles by Source")
    source_counts = filtered_df['source'].value_counts()
    fig1 = px.bar(source_counts, title='Total Articles by Source')
    st.plotly_chart(fig1)

    # 2. Yearly trends
    st.subheader("2. Yearly Trends")
    yearly_data = filtered_df.groupby('year').size().reset_index(name='count')
    yearly_data['year'] = yearly_data['year'].astype(int)

    fig2 = px.bar(yearly_data, x='year', y='count', title='Articles by Year')
    fig2.update_xaxes(tickmode='array', tickvals=yearly_data['year'].unique())
    st.plotly_chart(fig2)

    # 3. Monthly trends for selected year
    st.subheader("3. Monthly Trends for Selected Year")
    selected_year = st.selectbox('Select Year for Monthly Breakdown', sorted(filtered_df['year'].unique(), reverse=True))

    monthly_data = filtered_df[filtered_df['year'] == selected_year] \
        .groupby(['month', 'source']).size().reset_index(name='count')

    # Ubah angka bulan jadi nama bulan
    monthly_data['month'] = pd.to_datetime(monthly_data['month'], format='%m').dt.strftime('%B')

    fig3 = px.bar(monthly_data, x='month', y='count', color='source',
                  title=f'Monthly Articles in {selected_year}')
    st.plotly_chart(fig3)

    # Debugging output (optional)
    # st.write(df[['date', 'parsed_date', 'year', 'month']].head(10))

    # 4. Word frequency analysis as word cloud
    st.subheader("4. Most Common Words - Bar Chart & Word Cloud")
    
    try:
        # Process text
        all_words = []
        for title in filtered_df['title']:
            # Convert to lowercase and split by spaces
            words = title.lower().split()
            # Clean words and filter stopwords and non-ballet-related words
            words = [word.strip('.,!?()[]{}:;"\'') for word in words]
            words = [word for word in words if word and word not in stop_words and word in ballet_vocabulary]
            all_words.extend(words)
        
        if all_words:
            # Create a word cloud
            wordcloud = WordCloud(width=800, height=400, background_color='black').generate(' '.join(all_words))
            
            # Display the word cloud
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')  # Hide the axes
            st.pyplot(plt)
            
            # Bar Chart for Most Common Words
            st.subheader("5. Most Common Words - Bar Chart")
            word_freq = pd.Series(all_words).value_counts().head(15)
            fig4 = px.bar(
                word_freq, 
                title='Most Common Words in Titles',
                labels={'index': 'Word', 'value': 'Frequency'}
            )
            st.plotly_chart(fig4)
        else:
            st.warning("No words to analyze after filtering")
            
    except Exception as e:
        st.error(f"Error in word frequency analysis: {str(e)}")

    # 5. Data tables by source
    st.subheader("5. Article Details")
    st.dataframe(filtered_df[['title', 'date', 'link', 'scraped_at']])
def main():
    st.title("Ballet News Scraper")
    
    # Main content
    tab1, tab2 = st.tabs(["Recent Articles", "Visualizations"])
    
    with tab1:
        st.subheader("Recent Articles")
        articles = list(collection.find().sort('scraped_at', -1).limit(10))
        for article in articles:
            st.write(f"**{article['title']}**")
            st.write(f"Date: {article['date']} | Category: {article['category']}") 
            st.write(f"[Read more]({article['link']})")
            st.markdown("---")
    
    with tab2:
        st.subheader("Data Visualization")
        visualize_data()

if __name__ == "__main__":
    main()
