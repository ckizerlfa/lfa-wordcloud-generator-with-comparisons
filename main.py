import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import re
from collections import Counter
import io

# Set page configuration
st.set_page_config(
    page_title="Word Cloud and Keyword Gap Analysis App",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Download only stopwords
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('stopwords', quiet=True)
    except Exception as e:
        st.error(f"Error downloading NLTK data: {str(e)}")
        return False
    return True

def simple_tokenize(text):
    """Simple word tokenization using regex."""
    # Split on word boundaries and remove empty strings
    return [word.strip() for word in re.findall(r'\b\w+\b', text.lower()) if word.strip()]

def clean_text(text):
    """Clean and preprocess text for word cloud generation."""
    if pd.isna(text):
        return ""
    
    # Convert to string if not already
    text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    try:
        # Use simple tokenization
        tokens = simple_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        
        return ' '.join(tokens)
    except Exception as e:
        st.error(f"Error in text cleaning: {str(e)}")
        return text

def generate_wordcloud(text, max_words, width, height, background_color):
    """Generate a WordCloud object."""
    wordcloud = WordCloud(
        width=width,
        height=height,
        background_color=background_color,
        max_words=max_words,
        contour_width=1,
        contour_color='steelblue'
    ).generate(text)
    return wordcloud

def create_frequency_table(text):
    """Create a frequency table from text."""
    words = text.split()
    word_freq = Counter(words)
    freq_df = pd.DataFrame(word_freq.items(), columns=['Word', 'Frequency'])
    freq_df = freq_df.sort_values(by='Frequency', ascending=False).reset_index(drop=True)
    return freq_df

def perform_gap_analysis(client_freq, competitor_freq, top_n):
    """Perform keyword gap analysis between client and competitor."""
    # Convert to dictionaries
    client_dict = dict(zip(client_freq['Word'], client_freq['Frequency']))
    competitor_dict = dict(zip(competitor_freq['Word'], competitor_freq['Frequency']))
    
    # Words in competitor not in client
    opportunities = []
    for word, freq in competitor_dict.items():
        client_freq_val = client_dict.get(word, 0)
        if freq > client_freq_val:
            opportunities.append({
                'Word': word,
                'Competitor Frequency': freq,
                'Client Frequency': client_freq_val,
                'Frequency Difference': freq - client_freq_val
            })
    
    # Words in client not in competitor
    strengths = []
    for word, freq in client_dict.items():
        competitor_freq_val = competitor_dict.get(word, 0)
        if freq > competitor_freq_val:
            strengths.append({
                'Word': word,
                'Client Frequency': freq,
                'Competitor Frequency': competitor_freq_val,
                'Frequency Difference': freq - competitor_freq_val
            })
    
    # Create DataFrames
    opportunities_df = pd.DataFrame(opportunities).sort_values(by='Frequency Difference', ascending=False).head(top_n)
    strengths_df = pd.DataFrame(strengths).sort_values(by='Frequency Difference', ascending=False).head(top_n)
    
    return opportunities_df, strengths_df

def main():
    st.title("📝 Word Cloud and Keyword Gap Analysis App")
    
    # Download NLTK stopwords at startup
    with st.spinner("Downloading required NLTK data..."):
        if not download_nltk_data():
            st.error("Failed to download required NLTK data. Please try again.")
            return
    
    st.markdown("""
    ### Upload Your Datasets
    Upload two files containing articles: one for the **Client** and another for the **Competitor**.
    Each file should be a CSV or Excel file with **exactly one column** containing the article texts.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_client = st.file_uploader("📄 Upload Client File", type=['xlsx', 'csv'], key='client')
    
    with col2:
        uploaded_competitor = st.file_uploader("📄 Upload Competitor File", type=['xlsx', 'csv'], key='competitor')
    
    if uploaded_client and uploaded_competitor:
        try:
            # Function to read uploaded file
            def read_uploaded_file(uploaded_file):
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                return df
            
            # Read client file
            client_df = read_uploaded_file(uploaded_client)
            if client_df.shape[1] != 1:
                st.error("**Client file** must contain exactly one column with article texts.")
                return
            if client_df.empty:
                st.error("**Client file** is empty.")
                return
            
            # Read competitor file
            competitor_df = read_uploaded_file(uploaded_competitor)
            if competitor_df.shape[1] != 1:
                st.error("**Competitor file** must contain exactly one column with article texts.")
                return
            if competitor_df.empty:
                st.error("**Competitor file** is empty.")
                return
            
            # Sidebar Controls
            st.sidebar.header("📊 Word Cloud Settings")
            max_words = st.sidebar.slider("Maximum number of words", 50, 500, 200, key='max_words')
            width = st.sidebar.slider("Word Cloud Width (px)", 400, 2000, 800, key='width')
            height = st.sidebar.slider("Word Cloud Height (px)", 400, 2000, 400, key='height')
            background_color = st.sidebar.color_picker("Background Color", "#ffffff", key='bg_color')
            
            st.sidebar.header("📈 Frequency Table Settings")
            top_n_freq = st.sidebar.slider("Top N Words in Frequency Tables", 10, 500, 100, key='top_n_freq')
            
            st.sidebar.header("🔍 Gap Analysis Settings")
            top_n_gap = st.sidebar.slider("Top N Gaps to Display", 10, 100, 20, key='top_n_gap')
            
            # Clean and combine texts
            with st.spinner("Cleaning and processing texts..."):
                client_text = ' '.join(client_df.iloc[:,0].apply(clean_text))
                competitor_text = ' '.join(competitor_df.iloc[:,0].apply(clean_text))
            
            if not client_text.strip():
                st.error("No valid text found in the **Client** file after cleaning.")
                return
            if not competitor_text.strip():
                st.error("No valid text found in the **Competitor** file after cleaning.")
                return
            
            # Generate Word Clouds
            with st.spinner("Generating word clouds..."):
                client_wordcloud = generate_wordcloud(client_text, max_words, width, height, background_color)
                competitor_wordcloud = generate_wordcloud(competitor_text, max_words, width, height, background_color)
            
            # Display Word Clouds Side by Side
            st.markdown("### 🟢 Client Word Cloud vs 🔴 Competitor Word Cloud")
            wc_col1, wc_col2 = st.columns(2)
            
            with wc_col1:
                st.subheader("🟢 Client Word Cloud")
                fig1, ax1 = plt.subplots(figsize=(width/100, height/100), dpi=100)
                ax1.imshow(client_wordcloud, interpolation='bilinear')
                ax1.axis('off')
                plt.tight_layout(pad=0)
                st.pyplot(fig1)
                
                # Download Client Word Cloud
                buf_client = io.BytesIO()
                client_wordcloud.to_image().save(buf_client, format='PNG')
                byte_client = buf_client.getvalue()
                st.download_button(
                    label="📥 Download Client Word Cloud",
                    data=byte_client,
                    file_name="client_wordcloud.png",
                    mime="image/png"
                )
            
            with wc_col2:
                st.subheader("🔴 Competitor Word Cloud")
                fig2, ax2 = plt.subplots(figsize=(width/100, height/100), dpi=100)
                ax2.imshow(competitor_wordcloud, interpolation='bilinear')
                ax2.axis('off')
                plt.tight_layout(pad=0)
                st.pyplot(fig2)
                
                # Download Competitor Word Cloud
                buf_competitor = io.BytesIO()
                competitor_wordcloud.to_image().save(buf_competitor, format='PNG')
                byte_competitor = buf_competitor.getvalue()
                st.download_button(
                    label="📥 Download Competitor Word Cloud",
                    data=byte_competitor,
                    file_name="competitor_wordcloud.png",
                    mime="image/png"
                )
            
            # Generate Frequency Tables
            with st.spinner("Generating frequency tables..."):
                client_freq_df = create_frequency_table(client_text).head(top_n_freq)
                competitor_freq_df = create_frequency_table(competitor_text).head(top_n_freq)
            
            # Display Frequency Tables
            st.markdown("### 📊 Keyword Frequency Tables")
            freq_col1, freq_col2 = st.columns(2)
            
            with freq_col1:
                st.subheader("🟢 Client Keyword Frequency")
                st.dataframe(client_freq_df, width=600, height=400)
                
                # Download Client Frequency Table
                csv_client = client_freq_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Client Frequency Table (CSV)",
                    data=csv_client,
                    file_name='client_frequency_table.csv',
                    mime='text/csv'
                )
            
            with freq_col2:
                st.subheader("🔴 Competitor Keyword Frequency")
                st.dataframe(competitor_freq_df, width=600, height=400)
                
                # Download Competitor Frequency Table
                csv_competitor = competitor_freq_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Competitor Frequency Table (CSV)",
                    data=csv_competitor,
                    file_name='competitor_frequency_table.csv',
                    mime='text/csv'
                )
            
            # Perform Gap Analysis
            with st.spinner("Performing keyword gap analysis..."):
                opportunities_df, strengths_df = perform_gap_analysis(client_freq_df, competitor_freq_df, top_n_gap)
            
            # Display Gap Analysis
            st.markdown("### 🔍 Keyword Gap Analysis")
            
            gap_col1, gap_col2 = st.columns(2)
            
            with gap_col1:
                st.subheader("🟢 Client Opportunities")
                if not opportunities_df.empty:
                    st.dataframe(opportunities_df, width=600, height=400)
                    
                    # Download Opportunities Table
                    csv_opportunities = opportunities_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Opportunities (CSV)",
                        data=csv_opportunities,
                        file_name='client_opportunities.csv',
                        mime='text/csv'
                    )
                else:
                    st.info("No significant opportunities found where competitor uses words not used by client.")
            
            with gap_col2:
                st.subheader("🟢 Client Strengths")
                if not strengths_df.empty:
                    st.dataframe(strengths_df, width=600, height=400)
                    
                    # Download Strengths Table
                    csv_strengths = strengths_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Strengths (CSV)",
                        data=csv_strengths,
                        file_name='client_strengths.csv',
                        mime='text/csv'
                    )
                else:
                    st.info("No significant strengths found where client uses words not used by competitor.")
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Full error details:")
            st.exception(e)
    else:
        st.info("Please upload both **Client** and **Competitor** files to proceed.")

if __name__ == "__main__":
    main()
