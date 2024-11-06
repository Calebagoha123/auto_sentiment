import streamlit as st
import matplotlib.pyplot as plt
from sentiment_analysis import analyze_sentiment, visualize_sentiment, generate_wordcloud, highlight_sentiment, detect_emotion, plot_emotion_results
import fitz  # for PDF extraction

def extract_text_from_pdf(file):
    # Use file-like object from Streamlit
    pdf_reader = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page_num in range(pdf_reader.page_count):
        page = pdf_reader[page_num]
        text += page.get_text()
    return text

def main():
    st.title("Sentiment Analysis")
    
    # Upload or manual text entry
    upload_option = st.radio("Upload PDF or enter text manually:", ("PDF", "Manual Entry"))
    
    if upload_option == "PDF":
        pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])
        if pdf_file:
            text = extract_text_from_pdf(pdf_file)
    else:
        text = st.text_area("Enter your text here:")
    
    # Sentiment analysis options
    analysis_type = st.selectbox("Analyze by:", ("Sentence", "Full Text"))
    
    # Select short-form (social media) or long-form text
    text_length = st.selectbox("Is this short-form (e.g., social media) or long-form text?", ("Short Form", "Long Form"))
    
    # Analysis options
    analysis_option = st.radio("Choose Analysis Type:", ("Sentiment Analysis", "Emotion Detection"))
    
    if st.button("Analyze"):
        try:
            if analysis_option == "Sentiment Analysis":
                # Adjust sentiment analysis processing based on text length
                sentiment_results = analyze_sentiment(text, analysis_type, text_length)
                
                # Display results and plot
                st.write("Sentiment Results:", sentiment_results)
                fig = visualize_sentiment(sentiment_results)
                st.pyplot(fig)
                
                # Word Cloud
                st.subheader("Word Cloud Visualization")
                wordcloud_fig = generate_wordcloud(text, 'country_shape.jpg')
                st.pyplot(wordcloud_fig)
                
                # Sentiment Highlighting
                st.subheader("Sentiment Highlighted Text")
                highlighted_text = highlight_sentiment(text)
                for sentence, sentiment in highlighted_text:
                    if sentiment == 'positive':
                        st.markdown(f'<span style="background-color: #90EE90">{sentence}</span>', unsafe_allow_html=True)
                    elif sentiment == 'negative':
                        st.markdown(f'<span style="background-color: #FFB6C1">{sentence}</span>', unsafe_allow_html=True)
                    else:
                        st.write(sentence)
            elif analysis_option == "Emotion Detection":
                emotion_results = detect_emotion(text, analysis_type)
                st.write("Emotion Detection Results:")
                
                if analysis_type == "Sentence":
                    for item in emotion_results:
                        st.write(f"Sentence: {item['sentence']}")
                        fig = plot_emotion_results(item['emotions'])
                        st.pyplot(fig)
                else:
                    fig = plot_emotion_results(emotion_results)
                    st.pyplot(fig)
        except Exception as e:
            st.warning("Please enter some text or upload a PDF file.")
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
