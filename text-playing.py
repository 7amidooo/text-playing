import streamlit as st
from transformers import pipeline
import torch
import gc

# Caching models to optimize performance
@st.cache_resource
def load_summarization_model():
    return pipeline("summarization", model="facebook/bart-large-cnn", trust_remote_code=True)

@st.cache_resource
def load_translation_model(lang_code):
    return pipeline("translation", model=f"Helsinki-NLP/opus-mt-en-{lang_code}", trust_remote_code=True)

@st.cache_resource
def load_text_generation_model():
    return pipeline("text-generation", model="gpt2", trust_remote_code=True)

@st.cache_resource
def load_ner_model():
    return pipeline("ner", grouped_entities=True, trust_remote_code=True)

@st.cache_resource
def load_qa_model():
    return pipeline("question-answering", trust_remote_code=True)


# Task-specific functions
def perform_text_summarization():
    user_input = st.text_area("Enter the text you want to summarize:")
    max_length = st.slider("Max Length of Summary:", min_value=30, max_value=200, value=100)
    if user_input and st.button("Summarize", key="summarize"):
        with st.spinner("Summarizing the text..."):
            summarizer = load_summarization_model()
            summary = summarizer(user_input, max_length=max_length, min_length=30, do_sample=False)
            st.write(f"Summary: {summary[0]['summary_text']}")


def perform_translation():
    user_input = st.text_area("Enter the text you want to translate:")
    target_language = st.selectbox("Select the target language", ["French", "Spanish", "German", "Arabic"])
    lang_codes = {"French": "fr", "Spanish": "es", "German": "de", "Arabic": "ar"}
    
    if user_input and st.button("Translate", key="translate"):
        with st.spinner("Translating the text..."):
            translator = load_translation_model(lang_codes[target_language])
            translation = translator(user_input)
            st.write(f"Translation: {translation[0]['translation_text']}")


def perform_text_generation():
    user_input = st.text_area("Enter the beginning of your text:")
    max_length = st.slider("Max Length of Generated Text:", min_value=50, max_value=200, value=100)
    
    if user_input and st.button("Generate", key="generate"):
        with st.spinner("Generating text..."):
            text_generator = load_text_generation_model()
            generated_text = text_generator(user_input, max_length=max_length, num_return_sequences=1)
            st.write(f"Generated Text: {generated_text[0]['generated_text']}")


def perform_ner():
    user_input = st.text_area("Enter text to extract named entities:")
    
    if user_input and st.button("Extract Entities", key="ner"):
        with st.spinner("Extracting named entities..."):
            ner_pipeline = load_ner_model()
            entities = ner_pipeline(user_input)
            st.write("Named Entities:")
            for entity in entities:
                st.write(f"{entity['word']} ({entity['entity_group']}) - Confidence: {entity['score']:.2f}")


def perform_question_answering():
    st.write("Upload a text document (plain text only) or enter a paragraph for question answering.")
    uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
    
    if uploaded_file:
        document = uploaded_file.read().decode("utf-8")
    else:
        document = st.text_area("Or, paste a paragraph:")
    
    if document:
        st.write("Document Preview:")
        st.write(document)
        question = st.text_input("Enter your question:")
        
        if question and st.button("Get Answer", key="qa"):
            with st.spinner("Answering your question..."):
                qa_pipeline = load_qa_model()
                answer = qa_pipeline(question=question, context=document)
                st.write(f"Answer: {answer['answer']}")


# Main app logic
st.title("Text Toolkit AI")
st.sidebar.title("Choose a task:")
task = st.sidebar.selectbox(
    "Select a task", 
    ("Text Summarization", 
     "Translation", 
     "Text Generation", 
     "Named Entity Recognition", 
     "Question Answering")
)

# Execute the selected task
if task == "Text Summarization":
    perform_text_summarization()
elif task == "Translation":
    perform_translation()
elif task == "Text Generation":
    perform_text_generation()
elif task == "Named Entity Recognition":
    perform_ner()
elif task == "Question Answering":
    perform_question_answering()

# Call garbage collection at the end to free up memory
gc.collect()
