import streamlit as st
import os
import json
from dotenv import load_dotenv
import requests

# LangChain系
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from app import predict_image

# watsonx.ai Service
from langchain_ibm import WatsonxLLM

# 環境変数のロード
load_dotenv()

# LLMのインスタンス作成
WATSONX_AI_ENDPOINT = os.getenv("WATSONX_AI_ENDPOINT")
WATSONX_AI_PROJECT_ID = os.getenv("WATSONX_AI_PROJECT_ID")
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")

# 環境変数のエラーチェック
if not all([WATSONX_AI_ENDPOINT, WATSONX_AI_PROJECT_ID, WATSONX_API_KEY]):
    st.error("環境変数 (WATSONX_AI_ENDPOINT, WATSONX_AI_PROJECT_ID, WATSONX_API_KEY) が設定されていません。")
    st.stop()

# LLMのパラメータ設定
params = {
    "max_new_tokens": 16000,
    "min_new_tokens": 1
}

llm = WatsonxLLM(
    model_id="mistralai/mistral-large",
    url=WATSONX_AI_ENDPOINT,
    apikey=WATSONX_API_KEY,
    params=params,
    project_id=WATSONX_AI_PROJECT_ID,
)

# タイトル
st.title("食べ物の予測とカロリー推定")

# File uploader for image input
uploaded_file = st.file_uploader("食べ物の画像をアップロード", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Display uploaded image
    st.image(uploaded_file, caption="アップロードされた画像", use_container_width=True)
    
    # Predict food category
    st.write("画像を分析中...")
    predictions = predict_image(uploaded_file)
    
    # Display predictions
    st.subheader("予測結果")
    food_names = [pred['label'] for pred in predictions]  # Ensure food_names is defined here
    for pred in predictions:
        st.write(f"**{pred['label']}**: {pred['score']*100:.2f}% 確率")
    
    # Generate calorie estimation using LangChain and Watsonx
    st.subheader("カロリー推定")
    if food_names:  # Check if food_names is non-empty before proceeding
        # Construct prompt for LLM
        prompt_template = PromptTemplate(
            input_variables=["food_names"],
            template="以下の食べ物の一人前のカロリー推定を行なってください: {food_names}. 範囲がある場合は平均値をにして簡潔に回答してください。」"
        )
        query = prompt_template.format(food_names=", ".join(food_names))
        
        try:
            response = llm.generate([query])
            st.write(response.generations[0][0].text)
        except Exception as e:
            st.error(f"An error occurred during calorie estimation: {e}")