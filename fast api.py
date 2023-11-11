#FASTAPIの設定
from fastapi import FastAPI, UploadFile, HTTPException
import cv2
import numpy as np

app = FastAPI()

@app.post("/infer/")
async def predict(file: UploadFile):
    file_contents = await file.read()
    image = cv2.imdecode(np.frombuffer(file_contents, np.uint8), -1)

    # ここで姿勢推定の処理を行う
    # result = your_pose_estimation_function(image)

    return {"result": result}


#STREAMLITの設定
import streamlit as st
import requests

uploaded_file = st.file_uploader("動画を選択してください", type=['mp4', 'avi'])

if uploaded_file:
    response = requests.post("http://localhost:8000/infer/", files={"file": uploaded_file.getvalue()})
    
    if response.status_code == 200:
        result = response.json()["result"]
        st.write("推論結果:", result)
    else:
        st.write("推論に失敗しました。")


#FASTAPIの起動
uvicorn your_fastapi_file_name:app --reload


#STREAMLITの起動
streamlit run your_streamlit_file_name.py

