# streamlit_app.py
import streamlit as st
import requests

st.title('テニスフォーム判別アプリ')
st.write('Play Tennis')

uploaded_file = st.file_uploader('動画を選択してください...', type='mp4')

if uploaded_file:
    response = requests.post("http://localhost:8000/upload_video", files={"file": uploaded_file})
    if response.status_code == 200:
        st.write("ビデオ処理が完了しました")
        # APIから受け取った他の結果を表示します
    else:
        st.write("ビデオの処理中にエラーが発生しました")
        

if __name__ == "__main__":
    st.run()