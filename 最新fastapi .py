import streamlit as st
import cv2
import tempfile
import numpy as np
import mediapipe as mp
import math
from fastapi import FastAPI, File, UploadFile,  HTTPException
from typing import List

app = FastAPI()

# ここに必要な関数を定義します。
def read_video_frames(uploaded_file):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap_file = cv2.VideoCapture(tfile.name)
    frames = []

    while cap_file.isOpened():
        success, image = cap_file.read()
        if not success:
            break
        image = cv2.resize(image, dsize=None, fx=0.3, fy=0.3)
        frames.append(image)

    cap_file.release()

    return frames


@app.post("/process_video/")
async def process_video(file: UploadFile = File(...)):
    try:
        # ファイルを一時的に保存する
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(await file.read())

        frames = read_video_frames(tfile)
        processed_frames = []

        for frame in frames:
            processed_frame = process_frame_for_landmarks(frame)  # 非同期処理を削除
            processed_frames.append(processed_frame)

        return {"processed_frames": processed_frames}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    

@app.post("/calculate_angle/")
async def calculate_angle(points: List[List[int]]):
    try:
        a, b, c = np.array(points[0]), np.array(points[1]), np.array(points[2])
        angle = calculate_angle(a, b, c)
        return {"angle": angle}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/process_frame_for_landmarks/")
async def process_frame_for_landmarks(image: UploadFile = File(...)):
    try:
        # UploadFileをcv2の画像に変換して処理
        np_image = np.fromstring(await image.read(), np.uint8)
        frame = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        processed_frame = process_frame_for_landmarks(frame)  # 非同期処理を削除
        return {"processed_frame": processed_frame.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def process_frame_for_landmarks(image):
# mediapipeの設定
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    mesh_drawing_spec = mp_drawing.DrawingSpec(thickness=2, color=(0,255,0))
    mark_drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2, color=(0,0,255))

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_holistic.Holistic(min_detection_confidence=0.5, static_image_mode=False) as holistic_detection:
        results = holistic_detection.process(rgb_image)

        # 姿勢推定の結果を描画
        mp_drawing.draw_landmarks(image=image,landmark_list=results.face_landmarks, connections=mp_holistic.FACEMESH_TESSELATION,landmark_drawing_spec=None, connection_drawing_spec=mesh_drawing_spec)
        mp_drawing.draw_landmarks(image=image,landmark_list=results.pose_landmarks,connections=mp_holistic.POSE_CONNECTIONS,landmark_drawing_spec=mark_drawing_spec,connection_drawing_spec=mesh_drawing_spec)
        mp_drawing.draw_landmarks(image=image,landmark_list=results.left_hand_landmarks,connections=mp_holistic.HAND_CONNECTIONS,landmark_drawing_spec=mark_drawing_spec,connection_drawing_spec=mesh_drawing_spec)
        mp_drawing.draw_landmarks(image=image,landmark_list=results.right_hand_landmarks,connections=mp_holistic.HAND_CONNECTIONS,landmark_drawing_spec=mark_drawing_spec,connection_drawing_spec=mesh_drawing_spec)

    return image


def calculate_angle(a, b, c):
    # ベクトルの計算
    ba = a - b
    bc = c - b

    # cosine theta
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)

def get_landmarks_pose(image):
    mp_holistic = mp.solutions.holistic
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb_image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
   

    with mp_holistic.Holistic(min_detection_confidence=0.5, static_image_mode=False) as holistic_detection:
        results = holistic_detection.process(rgb_image)
        return results.pose_landmarks
    
def find_point_c(x1, y1, x2, y2,alpha, beta):
    # ポイントAとBの距離を計算
    AB = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    # 角Cを計算
    gamma = math.radians(180 - math.degrees(alpha) - math.degrees(beta))

    # ローソク法を使用してACおよびBCの長さを計算
    AC = AB * math.sin(beta) / math.sin(gamma)
    BC = AB * math.sin(alpha) / math.sin(gamma)

    # 計算された距離と角度を使用して点Cの座標を求める
    x3a = x1 + AC * math.cos(alpha)
    y3a = y1 + AC * math.sin(alpha)

    x3b = x2 + BC * math.cos(beta)
    y3b = y2 + BC * math.sin(beta)

    # 2つの候補点を返す
    return (x3a, y3a), (x3b, y3b)

# 1つ目の動画
uploaded_file1 = st.file_uploader('1つ目の動画を選択してください...', type='mp4', key="uploader1")
if uploaded_file1 is not None:
    frames1 = read_video_frames(uploaded_file1)
    frame_idx1 = st.slider('動画1 フレームを選択', 1, len(frames1)-1, 1, key='slider_1')
    selected_frame1 = frames1[frame_idx1]
    processed_frame1 = process_frame_for_landmarks(selected_frame1)
    st.image(processed_frame1, channels="BGR", caption=f"動画1 {frame_idx1}番目のフレーム")
    
    landmarks1 = get_landmarks_pose(selected_frame1)
    height1=selected_frame1.shape[0]
    width1=selected_frame1.shape[1]
    
    # 2つ目の動画
uploaded_file2 = st.file_uploader('2つ目の動画を選択してください...', type='mp4', key="uploader2")
if uploaded_file2 is not None:
    frames2 = read_video_frames(uploaded_file2)
    frame_idx2 = st.slider('動画2 フレームを選択', 1, len(frames2)-1, 1, key='slider_2')
    selected_frame2 = frames2[frame_idx2]
    processed_frame2 = process_frame_for_landmarks(selected_frame2)
    landmarks2= get_landmarks_pose(selected_frame2)
    height2=selected_frame2.shape[0]
    width2=selected_frame2.shape[1]
    
    if landmarks1:
        options = ["右の肘", "左の肘", "右の膝", "左の膝"] # 他のオプションも追加可能
        choice = st.selectbox("角度を計算するランドマークを選択してください", options)
        
        if choice == "右の肘":
            angle = calculate_angle(np.array([int(landmarks1.landmark[12].x*width1),int(landmarks1.landmark[12].y*height1)]), 
                                    np.array([int(landmarks1.landmark[14].x*width1),int(landmarks1.landmark[14].y*height1)]), 
                                    np.array([int(landmarks1.landmark[16].x*width1),int(landmarks1.landmark[16].y*height1)]))
            
            AL=calculate_angle(np.array([int(landmarks1.landmark[14].x*width1),int(landmarks1.landmark[14].y*height1)]), 
                                    np.array([int(landmarks1.landmark[12].x*width1),int(landmarks1.landmark[12].y*height1)]), 
                                    np.array([int(landmarks1.landmark[16].x*width1),int(landmarks1.landmark[16].y*height1)]))

            BE=calculate_angle(np.array([int(landmarks1.landmark[12].x*width1),int(landmarks1.landmark[12].y*height1)]), 
                                    np.array([int(landmarks1.landmark[16].x*width1),int(landmarks1.landmark[16].y*height1)]), 
                                    np.array([int(landmarks1.landmark[14].x*width1),int(landmarks1.landmark[14].y*height1)]))
            
            x1,y1=int(landmarks2.landmark[12].x*width2),int(landmarks2.landmark[12].y*height2)
            x2,y2=int(landmarks2.landmark[16].x*width2),int(landmarks2.landmark[16].y*height2)
             
        if choice == "左の肘":
            angle = calculate_angle(np.array([int(landmarks1.landmark[11].x*width1),int(landmarks1.landmark[11].y*height1)]), 
                                    np.array([int(landmarks1.landmark[13].x*width1),int(landmarks1.landmark[13].y*height1)]), 
                                    np.array([int(landmarks1.landmark[15].x*width1),int(landmarks1.landmark[15].y*height1)]))
            
            AL=calculate_angle(np.array([int(landmarks1.landmark[13].x*width1),int(landmarks1.landmark[13].y*height1)]), 
                                    np.array([int(landmarks1.landmark[11].x*width1),int(landmarks1.landmark[11].y*height1)]), 
                                    np.array([int(landmarks1.landmark[15].x*width1),int(landmarks1.landmark[15].y*height1)]))

            BE=calculate_angle(np.array([int(landmarks1.landmark[13].x*width1),int(landmarks1.landmark[13].y*height1)]), 
                                    np.array([int(landmarks1.landmark[15].x*width1),int(landmarks1.landmark[15].y*height1)]), 
                                    np.array([int(landmarks1.landmark[11].x*width1),int(landmarks1.landmark[11].y*height1)]))
            
            x1,y1=int(landmarks2.landmark[11].x*width2),int(landmarks2.landmark[11].y*height2)
            x2,y2=int(landmarks2.landmark[15].x*width2),int(landmarks2.landmark[15].y*height2)
                  
        if choice == "右の膝":
            angle = calculate_angle(np.array([int(landmarks1.landmark[24].x*width1),int(landmarks1.landmark[24].y*height1)]), 
                                    np.array([int(landmarks1.landmark[26].x*width1),int(landmarks1.landmark[26].y*height1)]), 
                                    np.array([int(landmarks1.landmark[28].x*width1),int(landmarks1.landmark[28].y*height1)]))

            AL=calculate_angle(np.array([int(landmarks1.landmark[26].x*width1),int(landmarks1.landmark[26].y*height1)]), 
                                    np.array([int(landmarks1.landmark[24].x*width1),int(landmarks1.landmark[24].y*height1)]), 
                                    np.array([int(landmarks1.landmark[28].x*width1),int(landmarks1.landmark[28].y*height1)]))

            BE=calculate_angle(np.array([int(landmarks1.landmark[26].x*width1),int(landmarks1.landmark[26].y*height1)]), 
                                    np.array([int(landmarks1.landmark[28].x*width1),int(landmarks1.landmark[28].y*height1)]), 
                                    np.array([int(landmarks1.landmark[24].x*width1),int(landmarks1.landmark[24].y*height1)]))
            
            x1,y1=int(landmarks2.landmark[24].x*width2),int(landmarks2.landmark[24].y*height2)
            x2,y2=int(landmarks2.landmark[28].x*width2),int(landmarks2.landmark[28].y*height2)
            
        if choice == "左の膝":
            angle = calculate_angle(np.array([int(landmarks1.landmark[23].x*width1),int(landmarks1.landmark[23].y*height1)]), 
                                    np.array([int(landmarks1.landmark[25].x*width1),int(landmarks1.landmark[25].y*height1)]), 
                                    np.array([int(landmarks1.landmark[27].x*width1),int(landmarks1.landmark[27].y*height1)]))
            
            AL=calculate_angle(np.array([int(landmarks1.landmark[25].x*width1),int(landmarks1.landmark[25].y*height1)]), 
                                    np.array([int(landmarks1.landmark[23].x*width1),int(landmarks1.landmark[23].y*height1)]), 
                                    np.array([int(landmarks1.landmark[27].x*width1),int(landmarks1.landmark[27].y*height1)]))

            BE=calculate_angle(np.array([int(landmarks1.landmark[25].x*width1),int(landmarks1.landmark[25].y*height1)]), 
                                    np.array([int(landmarks1.landmark[27].x*width1),int(landmarks1.landmark[27].y*height1)]), 
                                    np.array([int(landmarks1.landmark[23].x*width1),int(landmarks1.landmark[23].y*height1)]))
            
            x1,y1=int(landmarks2.landmark[23].x*width2),int(landmarks2.landmark[23].y*height2)
            x2,y2=int(landmarks2.landmark[27].x*width2),int(landmarks2.landmark[27].y*height2)

    st.write(f"{choice}の角度: {angle:.2f}度")        

    alpha = math.radians(AL)  
    beta = math.radians(BE) 
    point_c1, point_c2 = find_point_c(x1, y1, x2, y2, alpha, beta)
    point_c1 = (int(point_c1[0]), int(point_c1[1]))
    point_c2 = (int(point_c2[0]), int(point_c2[1]))


    st.write(f"Cの座標:{point_c1}")
    cv2.line(processed_frame2,
            pt1=(x1, y1),
            pt2=point_c1,
            color=(255, 0, 0),
            thickness=3,
            lineType=cv2.LINE_4,
            shift=0)
    cv2.line(processed_frame2,
            pt1=(x2, y2),
            pt2=point_c1,
            color=(255, 0, 0),
            thickness=3,
            lineType=cv2.LINE_4,
            shift=0)

    st.image(processed_frame2, channels="BGR", caption=f"動画2 {frame_idx2}番目のフレーム")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)