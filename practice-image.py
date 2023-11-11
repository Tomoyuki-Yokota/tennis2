import streamlit as st
import cv2
import tempfile
import mediapipe as mp

st.title('テニスフォーム判別アプリ')
st.write('Play Tennis')

def process_uploaded_video(uploaded_file, key_prefix):
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    mesh_drawing_spec = mp_drawing.DrawingSpec(thickness=2, color=(0,255,0))
    mark_drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2, color=(0,0,255))

    cap_file = cv2.VideoCapture(tfile.name)
    frames = []

    with mp_holistic.Holistic(min_detection_confidence=0.5, static_image_mode=False) as holistic_detection:
        while cap_file.isOpened():
            success, image = cap_file.read()
            if not success:
                break
            image = cv2.resize(image, dsize=None, fx=0.3, fy=0.3)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic_detection.process(rgb_image)
            
            mp_drawing.draw_landmarks(image=image,landmark_list=results.face_landmarks,
connections=mp_holistic.FACEMESH_TESSELATION,landmark_drawing_spec=None,
connection_drawing_spec=mesh_drawing_spec)
            mp_drawing.draw_landmarks(image=image,landmark_list=results.pose_landmarks,
    connections=mp_holistic.POSE_CONNECTIONS,landmark_drawing_spec=mark_drawing_spec,
    connection_drawing_spec=mesh_drawing_spec)
            mp_drawing.draw_landmarks(image=image,landmark_list=results.left_hand_landmarks,
    connections=mp_holistic.HAND_CONNECTIONS, landmark_drawing_spec=mark_drawing_spec,connection_drawing_spec=mesh_drawing_spec)
            mp_drawing.draw_landmarks(image=image,landmark_list=results.right_hand_landmarks,
    connections=mp_holistic.HAND_CONNECTIONS,landmark_drawing_spec=mark_drawing_spec,connection_drawing_spec=mesh_drawing_spec)


            frames.append(image)

    cap_file.release()

    # フレームを画像として表示し、ユーザーが選択できるようにする
    frame_idx = st.slider(f'{key_prefix} フレームを選択', 1, len(frames)-1, 1, key=f'slider_{key_prefix}')
    st.image(frames[frame_idx], channels="BGR", caption=f"{frame_idx}番目のフレーム")

# 1つ目の動画
uploaded_file1 = st.file_uploader('1つ目の動画を選択してください...', type='mp4', key="uploader1")
if uploaded_file1 is not None:
    process_uploaded_video(uploaded_file1, "動画1")

# 2つ目の動画
uploaded_file2 = st.file_uploader('2つ目の動画を選択してください...', type='mp4', key="uploader2")
if uploaded_file2 is not None:
    process_uploaded_video(uploaded_file2, "動画2")

