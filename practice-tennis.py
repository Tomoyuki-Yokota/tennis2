import streamlit as st
import cv2
import tempfile
import mediapipe as mp

st.title('テニスフォーム判別アプリ')
st.write('Play Tennis')

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

# 1つ目の動画
uploaded_file1 = st.file_uploader('1つ目の動画を選択してください...', type='mp4', key="uploader1")
if uploaded_file1 is not None:
    frames1 = read_video_frames(uploaded_file1)
    frame_idx1 = st.slider('動画1 フレームを選択', 1, len(frames1)-1, 1, key='slider_1')
    selected_frame1 = frames1[frame_idx1]
    processed_frame1 = process_frame_for_landmarks(selected_frame1)
    st.image(processed_frame1, channels="BGR", caption=f"動画1 {frame_idx1}番目のフレーム")

# 2つ目の動画
uploaded_file2 = st.file_uploader('2つ目の動画を選択してください...', type='mp4', key="uploader2")
if uploaded_file2 is not None:
    frames2 = read_video_frames(uploaded_file2)
    frame_idx2 = st.slider('動画2 フレームを選択', 1, len(frames2)-1, 1, key='slider_2')
    selected_frame2 = frames2[frame_idx2]
    processed_frame2 = process_frame_for_landmarks(selected_frame2)
    st.image(processed_frame2, channels="BGR", caption=f"動画2 {frame_idx2}番目のフレーム")

