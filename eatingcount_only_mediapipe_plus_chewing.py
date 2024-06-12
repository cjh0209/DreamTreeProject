import cv2
import mediapipe as mp
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
import torch
import pathlib

# YOLO 설정
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath  # 경로 설정을 위해 임시로 PosixPath 클래스 변경
#model_path = './fork_soon_notbad.pt'  # YOLO 모델 경로
#model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)  # YOLO 모델 로드

# MediaPipe 초기화
mp_face_mesh = mp.solutions.face_mesh  # 얼굴 메쉬 모델
mp_hands = mp.solutions.hands  # 손 모델
mp_face_detection = mp.solutions.face_detection  # 얼굴 검출 모델
mp_drawing = mp.solutions.drawing_utils  # 그리기 유틸리티

# 동영상 파일 경로
video_path = 'ymin2t.mp4'
cap = cv2.VideoCapture(0)  # 동영상 캡처 객체 생성

# 비디오의 FPS 가져오기
fps = cap.get(cv2.CAP_PROP_FPS)

# 상태 변수 및 설정값 초기화
NO_HAND = 0  # 손을 인식하지 않은 상태
HAND_DETECTED = 1  # 손을 인식한 상태
HAND_NEAR_MOUTH = 2  # 손이 입 근처에 있는 상태
HAND_MOVING_AWAY = 3  # 손이 멀어지는 상태
state = NO_HAND  # 초기 상태: 손을 인식하지 않은 상태

distance_threshold = 250  # 손과 입 사이의 거리 임계값
hand_positions = []  # 손 위치 저장 리스트
spoon_fork_positions = []  # 숟가락/포크 위치 저장 리스트
eating_counter = 0  # 식사 횟수 카운트
distances = []  # 손과 얼굴 사이의 거리 저장 리스트
face_width = None
first_face_width = None
average_distance_list = []  # 턱과 코의 평균 거리를 저장할 리스트
horizontal_distance = []  # 얼굴의 가로 거리를 저장할 리스트


# 관자놀이 지점 간의 유클리드 거리 계산(얼굴 가로거리 계산)
def calculate_horizontal_distance(face_landmarks, frame_shape):
    ih, iw, _ = frame_shape
    leftmost_x = int(face_landmarks.landmark[234].x * iw)
    leftmost_y = int(face_landmarks.landmark[234].y * ih)
    rightmost_x = int(face_landmarks.landmark[454].x * iw)
    rightmost_y = int(face_landmarks.landmark[454].y * ih)
    horizontal_distance = np.linalg.norm(np.array([leftmost_x, leftmost_y]) - np.array([rightmost_x, rightmost_y]))
    return horizontal_distance


# 정규화된 임계값을 계산하는 함수(객체와 입술거리, 손과 입술거리 측정을 위함)
def calculate_normalized_threshold(face_width, first_face_width, distance_threshold):
    normalized_threshold = (face_width / first_face_width) * distance_threshold
    return normalized_threshold


# 코와 턱의 유클리드 거리 계산 함수
def calculate_nose_to_jaw_distance(face_landmarks):
    nose_point = face_landmarks.landmark[4]
    jaw_points_indices = [176, 148, 152, 377, 400]
    jaw_distances = []

    for jaw_index in jaw_points_indices:
        jaw_point = face_landmarks.landmark[jaw_index]
        distance = np.linalg.norm(np.array([nose_point.x, nose_point.y]) - np.array([jaw_point.x, jaw_point.y]))
        jaw_distances.append(distance)
    average_distance = np.mean(jaw_distances)
    return average_distance


# 얼굴 메쉬와 손 인식 모델 사용
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh, \
        mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("비디오 처리 완료.")
            break

        # 프레임당 시간 계산
        frame_time = 1.0 / fps

        # YOLO로 객체 검출, 바운딩 박스 그리기
        #yolo_result = model(frame)
        #yolo_detect = yolo_result.xyxyn[0].numpy()
        yolo_x = frame.shape[1]
        yolo_y = frame.shape[0]
        location_box = None
        object_detected = False

        #for i in range(len(yolo_detect)):
        #    yolo_x1 = int(yolo_detect[i, 0] * yolo_x)
        #    yolo_y1 = int(yolo_detect[i, 1] * yolo_y)
        #    yolo_x2 = int(yolo_detect[i, 2] * yolo_x)
        #    yolo_y2 = int(yolo_detect[i, 3] * yolo_y)

        #    cv2.rectangle(frame, (yolo_x1, yolo_y1), (yolo_x2, yolo_y2), (0, 0, 255), 2)
        #    yolo_center_x = int((yolo_x1 + yolo_x2) / 2)
        #    yolo_center_y = int((yolo_y1 + yolo_y2) / 2)
        #    cv2.circle(frame, (yolo_center_x, yolo_center_y), 3, (0, 0, 255), cv2.FILLED)
        #    location_box = (yolo_center_x, yolo_center_y)
        #    object_detected = True
        #    cv2.circle(frame, tuple(location_box), 10, (255, 0, 0), cv2.FILLED)

        # MediaPipe로 얼굴과 손 인식
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_mesh.process(frame_rgb)
        lip_center = None

        # 얼굴 랜드마크가 검출된 경우
        if face_results.multi_face_landmarks:
            # 가장 큰 얼굴 선택
            for face_landmarks in face_results.multi_face_landmarks:
                ih, iw, _ = frame.shape

                # 입술 중심 좌표 계산
                lip_coords = [(int(face_landmarks.landmark[i].x * iw), int(face_landmarks.landmark[i].y * ih)) for i in
                              [13, 14, 312, 317]]
                if lip_coords:
                    lip_center = np.mean(lip_coords, axis=0).astype(int)
                    cv2.circle(frame, tuple(lip_center), 5, (0, 0, 255), cv2.FILLED)

                # 얼굴 폭 계산 및 정규화된 임계값 계산
                horizon_distance = calculate_horizontal_distance(face_landmarks, frame.shape)
                # 가로 거리 리스트에 추가
                horizontal_distance.append(horizon_distance)
                normalized_threshold = calculate_normalized_threshold(horizon_distance, horizontal_distance, distance_threshold) #현재 가로길이


            # 손 랜드마크가 검출된 경우
        hand_results = hands.process(frame_rgb)
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                lm_list = []
                for id, lm in enumerate(hand_landmarks.landmark):
                    if id in [4, 8, 12, 16, 20]:  # 주요 손가락 끝점들
                        h, w, _ = frame.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lm_list.append((cx, cy))
                        cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

                if lm_list:
                    avg_x = int(np.mean([pt[0] for pt in lm_list]))
                    avg_y = int(np.mean([pt[1] for pt in lm_list]))
                    hand_center = (avg_x, avg_y)
                    cv2.circle(frame, hand_center, 10, (0, 255, 255), cv2.FILLED)
                    cv2.putText(frame, 'Hand Center', (hand_center[0] - 20, hand_center[1] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                    # 손과 입술 중심 간 거리 계산 및 상태 업데이트
                    if lip_center is not None:
                        distance_to_hand = np.linalg.norm(np.array(hand_center) - np.array(lip_center))
                        distances.append(distance_to_hand)

                        if distance_to_hand < distance_threshold and state == NO_HAND:
                            state = HAND_DETECTED
                        elif distance_to_hand < distance_threshold and state == HAND_DETECTED:
                            state = HAND_NEAR_MOUTH
                        elif distance_to_hand > distance_threshold and state == HAND_NEAR_MOUTH:
                            state = HAND_MOVING_AWAY
                            eating_counter += 1
                            state = NO_HAND

        # 코와 턱의 유클리드 거리 계산(eating_count가 1이상일 경우 실행)
        if eating_counter >= 1:
            average_distance = calculate_nose_to_jaw_distance(face_landmarks)
            average_distance_list.append(average_distance)

        cv2.putText(frame, f'Eating Count: {eating_counter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2)

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

print(average_distance_list)



# 그래프 데이터 초기화
normalized_distances = []

# 첫 번째 프레임의 가로 거리
previous_horizontal_distance = horizontal_distance[0]

# 첫 번째 프레임에서는 가로 거리가 0이므로 그냥 0으로 초기화
normalized_distances.append(0)

# 각 프레임에서의 평균 거리를 해당 프레임의 가로 거리로 정규화하여 리스트에 추가
for i in range(1, len(average_distance_list)):
    current_distance = average_distance_list[i]
    current_horizontal_distance = horizontal_distance[i]
    normalized_distance = current_distance * (current_horizontal_distance / previous_horizontal_distance)
    normalized_distances.append(normalized_distance)
    previous_horizontal_distance = current_horizontal_distance
normalized_distances[0]=normalized_distances[1]


# 부드럽게 만든 곡선 그래프 데이터, Savitzky-Golay 필터 적용
smoothed_curve = savgol_filter(normalized_distances, window_length=21, polyorder=3)
# 부드럽게 만든 곡선에 Savitzky-Golay 필터를 적용하여 미분 수행
derivative = np.gradient(smoothed_curve)

# 프레임 제한을 0.2s이라 가정
k = 1  # 누적 프레임 초기화
limit_frame = 0.2  # 제한 시간 설정

while frame_time < limit_frame:
    frame_time += frame_time  # 누적 시간 계산
    k +=1

print("Accumulated time:", k)

# 기울기가 양수에서 음수로 바뀌는 꼭짓점 찾기
peaks = []
i = 0
threshold = 0.6  # 비율 임계값 설정
while i < len(derivative) - 1:
    if derivative[i - 1] > 0 and derivative[i] < 0 and sum(1 for d in derivative[i-k:i] if d >= 0) / k >= threshold and sum(1 for d in derivative[i:i+k] if d <= 0) / k >= threshold:
        # peak인 지점 and i-k까지 양수인 비율이 threshold 이상 and  i-k까지 음수인 비율이 threshold 이상
        peaks.append(i)
        # 현재 꼭짓점 일정 범위 이후 다음 피크를 찾도록 함 (중복 방지)
        i +=k*2  # 일정 범위를 k*2로 설정
    i += 1

# 씹는 동작 횟수 계산
chewing_count = len(peaks)

# 씹는 동작 횟수 출력
print("Chewing count:", chewing_count)

# 부드럽게 만들어진 그래프 데이터로 그래프 그리기
plt.plot(smoothed_curve, label='Smoothed Curve')
plt.xlabel('Frame')
plt.ylabel('Normalized Distance')
plt.title('chewing_output_data10')
plt.legend()

# 꼭짓점 지점을 그래프에 표시
plt.plot(peaks, [smoothed_curve[i] for i in peaks], "x", color='red')
plt.show()
