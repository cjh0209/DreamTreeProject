import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# MediaPipe 초기화
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

# 동영상 파일 경로
video_path = "nonmoont1.mp4"

# 비디오 파일 읽기
cap = cv2.VideoCapture(video_path)

# 비디오의 FPS(프레임 속도) 가져오기
fps = cap.get(cv2.CAP_PROP_FPS)

# 윈도우 생성 및 초기 크기 설정
cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Frame", 800, 800)

# 턱과 코의 평균 거리를 저장할 리스트
average_distance_list = []
# 얼굴의 가로 거리를 저장할 리스트
horizontal_distance = []

# MediaPipe face detection과 face mesh 초기화
with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        while cap.isOpened():
            # 비디오에서 프레임 읽기
            ret, frame = cap.read()
            if not ret:
                break

            # 프레임당 시간 계산
            frame_time = 1.0 / fps

            # 이미지를 RGB로 변환
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 얼굴 검출
            results = face_detection.process(frame_rgb)

            if results.detections:
                for detection in results.detections:
                    # 얼굴 영역 추출
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    (startX, startY, endX, endY) = (int(bboxC.xmin * iw), int(bboxC.ymin * ih),
                                                    int((bboxC.xmin + bboxC.width) * iw), int((bboxC.ymin + bboxC.height) * ih))

                    # 얼굴 랜드마크 검출
                    face_results = face_mesh.process(frame_rgb)
                    if face_results.multi_face_landmarks:
                        for face_landmarks in face_results.multi_face_landmarks:
                            face_points = np.array([(int(landmark.x * iw), int(landmark.y * ih)) for landmark in face_landmarks.landmark])

                            # 얼굴 윤곽의 점으로 표시
                            for idx, point in enumerate(face_points):
                                cv2.circle(frame, (point[0], point[1]), 1, (0, 255, 255), -1)

                            # 얼굴의 외곽 양쪽 점의 평균 거리 계산
                            left_face_outline_point = face_points[234]  # 얼굴 외곽 왼쪽 위치
                            right_face_outline_point = face_points[454]  # 얼굴 외곽 오른쪽 위치
                            horizon_distance = np.linalg.norm(left_face_outline_point - right_face_outline_point)
                            horizontal_distance.append(horizon_distance)

                            # 코와 턱 간 평균 거리 계산
                            nose_point = face_points[1]  # 코의 위치
                            jaw_points_indices = [152, 377, 400, 148, 178]  # 턱의 위치
                            jaw_distances = []

                            for jaw_index in jaw_points_indices:
                                jaw_point = face_points[jaw_index]
                                distance = np.linalg.norm(nose_point - jaw_point)  # 유클리드 거리 계산
                                jaw_distances.append(distance)

                            # 평균 계산
                            average_distance = np.mean(jaw_distances)
                            average_distance_list.append(average_distance)

            # 프레임 출력
            cv2.imshow("Frame", frame)

            # 종료 키 처리
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# 동영상 파일 닫기
cap.release()
cv2.destroyAllWindows()

# print로 확인
print(horizontal_distance)
print(average_distance_list)
print("Length of forehead_distance list:", len(horizontal_distance))
print("Length of average_distance list:", len(average_distance_list))
print(frame_time)

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
normalized_distances[0] = normalized_distances[1]

# 부드럽게 만든 곡선 그래프 데이터, Savitzky-Golay 필터 적용
smoothed_curve = savgol_filter(normalized_distances, window_length=21, polyorder=3)
# 부드럽게 만든 곡선에 Savitzky-Golay 필터를 적용하여 미분 수행
derivative = np.gradient(smoothed_curve)

# 프레임 제한을 0.2s이라 가정
k = 1  # 누적 프레임 초기화
limit_frame = 0.2  # 제한 시간 설정

while frame_time < limit_frame:
    frame_time += frame_time  # 누적 시간 계산
    k += 1

print("Accumulated time:", k)

# 기울기가 양수에서 음수로 바뀌는 꼭짓점 찾기
peaks = []
i = 0
threshold = 0.6  # 비율 임계값 설정
while i < len(derivative) - 1:
    if derivative[i - 1] > 0 and derivative[i] < 0 and sum(1 for d in derivative[i - k:i] if d >= 0) / k >= threshold and sum(1 for d in derivative[i:i + k] if d <= 0) / k >= threshold:
        # peak인 지점 and i-k까지 양수인 비율이 threshold 이상 and  i-k까지 음수인 비율이 threshold 이상
        peaks.append(i)
        # 현재 꼭짓점 일정 범위 이후 다음 피크를 찾도록 함 (중복 방지)
        i += k * 2  # 일정 범위를 k*2로 설정
    i += 1

# 씹는 동작 횟수 계산
chewing_count = len(peaks)

# 씹는 동작 횟수 출력
print("Chewing count:", chewing_count)

# 부드럽게 만들어진 그래프 데이터로 그래프 그리기
plt.plot(smoothed_curve, label='Smoothed Curve')
plt.xlabel('Frame')
plt.ylabel('Normalized Distance')
plt.title('Normalized Distance Over Frames')
plt.legend()

# 꼭짓점 지점을 그래프에 표시
plt.plot(peaks, [smoothed_curve[i] for i in peaks], "x", color='red')
plt.show()
