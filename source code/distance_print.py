import cv2
import dlib
import numpy as np

# Viola-Jones 얼굴 검출기 생성
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# dlib 얼굴 랜드마크 검출기 생성
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# 동영상 로드
video_capture = cv2.VideoCapture('test1.mp4')

# 동영상이 정상적으로 열렸는지 확인
if not video_capture.isOpened():
    print("동영상 파일을 열 수 없습니다. 파일 경로를 확인하세요.")
    exit()

while True:
    # 프레임 읽기
    ret, frame = video_capture.read()

    # 프레임 읽기 실패시 루프 종료
    if not ret:
        break

    # 그레이스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴 검출
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 얼굴 및 턱선 시각화
    for (x, y, w, h) in faces:
        # Viola-Jones로 검출된 얼굴 바운딩 박스 그리기
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 바운딩 박스의 왼쪽 상단 모서리에 점 추가
        left_top_corner = (x, y)
        cv2.circle(frame, left_top_corner, 3, (0, 255, 0), -1)

        # dlib로 얼굴 검출
        dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        landmarks = predictor(gray, dlib_rect)

        # 턱선 좌표 계산
        jawline_points = []
        for i in range(0, 17):
            x_lm = landmarks.part(i).x
            y_lm = landmarks.part(i).y
            jawline_points.append((x_lm, y_lm))

        # 턱선 점들 중 11개 선택
        selected_jawline_points = []
        indices = np.linspace(0, len(jawline_points) - 1, 11, dtype=int)
        for idx in indices:
            selected_jawline_points.append(jawline_points[idx])

        # 선택된 턱선 점들 시각화
        for point in selected_jawline_points:
            cv2.circle(frame, tuple(point), 5, (0, 255, 0), -1)

        # 왼쪽 상단 모서리를 기준으로 턱선 점들과의 유클리드 거리 계산
        distances = [np.linalg.norm(np.array(left_top_corner) - np.array(point)) for point in selected_jawline_points]
        # 평균 거리 계산
        avg_distance = np.mean(distances)
        # 평균 거리 출력
        cv2.putText(frame, f'Average distance: {avg_distance:.2f}', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 결과 표시
    cv2.imshow('Video', frame)

    # 'q'를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 리소스 해제
video_capture.release()
cv2.destroyAllWindows()
