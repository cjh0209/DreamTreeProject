import os
import cv2
import dlib

# Haar-cascade 분류기의 경로 가져오기
cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
haar_model_face = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')

# 검출기로드
face_detector = cv2.CascadeClassifier(haar_model_face)

# 동영상로드(동영상 파일로 교체하십시오 'test1.mp4')
video_capture = cv2.VideoCapture('test1.mp4')

# 비디오 캡처 객체가 정상적으로 생성되었는지 확인
if not video_capture.isOpened():
    print("동영상 파일을 열 수 없습니다. 파일 경로를 확인하세요.")
    exit()

while True:
    # 동영상에서 프레임 읽기
    ret, frame = video_capture.read()

    # 프레임 읽기 실패시 루프 종료
    if not ret:
        break

    # 프레임을 그레이 스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴 감지
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2, minSize=(50, 50),
                                           flags=cv2.CASCADE_SCALE_IMAGE)

    # 경계 상자 그리기
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 프레임 표시
    cv2.imshow('동영상', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 작업 완료 후 리소스 해제
video_capture.release()
cv2.destroyAllWindows()
