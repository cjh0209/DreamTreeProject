import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def counting():
    # SSD 모델 설정
    model_file = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    config_file = "deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(config_file, model_file)

    # Dlib의 얼굴 랜드마크 모델 초기화
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # 얼굴에서 턱 아래쪽 랜드마크의 인덱스 범위
    jaw_landmarks_indices = list(range(6, 12))
    # 얼굴에서 코 랜드마크의 인덱스 범위
    nose_landmarks = list(range(27, 36))

    # 웹캠 연결
    cap = cv2.VideoCapture(0)

    # 프레임당 시간 초기화
    frame_time = 0

    # 비디오의 FPS(프레임 속도) 가져오기
    fps = 30  # 웹캠은 일반적으로 30fps로 가정

    # 윈도우 생성 및 초기 크기 설정
    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Frame", 800, 800)

    # 턱과 코의 평균 거리를 저장할 리스트
    average_distance_list = []
    # 얼굴의 가로 거리를 저장할 리스트
    horizontal_distance = []

    while cap.isOpened():
        # 웹캠에서 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임당 시간 계산
        frame_time += 1.0 / fps

        # 이미지의 높이와 너비 추출
        height, width = frame.shape[:2]

        # 이미지 전처리 (SSD 입력 형식에 맞게 변환)
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

        # SSD로 얼굴 검출
        net.setInput(blob)
        detections = net.forward()

        # 가장 큰 얼굴의 경계 상자 초기화
        largest_face_box = None
        largest_face_size = 0

        # 검출된 얼굴들을 순회하며 얼굴과 랜드마크 추출
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # 신뢰도가 일정 수준 이상인 얼굴들만 추출
            if confidence > 0.5:
                # 얼굴 영역의 경계 상자 좌표 계산
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (startX, startY, endX, endY) = box.astype("int")

                # 얼굴 크기 계산
                face_size = (endX - startX) * (endY - startY)

                # 현재 얼굴이 가장 큰 얼굴인지 확인
                if face_size > largest_face_size:
                    largest_face_box = (startX, startY, endX, endY)
                    largest_face_size = face_size

        # 가장 큰 얼굴만 처리
        if largest_face_box is not None:
            (startX, startY, endX, endY) = largest_face_box

            # 얼굴 영역 추출
            face = frame[startY:endY, startX:endX]

            # 얼굴 랜드마크 검출
            rect = dlib.rectangle(left=startX, top=startY, right=endX, bottom=endY)
            shape = predictor(frame, rect)

            # 얼굴 윤곽 추출
            face_points = np.array([(shape.part(idx).x, shape.part(idx).y) for idx in range(0, 68)])

            # 얼굴 윤곽의 점으로 표시
            for i in range(0, 68):
                x = shape.part(i).x
                y = shape.part(i).y
                cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)

            # 얼굴의 외곽 양쪽 점의 평균 거리 계산
            left_face_outline_point = face_points[0]  # index 1에 얼굴의 외곽 왼쪽의 위치
            right_face_outline_point = face_points[16]  # index 17에 오른쪽 눈가의 위치
            horizon_distance = np.linalg.norm(left_face_outline_point - right_face_outline_point)
            horizontal_distance.append(horizon_distance)

            # 코와 턱 간 평균 거리 계산
            nose_point = face_points[33]  # index 34에 해당되는 코의 위치
            jaw_points_indices = [6, 7, 8, 9, 10]  # index 7,8,9,10,11에 해당되는 턱의 위치
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

    # 웹캠 해제 및 윈도우 닫기
    cap.release
