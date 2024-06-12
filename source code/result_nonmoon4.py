
import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Viola-Jones 얼굴 검출기 생성
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# dlib 얼굴 랜드마크 검출기 생성
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# 동영상 로드
video_capture = cv2.VideoCapture('ymint4.mp4')

# 동영상이 정상적으로 열렸는지 확인
if not video_capture.isOpened():
    print("동영상 파일을 열 수 없습니다. 파일 경로를 확인하세요.")
    exit()

# 평균 거리를 저장할 리스트
avg_distances = []

# 창 이름 설정 및 크기 조정
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Video', 600, 800)  # 창 크기를 600x800으로 조정

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

    for (x, y, w, h) in faces:
        # 얼굴 영역에 바운딩 박스 그리기
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # dlib로 얼굴 랜드마크 검출
        dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        landmarks = predictor(gray, dlib_rect)

        # 바운딩 박스의 왼쪽 상단 모서리에 점 추가
        left_top_corner = (x, y)
        cv2.circle(frame, left_top_corner, 3, (0, 255, 0), -1)

        # 턱선 좌표 계산 (3번부터 13번까지)
        jawline_points = []
        for i in range(3, 14):
            x_lm = landmarks.part(i).x
            y_lm = landmarks.part(i).y
            jawline_points.append((x_lm, y_lm))

        # 왼쪽 상단 모서리를 기준으로 턱선 점들과의 유클리드 거리 계산
        distances = [np.linalg.norm(np.array(left_top_corner) - np.array(point)) for point in jawline_points]
        # 평균 거리 계산
        avg_distance = np.mean(distances)

        # 평균 거리를 리스트에 추가
        avg_distances.append(avg_distance)

        # 선택된 턱선 점들 시각화
        for point in jawline_points:
            cv2.circle(frame, tuple(point), 5, (0, 255, 0), -1)

    # 결과 표시
    cv2.imshow('Video', frame)

    # 'q'를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 리소스 해제
video_capture.release()
cv2.destroyAllWindows()

# LPF 필터 적용
class LowPassFilter:
    def __init__(self, cutoff_freq, ts):
        self.ts = ts
        self.cutoff_freq = cutoff_freq
        self.pre_out = 0.
        self.tau = self.calc_filter_coef()

    def calc_filter_coef(self):
        w_cut = 2 * np.pi * self.cutoff_freq
        return 1 / w_cut

    def filter(self, data):
        out = (self.tau * self.pre_out + self.ts * data) / (self.tau + self.ts)
        self.pre_out = out
        return out

# DWD 필터 적용
def dwd_filter(data, window_size=5):
    dwd = np.zeros_like(data)
    for i in range(len(data)):
        start = max(0, i - window_size)
        end = min(len(data), i + window_size)
        dwd[i] = np.mean(np.abs(np.diff(data[start:end])))
    return dwd

# LPF 필터 적용한 결과
lpf = LowPassFilter(cutoff_freq=1.0, ts=0.3)
lpf_result = [lpf.filter(data) for data in avg_distances]

dwd_result = dwd_filter(lpf_result, window_size=5)

# 평균 거리에서 피크 감지
peaks, _ = find_peaks(dwd_result, distance=20)  # 적절한 최소 거리 지정

# 평균 피크 높이 계산
mean_peak_height = np.mean(dwd_result[peaks])

# 평균 피크 높이의 절반 이상인 피크 선택
selected_peaks = [peak for peak in peaks if dwd_result[peak] >= mean_peak_height / 2]


font_path = r'C:\Windows\Fonts\NanumGothic.ttf'

# 나눔고딕 폰트를 사용하도록 설정
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['font.sans-serif'] = ['NanumGothic']

# 평균 거리 그래프 그리기
plt.figure(figsize=(10, 5))
plt.plot(dwd_result, label='chewing flow', color='blue')
plt.plot(selected_peaks, np.array(dwd_result)[selected_peaks], 'o', color='red', label='peak')
plt.legend()

plt.xlabel('프레임 번호')
plt.ylabel('평균 거리')
plt.title('논문 구현 결과')
#plt.ylim(0, 10)
plt.grid(True)
plt.show()


