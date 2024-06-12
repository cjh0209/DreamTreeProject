import matplotlib.font_manager

# 시스템에 설치된 폰트 목록 가져오기
font_list = matplotlib.font_manager.findSystemFonts()

# 나눔고딕 폰트 확인
for font_path in font_list:
    if 'NanumGothic' in font_path:
        print("나눔고딕 폰트 경로:", font_path)