import cv2

# 클릭 이벤트 핸들러 함수
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # 마우스 왼쪽 버튼 클릭 시
        print(f"Clicked at: ({x}, {y})")
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)  # 클릭한 위치에 점 표시
        cv2.imshow("Image", img)

# 이미지 불러오기
img = cv2.imread("0306.jpg")
cv2.imshow("Image", img)

# 마우스 클릭 이벤트 설정
cv2.setMouseCallback("Image", click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()