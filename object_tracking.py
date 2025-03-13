import time
import cv2
import torch
import numpy as np
from ultralytics import YOLO


# 이미지 좌표 (픽셀 좌표)
image_points = np.array([
    [335, 102],   
    [23, 251],    
    [584, 234],   
    [146, 404]    
], dtype=np.float32)

# 실제 세계 좌표 (위도, 경도)
world_points = np.array([
    [37.67675942, 126.74583666],  
    [37.67696082, 126.74597894],  
    [37.67687015, 126.74558537],  
    [37.67703350, 126.74581464]   
], dtype=np.float32)

# Homography 행렬 계산
H, status = cv2.findHomography(image_points, world_points)

# 이미지 좌표를 실세계 좌표로 변환하는 함수
def convert_image_to_world(image_coords, homography_matrix):
    image_coords_homogeneous = np.array([image_coords[0], image_coords[1], 1]).reshape(3, 1)
    world_coords_homogeneous = np.dot(homography_matrix, image_coords_homogeneous)
    world_coords = world_coords_homogeneous / world_coords_homogeneous[2]
    lat = world_coords[0][0]
    lon = world_coords[1][0]
    return lat, lon



# YOLOv8 모델 로드 (GPU 사용 설정)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('yolo_version/yolov8n.pt').to(device)

# COCO 데이터셋에서 사람과 차량 관련 클래스 ID
TARGET_CLASSES = [0, 2, 3, 5]  # 사람(0), 자동차(2), 오토바이(3), 버스(5)

# 영상 파일 경로
video_path = 'video/ilsan.mp4'
output_path = f"{video_path.rsplit('.', 1)[0]}_output.mp4"

# 영상 파일 열기
cap = cv2.VideoCapture(video_path)

# 원본 영상의 프레임, 해상도 가져오기
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# VideoWriter 객체 생성
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# 전체 실행 시작 시간
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 객체 감지 수행 (추적 모드)
    results = model.track(frame, classes=TARGET_CLASSES, persist=True)

    # 감지된 객체의 좌표 추적
    for obj in results[0].boxes:
        obj_id = int(obj.id) if obj.id is not None else -1
        bbox = obj.xyxy.cpu().numpy()[0]

        # 바운딩 박스의 중앙 좌표 계산
        center_x = int((bbox[0] + bbox[2]) / 2)
        center_y = int((bbox[1] + bbox[3]) / 2)
        
        lat, lon = convert_image_to_world([center_x, center_y], H)
        print(f"객체 ID: {obj_id}, 좌표: ({lat}, {lon})")
        
    # 감지된 객체를 프레임에 그리기
    frame = results[0].plot()
    
    out.write(frame)
    
    cv2.imshow('YOLOv8 Video Detection', frame)

    # 'q' 입력 시 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 전체 실행 종료 시간
end_time = time.time()

cap.release()
out.release()
cv2.destroyAllWindows()

# 총 소요 시간 출력
total_time = end_time - start_time
print(f"\n🔍 실행 디바이스: {device.upper()}")
print(f"총 소요 시간: {total_time:.2f}초")