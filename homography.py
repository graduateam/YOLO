import numpy as np
import cv2

# 이미지 좌표 (픽셀 좌표)
image_points = np.array([
    [335, 102],   
    [23, 251],    
    [584, 234],   
    [146, 404]    
], dtype=np.float32)

# 실제 세계 좌표 (위도, 경도)
# 위도(lat), 경도(lon)을 사용 (실제 세계 좌표)
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
    # 이미지 좌표를 homogeneous 좌표로 변환 (3x1 벡터)
    image_coords_homogeneous = np.array([image_coords[0], image_coords[1], 1]).reshape(3, 1)

    # Homography 행렬을 사용해 실세계 좌표로 변환
    world_coords_homogeneous = np.dot(homography_matrix, image_coords_homogeneous)

    # Homogeneous 좌표를 유클리드 좌표로 변환 (w 값으로 나눔)
    world_coords = world_coords_homogeneous / world_coords_homogeneous[2]

    # 결과를 반환 (위도, 경도)
    lat = world_coords[0][0]
    lon = world_coords[1][0]

    return lat, lon

# 테스트: 이미지 상의 좌표 -> 실세계 좌표로 변환
test_image_point = [516, 74]  # 변환할 이미지 좌표
lat, lon = convert_image_to_world(test_image_point, H)

print(f"이미지 좌표 {test_image_point}에 해당하는 실제 세계 좌표: 위도 {round(lat, 8)}, 경도 {round(lon, 8)}")