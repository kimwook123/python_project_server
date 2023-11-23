import socket
import cv2
import numpy as np

RECV_TCP_IP = '10.10.21.117'
RECV_TCP_PORT = 25000

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 소켓 오픈
s.bind((RECV_TCP_IP, RECV_TCP_PORT))

s.listen()  # 연결 대기
print("클라이언트 연결요청 대기중...")
client_socket, addr = s.accept()
client_socket.setblocking(True)  # 연결 오류 방지

message = ""  # 결과치 저장할 전


def preprocess_image(image_path):
    # 이미지 불러오기
    image = cv2.imread(image_path)
    # BGR에서 그레이스케일로 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 가우시안 블러 적용(노이즈 감소)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # 그레이스케일 이미지의 적응형 스레시홀드 적용
    _, thresholded = cv2.threshold(blurred, 240, 255, cv2.THRESH_BINARY_INV)
    # 모폴로지 연산을 사용하여 그림자 제거
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel, iterations=2)
    # 이미지의 노이즈를 제거하기 위해 모폴로지 클로징 수행
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
    # 검은 배경 이미지 생성
    black_background = np.zeros_like(image)
    # 원본 이미지에서 검은 배경에 해당 영역을 복사
    black_background[closed == 255] = image[closed == 255]

    return black_background


def determine_shape(contour):
    # 윤곽의 길이 확인
    if len(contour) < 3:
        return 'undefined shape'

    # 도형 근사화
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # 도형의 꼭지점 수
    num_vertices = len(approx)

    # 각 도형에 대한 근사화된 꼭지점 수에 따라 도형 결정
    if num_vertices == 3:
        return 'triangle'
    elif num_vertices == 4:
        # 꼭지점이 4개일 때, 정사각형과 직사각형 구분
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h
        if 0.95 <= aspect_ratio <= 1.05:
            return 'foursquare'
        else:
            return 'rectangle'
    elif num_vertices == 5:
        return 'pentagon'
    elif num_vertices == 6:
        return 'hexagon'
    else:
        return 'circle'


def detect_shapes_and_colors(image_path):
    # 이미지 불러오기
    preprocessed_image = preprocess_image(image_path)

    # BGR에서 HSV로 변환
    hsvFrame = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2HSV)

    # 각 색상에 대한 범위 설정
    color_ranges = {
        'red': ([0, 100, 100], [10, 255, 255]),
        'orange': ([15, 100, 100], [25, 255, 255]),
        'yellow': ([25, 52, 72], [45, 255, 255]),
        'green': ([35, 52, 72], [80, 255, 255]),
        'blue': ([90, 80, 2], [120, 255, 255]),
        'pink': ([150, 40, 180], [170, 255, 255]),
        'purple': ([120, 40, 100], [150, 255, 255]),
        'brown': ([10, 100, 20], [20, 255, 200]),
        'sky_blue': ([80, 100, 100], [100, 255, 255]),
        # 'lavender': ([110, 38, 100], [130, 255, 255]),
        'gold': ([20, 100, 100], [30, 255, 255]),
        'silver': ([0, 0, 75], [180, 10, 150]),
        'cyan': ([85, 60, 60], [105, 255, 255]),
    }

    # 이미 처리된 영역을 저장할 배열
    processed_regions = np.zeros_like(preprocessed_image)

    # 전체 도형 카운트 변수
    total_shape_count = 0

    # 각 색상에 대한 영역 검출 및 표시
    for color, (lower, upper) in color_ranges.items():
        color_lower = np.array(lower, np.uint8)
        color_upper = np.array(upper, np.uint8)
        color_mask = cv2.inRange(hsvFrame, color_lower, color_upper)
        color_mask = cv2.dilate(color_mask, np.ones((5, 5), "uint8"))

        # 이미 처리된 영역을 마스킹
        color_mask = cv2.subtract(color_mask, processed_regions[:, :, 0])
        contours, hierarchy = cv2.findContours(color_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        max_area = 0  # 가장 큰 면적 초기화
        max_area_color = ''  # 가장 큰 면적을 가진 색상 초기화

        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)

            if area > 200:
                # 외곽선의 경계 확인
                x, y, w, h = cv2.boundingRect(contour)
                if x > 5 and y > 5 and x + w < preprocessed_image.shape[1] - 5 and y + h < preprocessed_image.shape[
                    0] - 5:
                    # 모양 추출
                    shape_name = determine_shape(contour)

                    # 중심 좌표 계산
                    M = cv2.moments(contour)
                    if M['m00'] != 0.0:
                        x = int(M['m10'] / M['m00'])
                        y = int(M['m01'] / M['m00'])

                        # 색상 면적이 가장 큰 경우 업데이트
                        if area > max_area:
                            max_area = area
                            max_area_color = color

                        total_shape_count += 1
                        print(
                            f'{total_shape_count}. 도형 정보 - 색상: {max_area_color.upper()}, '
                            f'모양: {shape_name}, 좌표: ({x}, {y})')
                    total_result = color.upper() + "," + shape_name + "," + str(x) + "," + str(y)  # 결과치 압축

                    return total_result

                    # # 이미지에 윤곽선 표시
                    # cv2.drawContours(imageFrame, [contour], 0, (0, 255, 0), 2)
                    #
                    # # 좌표값 텍스트 표시
                    # cv2.putText(imageFrame, f'({x}, {y})', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
                    #
                    # # 처리된 영역 업데이트
                    # cv2.drawContours(processed_regions, [contour], 0, (255, 255, 255), thickness=cv2.FILLED)


def recvAndLoad(accept_socket):
    try:
        read_image = accept_socket.recv(4096)  # 전송된 이미지 read
        if not read_image:
            print("이미지 전송 / 수신 오류")
            return

        image_size = int.from_bytes(read_image, byteorder='little')  # 이미지 형식 변환
        print(f"이미지 크기 : {image_size} 바이트")

        read_image = b""
        while len(read_image) < image_size:  # 이미지 크기가 클 경우, 분할해서 받고 합친다.
            chunk = accept_socket.recv(4096)
            if not chunk:
                print("이미지 전송 / 수신 오류")
                return
            read_image += chunk

        image_array = np.frombuffer(read_image, dtype=np.uint8)  # 이미지 numpy 배열로 변환
        image_origin = cv2.imdecode(image_array, cv2.IMREAD_COLOR)  # 이미지 읽을 수 있게끔 재변환

        file_path = "C:/Users/kimwook/PycharmProjects/pythonProject5/saved_image/image.jpg"  # 이미지 저장할 경로와 이름 설정

        cv2.imwrite(file_path, image_origin)  # 이미지 저장
        print(f'이미지 저장 성공. 경로 : {file_path}')

        finish_analyze = detect_shapes_and_colors(file_path)  # 이미지 분석 후 결과 저장

        send_result(client_socket, finish_analyze)  # 결과내역 서버로 전송

    except Exception as e:
        print(f"Error : {e}")

    finally:
        accept_socket.close()  # 한 번 발신이 끝났을 때 연결이 종료되게끔 하였는데, 이후 연결을 끊지 않을거면 수정해야함.
        print("연결 종료")


def send_result(send_socket, message):
    print(f"결과 : {message}")

    send_socket.send(message.encode('utf-8'))


if __name__ == '__main__':
    recvAndLoad(client_socket)
