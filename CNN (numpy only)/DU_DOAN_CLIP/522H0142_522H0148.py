import cv2
import numpy as np
import os

#Clean anh 
def clean_images():
    
    file_list=os.listdir('./')
    for file_name in file_list:
        if '.png' in file_name:
            os.remove(file_name)

#Ham xu ly anh
def preprocess_image(image):
    
    #Chuyen anh sang he mau HSV
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

    #Mau do (red color)
    lower_red1=np.array([0,150,50])
    upper_red1=np.array([10,255,150])
    lower_red2=np.array([150,100,20])
    upper_red2=np.array([180,255,150])
    mask_red1=cv2.inRange(hsv,lower_red1,upper_red1)
    mask_red2=cv2.inRange(hsv,lower_red2,upper_red2)
    mask_red=cv2.bitwise_or(mask_red1,mask_red2)

    #Mau xanh da troi (blue color)
    lower_blue=np.array([105,100,120])
    upper_blue=np.array([110,255,255])
    mask_blue=cv2.inRange(hsv,lower_blue,upper_blue)

    #Lam min va xu ly mat na
    mask_red=cv2.GaussianBlur(mask_red,(5,5),0)
    mask_red=cv2.erode(mask_red,None,iterations=2)
    mask_red=cv2.dilate(mask_red,None,iterations=2)

    mask_blue=cv2.GaussianBlur(mask_blue,(5,5),0)
    mask_blue=cv2.erode(mask_blue,None,iterations=2)
    mask_blue=cv2.dilate(mask_blue,None,iterations=2)

    return mask_red,mask_blue

def detect_circle_red(contour):
    
    area=cv2.contourArea(contour)
    
    #Loai tru cac hinh tron nho, tranh phat hien sai
    if area<500:
        return False

    perimeter=cv2.arcLength(contour,True)
    
    if perimeter == 0:
        return False

    x,y,w,h=cv2.boundingRect(contour)
    aspect_ratio=float(w)/h
    circularity=(4*np.pi*area)/(perimeter**2)
    
    #Danh sach dieu kien
    dks=[
        {
            "circularity_range": (0.65,1),
            "aspect_ratio_range": (0.75,1.2),
            "height_range": (37,120),
            "perimeter_min": 175
        },
        {
            "circularity_range": (0.23,0.24),
            "aspect_ratio_range": (0.75,1.2),
            "height_range": (37,120),
            "area_range": (595,700),
            "perimeter_range": (165,180)
        },
        {
            "circularity_range": (0.12,0.13),
            "area_range": (1590,1658),
            "perimeter_range": (390,420),
        }
    ]

    #kemtra
    for dk in dks:
        if (dk.get("circularity_range",(0,1))[0]<=circularity<=dk.get("circularity_range",(0,1))[1] and
            dk.get("aspect_ratio_range",(0,float("inf")))[0]<=aspect_ratio<=dk.get("aspect_ratio_range",(0,float("inf")))[1] and
            dk.get("height_range",(0,float("inf")))[0]<h<dk.get("height_range",(0,float("inf")))[1] and
            perimeter>=dk.get("perimeter_min",0) and
            dk.get("area_range",(0,float("inf")))[0]<=area<=dk.get("area_range",(0,float("inf")))[1] and
            dk.get("perimeter_range",(0,float("inf")))[0]<=perimeter<=dk.get("perimeter_range",(0,float("inf")))[1]):
            return True

    return False



#Ham phat hien hinh tron mau xanh
def detect_circle_blue(contour):
    area=cv2.contourArea(contour)
    if area<2300:
        return False

    perimeter=cv2.arcLength(contour,True)
    if perimeter == 0:
        return False

    x,y,w,h=cv2.boundingRect(contour)
    aspect_ratio=float(w)/h
    circularity=(4*np.pi*area)/(perimeter**2)

    #Dieu kien cho cac loai hinh tron khac nhau
    small_circle=0.67<=circularity<=1 and 0.9<=aspect_ratio<=1.2 and 37<h<150
    medium_circle=0.36<=circularity<0.67 and 0.9<=aspect_ratio<=1.2 and 37<h<150 and area>8500 and perimeter>500
    large_circle=0.25<=circularity<0.36 and 0.9<=aspect_ratio<=1.2 and 37<h<150 and area>14500 and perimeter>700

    #Dieu kien loai tru
    exclusion_area_perimeter=area<2500 and perimeter<210

    #Kiem tra cac dieu kien
    if exclusion_area_perimeter:
        return False
    if small_circle or medium_circle or large_circle:
        return True

    return False



def detect_triangle_red(contour):
    
    area=cv2.contourArea(contour)
    
    if area<2300:
        return False

    perimeter=cv2.arcLength(contour,True)
    approx=cv2.approxPolyDP(contour,0.04*perimeter,True)

    if len(approx) == 3:  #Hinh tam giac co 3 canh
        
        x,y,w,h=cv2.boundingRect(approx)
        aspect_ratio=float(w)/h

        #Dieu kien loai tru cho hinh tam giac nho
        if area<1400 and perimeter<150:
            return False

        #Dieu kien cho hinh tam giac hop le
        valid_triangle=0.9<aspect_ratio<1 and 30<w<150 and 30<h<150
        
        if valid_triangle:
            return True

    return False


def detect_rectangle_blue(contour):
    
    area=cv2.contourArea(contour)
    
    if area<1700:
        return False

    perimeter=cv2.arcLength(contour,True)
    approx=cv2.approxPolyDP(contour,0.02*perimeter,True)

    if len(approx)!=4:
        return False

    x,y,w,h=cv2.boundingRect(approx)
    aspect_ratio=float(w)/h

    #Cac dieu kien kich thuoc cho hinh chu nhat mong muon
    large_rectangle=w<150 and area>19000
    medium_rectangle=44<w<90 and 32<h<60 and 1200<area<6000 and 140<perimeter<400
    unwanted_rectangle=95<w<153 and 50<h<86 and perimeter<460 and area<8300

    #Loai tru cac truong hop chu vi va dien tich khong phu hop
    high_perimeter_exclusion=perimeter>700 and area<10000
    low_area_exclusion=perimeter>100 and area<900

    #Cac dieu kien ty le khung hinh va kich thuoc
    small_aspect_ratio=0.9<aspect_ratio<2 and 20<w<90 and 20<h<185
    large_aspect_ratio=0.9<aspect_ratio<2 and 90<w<300 and 20<h<185

    #Kiem tra cac dieu kien loai tru va mong muon
    if large_rectangle or medium_rectangle:
        return True
    
    if unwanted_rectangle or high_perimeter_exclusion or low_area_exclusion:
        return False
    
    if small_aspect_ratio or large_aspect_ratio:
        return True

    return False

def detect_white_center(contour):
    # Tính toán bounding box của đường viền
    x, y, w, h = cv2.boundingRect(contour)
    
    # Cắt phần hình tròn từ ảnh gốc
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    circle_region = mask

    # Tính toán trung bình màu của vùng tròn
    # Ở đây, giả sử ảnh gốc có màu sắc và có thể sử dụng OpenCV để trích xuất
    center_color = np.mean(circle_region)

    # Kiểm tra xem màu sắc trung tâm có phải là màu trắng
    # Giá trị trung bình gần với giá trị của màu trắng (255, 255, 255)
    if center_color > 200:
        return True
    return False

import cv2
import numpy as np

# Kiểm tra nếu hình tròn có dấu X đỏ ở trung tâm
def detect_red_x_in_circle(contour, image):
    # Tính toán bounding box của đường viền
    x, y, w, h = cv2.boundingRect(contour)
    
    # Cắt phần hình tròn từ ảnh gốc
    circle_region = image[y:y+h, x:x+w]
    
    # Chuyển sang không gian màu HSV để dễ dàng phát hiện màu đỏ
    hsv = cv2.cvtColor(circle_region, cv2.COLOR_BGR2HSV)
    
    # Định nghĩa phạm vi màu đỏ trong không gian màu HSV
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    
    # Kết hợp các vùng màu đỏ
    mask_red = cv2.bitwise_or(mask_red, mask_red2)
    
    # Tìm các đường viền trong vùng màu đỏ
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Kiểm tra nếu có dấu "X đỏ" trong vùng tròn (2 đường chéo cắt nhau)
    for contour in contours_red:
        # Xử lý các đường viền đỏ
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) > 2:  # Kiểm tra nếu có sự cắt nhau của các đường thẳng
            return True
    
    return False

# Kiểm tra nếu hình tròn có dấu sẹt đỏ ở trung tâm
def detect_red_line_in_circle(contour, image):
    # Tính toán bounding box của đường viền
    x, y, w, h = cv2.boundingRect(contour)
    
    # Cắt phần hình tròn từ ảnh gốc
    circle_region = image[y:y+h, x:x+w]
    
    # Chuyển sang không gian màu HSV để dễ dàng phát hiện màu đỏ
    hsv = cv2.cvtColor(circle_region, cv2.COLOR_BGR2HSV)
    
    # Định nghĩa phạm vi màu đỏ trong không gian màu HSV
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    
    # Kết hợp các vùng màu đỏ
    mask_red = cv2.bitwise_or(mask_red, mask_red2)
    
    # Tìm các đường viền trong vùng màu đỏ
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Kiểm tra nếu có dấu "sẹt đỏ" (đường thẳng)
    for contour in contours_red:
        # Xử lý các đường viền đỏ
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 2:  # Kiểm tra nếu có một đường thẳng
            return True
    
    return False

# Kiem tra cac bien bao
# Xác định các biển báo giao thông
def detect_traffic_signs(image):
    mask_red, mask_blue = preprocess_image(image)
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    traffic_signs = []
    red_bboxes = []

    # Xét đường viền màu đỏ (chỉ hình tròn và hình tam giác)
    for contour in contours_red:
        if detect_circle_red(contour):
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            red_bboxes.append((x, y, w, h))
            traffic_signs.append(('Cam re trai', x, y, w, h, f'Area: {area} px, Perimeter: {perimeter} px, Circularity: {circularity:.2f}'))

        elif detect_triangle_red(contour):
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            red_bboxes.append((x, y, w, h))
            traffic_signs.append(('Cam di nguoc chieu', x, y, w, h, f'Area: {area} px, Perimeter: {perimeter} px'))

    # Xét đường viền màu xanh (hình tròn, hình vuông, hình chữ nhật)
    for contour in contours_blue:
        x, y, w, h = cv2.boundingRect(contour)
        is_inside_red_area = any(rx <= x <= rx + rw and ry <= y <= ry + rh for rx, ry, rw, rh in red_bboxes)

        if not is_inside_red_area:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            circularity = (4 * np.pi * area) / (perimeter ** 2)

            # Xác định các loại biển báo với điều kiện thêm
            if detect_circle_blue(contour):
                # Nếu là hình tròn màu xanh
                if 40 < w < 100 and 40 < h < 100:  # Điều kiện kích thước biển báo hướng phải đi vùng phải
                    traffic_signs.append(('Huong phai di vung phai', x, y, w, h, f'Area: {area} px, Perimeter: {perimeter} px, Circularity: {circularity:.2f}'))
            
            elif detect_red_x_in_circle(contour, image):  # Cung cấp đối số 'image' khi gọi hàm
                traffic_signs.append(('Cam dau X do', x, y, w, h, f'Area: {area} px'))

            elif detect_red_line_in_circle(contour, image):  # Cung cấp đối số 'image' khi gọi hàm
                traffic_signs.append(('Cam dung va do xe', x, y, w, h, f'Area: {area} px'))

            elif detect_rectangle_blue(contour):
                # Nếu là hình chữ nhật màu xanh
                if 20 < w < 120 and 20 < h < 60:
                    traffic_signs.append(('Cam di nguoc chieu', x, y, w, h, f'Width: {w} px, Height: {h} px'))
                elif 40 < w < 150 and 30 < h < 70:
                    traffic_signs.append(('Thang hoac Phai, cam queo Trai', x, y, w, h, f'Width: {w} px, Height: {h} px'))
                elif 50 < w < 160 and 30 < h < 80:
                    traffic_signs.append(('Cam dung va do xe', x, y, w, h, f'Width: {w} px, Height: {h} px'))

    return traffic_signs




#Main
def main(video_file):
    clean_images()
    vidcap = cv2.VideoCapture(video_file)

    # Lấy kích thước frame
    frame_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = vidcap.get(cv2.CAP_PROP_FPS)

    print(f"Video loaded: {video_file} ({frame_width}x{frame_height} @ {frame_rate} FPS)")

    while True:
        success, frame = vidcap.read()
        if not success:
            print("Video ended or failed to load.")
            break

        # Phát hiện biển báo
        traffic_signs = detect_traffic_signs(frame)

        # Vẽ hình chữ nhật quanh biển báo
        for shape, x, y, w, h, info in traffic_signs:
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            # In nhãn lên frame
            label = f"{shape}"  # In thông tin thêm về biển báo
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Thêm MSSV vào mỗi frame
        cv2.putText(frame, "522H0142_522H0148", (frame_width - 250, frame_height - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Hiển thị frame
        cv2.imshow('Traffic Sign Detection', frame)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Video playback stopped by user.")
            break

    # Giải phóng tài nguyên
    vidcap.release()
    cv2.destroyAllWindows()

# Chạy main
if __name__ == '__main__':
    main('video1.mp4')