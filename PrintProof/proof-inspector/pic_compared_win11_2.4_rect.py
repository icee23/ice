# -*- coding: utf-8 -*-

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QLineEdit, QProgressBar
#QMainWindow, QLabel, QGridLayout, QWidget, QPushButton, QHBoxLayout, QFileDialog)
from PyQt5.QtCore import QThread, pyqtSignal
import sys
import os
import cv2
import numpy as np
import pymupdf  # PyMuPDF
import re
import fitz
from PIL import Image
import shutil
from pillow_heif import register_heif_opener

prob = 0.05 # 100%
bleed = 0
print_bleed = 60
dpi_index = 300
bleed_check = False
test_len = 0
pdf_check = 0 # 0 start 1 pdf 2 png 3 heic
hidden_pixel = 60
find_check = False
good_match_points = 20

register_heif_opener()

if getattr(sys, 'frozen', False):
    application_path_ = os.path.dirname(sys.executable) # sys.executable _MEIPASS
    # /Users/allen_lin/Desktop/workspace/PDFmerge/dist/PDF_Merge.app/Contents/MacOS
    temp = re.split('/', application_path_)
    application_path = temp[0]
    for i in range(1, len(temp)-3):
        application_path += '/'
        application_path += temp[i]
    # /Users/allen_lin/Desktop/workspace/PDFmerge/dist

else:
    application_path = os.path.dirname(os.path.abspath(__file__))

save_path = os.path.join(application_path, 'pic_compared_files')
os.makedirs(save_path, exist_ok=True)
os.makedirs(save_path + "/tmp", exist_ok=True)
os.makedirs(save_path + "/tmp2", exist_ok=True)
os.makedirs(save_path + "/output", exist_ok=True)
os.makedirs(save_path + "/Not_found", exist_ok=True)

def multi_scale_template_matching(image, template, scale_range=(0.25, 1), scale_steps=20, method=cv2.TM_CCOEFF_NORMED):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) == 3 else template
    
    best_match = None
    best_scale = None
    best_score = -np.inf
    
    h, w = gray_template.shape[:2]
    
    for scale in np.linspace(scale_range[0], scale_range[1], scale_steps):
        resized = cv2.resize(gray_template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        
        if resized.shape[0] > gray_image.shape[0] or resized.shape[1] > gray_image.shape[1]:
            continue
        
        res = cv2.matchTemplate(gray_image, resized, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            score = -min_val
            loc = min_loc
        else:
            score = max_val
            loc = max_loc
        
        if score > best_score:
            best_score = score
            best_match = loc
            best_scale = scale
            #cv2.imwrite(".\\pic_compared_files\\output\\template_2.png", resized)

    return best_match, best_scale, best_score

def locate_card(large_image, card_image, output):
    global good_match_points, find_check
    find_check = False
    # 轉換為灰度圖
    large_gray = cv2.cvtColor(large_image, cv2.COLOR_BGR2GRAY)
    card_gray = cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY)
    
    # 初始化SIFT檢測器
    sift = cv2.SIFT_create()
    
    # 找到特徵點和描述符
    kp1, des1 = sift.detectAndCompute(card_gray, None)
    kp2, des2 = sift.detectAndCompute(large_gray, None)
    
    # FLANN參數
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    
    # 使用FLANN匹配器
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    
    # 儲存好的匹配
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance: # 0.7 0.6 0.5
            good_matches.append(m)
    
    # 如果找到足夠的匹配點
    if len(good_matches) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # 計算單應性矩陣
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        # 檢查內點的數量
        matchesMask = mask.ravel().tolist()
        inliers = sum(matchesMask)

        # 計算匹配點的分佈
        if inliers > good_match_points:
            x_coords = dst_pts[:, 0, 0]
            y_coords = dst_pts[:, 0, 1]
            x_range = np.max(x_coords) - np.min(x_coords)
            y_range = np.max(y_coords) - np.min(y_coords)

            # 如果匹配點分佈過於集中，可能是誤匹配
            if x_range > 100 and y_range > 100:
                # 獲取卡片的角點
                h, w = card_gray.shape
                pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        
                # 透視變換
                dst = cv2.perspectiveTransform(pts, M)
                #print(dst)
                # 在大圖上畫出卡片的輪廓
                print(f"找到卡片，內點數量：{inliers}")
                cv2.polylines(output, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
                find_check = True
            else:
                print("匹配點分佈過於集中，可能是誤匹配")
        else:
            print("內點數量不足，未找到可靠匹配")
    else:
        print("良好匹配點不足，可能沒有找到卡片")
        
    return output, find_check

def convert_heic(input_path, output_path, output_format='png'):
    """
    將 HEIC 文件轉換為 JPG 或 PNG。

    :param input_path: HEIC 文件的路徑
    :param output_format: 輸出格式，'jpg' 或 'png'
    :return: 輸出文件的路徑
    """
    # 檢查輸出格式
    if output_format.lower() not in ['jpg', 'png']:
        raise ValueError("輸出格式必須是 'jpg' 或 'png'")

    try:
        # 打開 HEIC 文件
        with Image.open(input_path) as img:
            # 如果原圖有 EXIF 數據，保留它
            exif = img.info.get('exif')

            # 轉換並保存
            if output_format.lower() == 'jpg':
                img.convert('RGB').save(output_path, 'JPEG', quality=95, exif=exif)
            else:  # PNG
                img.save(output_path, 'PNG', exif=exif)

        print(f"文件已成功轉換：{output_path}")
        return output_path

    except Exception as e:
        print(f"轉換過程中出錯：{str(e)}")
        return None

def read_image_with_unicode_path(file_path):
    # 以二進制模式讀取文件
    with open(file_path, 'rb') as f:
        buffer = f.read()
    # 將二進制數據轉換為 NumPy 數組
    buffer_array = np.frombuffer(buffer, dtype=np.uint8)
    # 使用 cv2.imdecode 解碼圖像
    image = cv2.imdecode(buffer_array, cv2.IMREAD_COLOR)
    return image

def write_image_with_unicode_path(file_path, image):
    # 確保目錄存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # 將圖像編碼為 PNG 格式的字節串
    is_success, buffer = cv2.imencode(".png", image)
    if is_success:
        # 以二進制寫入模式打開文件
        with open(file_path, "wb") as f:
            # 將字節串寫入文件
            f.write(buffer)
        return True
    return False

def compare_process(thread_instance, self):
    global bleed, bleed_check, test_len, prob, hidden_pixel, find_check
    print("process")
    '''
    self.info_panel.setPlainText("process")
    QApplication.processEvents()
    '''

    # process
    #print(self.in_page)
    #print(self.in2_page)
    '''
    print()
    sys.exit()
    '''
    for index_j, j in enumerate(self.in2_page): # for j in range(self.in2_png_len):
        if not thread_instance.is_running:
            break
        image0 = read_image_with_unicode_path(self.in2_png[j]) # , cv2.IMREAD_GRAYSCALE 轉灰階
        image0_ = image0.copy()
        print_image = image0.copy()
        #image0_ = self.preprocess_image2(image0)

        fp_path = os.path.join(save_path, f"{self.in2_pure[j]}_findcheck.txt")
        #fp = open(save_path + "/" + self.in2_png_pure2[j] + "_findcheck.txt", "w")
        fp = open(fp_path, "w")
        for index_i, i in enumerate(self.in_page): # for i in range(self.in_png_len):
            if not thread_instance.is_running:
                break
            #print(i)
            if (bleed_check == True):
                image__ = read_image_with_unicode_path(self.in_png[0])
            else:
                image__ = read_image_with_unicode_path(self.in_png[i]) # , cv2.IMREAD_GRAYSCALE 轉灰階
            #image__ = self.preprocess_image2(cv2.imread(self.in_png[i]))
            h__, w__ = image__.shape[:2]
            '''
            print(h__, w__)
            print(bleed, bleed)
            print(w__ - bleed, h__ - bleed)
            '''
            #image_ = image__[bleed:h__ - bleed, bleed:w__ - bleed]
            image_ = image__[bleed:h__ - bleed - hidden_pixel, bleed:w__ - bleed]
            if (bleed_check == True):
                test_path = os.path.join(save_path, f"testtest.png")
                #cv2.imwrite(save_path + "/testtest.png", image_)
                test_success = write_image_with_unicode_path(test_path, image_)
                if test_success:
                    print(f"圖片成功保存至: {test_path}")
                else:
                    print(f"保存圖片失敗: {test_path}")
                print("bleed & hidden_pixel: ", end="")
                print(bleed, hidden_pixel)
                thread_instance.is_running = False

                self.computation_thread.signal_callback("test finish", 100)
                return None
                #sys.exit()

            h, w = image_.shape[:2]
            H, W = image0.shape[:2]
            if (h > H or w > W):
                self.prob.append(-1)
                self.x1.append((0, 0))
                self.x2.append((-1, -1))
                continue
            '''
            res = cv2.matchTemplate(image0, image_, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            '''
            #match, scale, score = multi_scale_template_matching(image0_, image_)
            print_image, find_check = locate_card(image0_, image_, print_image)

            #scaled_h, scaled_w = int(h * scale), int(w * scale)
            #top_left = match
            #bottom_right = (top_left[0] + scaled_w, top_left[1] + scaled_h)
            #top_left = (max_loc[0] + print_bleed, max_loc[1] + print_bleed)
            #bottom_right = (top_left[0] + w - print_bleed * 2, top_left[1] + h - print_bleed * 2)

            '''
            self.prob.append(score) # max_val
            self.x1.append(top_left)
            self.x2.append(bottom_right)
            self.scale.append(scale)
            '''

            #print(min_val, max_val, min_loc, max_loc)
            '''
            if (max_val) >= prob:
                cv2.rectangle(image0, top_left, bottom_right, (255, 255, 255), -1)
                self.in_png_find[i] = 1
                print(self.in_png[i], end=" 1 ", file=fp)
                print("%.5f"%(max_val), end=" ", file=fp)
                print("%d %d %d %d"%(top_left[0], top_left[1], bottom_right[0], bottom_right[1]), file=fp)
                    
            else:
                print(self.in_png[i], end=" 0 ", file=fp)
                print("%.5f"%(max_val), file=fp)
            '''
                
            print(self.in_png[i], end=" ", file=fp)
            #print("%.5f"%(score), end=" ", file=fp) # max_val
            #print("%.5f"%(scale), end=" ", file=fp)
            #print("%d %d %d %d"%(top_left[0], top_left[1], bottom_right[0], bottom_right[1]), file=fp)
            print(find_check, file=fp)
                
            
            process_ = str(index_i+1) + "/" + str(index_j+1) + "  total " + str(self.in2_page_len)
            #self.info_panel.setPlainText(process_)
            process_2 = int((index_i + 1) / self.in_page_len * 100)
            #self.progress_bar.setValue(process_2)
            if (process_2 < 100):
                self.computation_thread.signal_callback(process_, process_2)
            #QApplication.processEvents()
            

        fp.close()


        '''
        # print png
        print("draw")
        tmp_prob = prob
        tmp = image0.copy()
        for k in range(len(self.prob)):
            if self.prob[k] >= tmp_prob:
                cv2.rectangle(tmp, self.x1[k], self.x2[k], (0, 0, 255), 10)
                cv2.putText(tmp, f"Score: {self.prob[k]:.2f}, Scale: {self.scale[k]:.2f}", (self.x1[k][0], self.x1[k][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imwrite(save_path + "/output/" + self.in2_png_pure2[j] + "_" + str(tmp_prob) + ".png", tmp)
           
        cv2.imwrite(save_path + "/output/test_card.png", image_)
        cv2.imwrite(save_path + "/output/test_Limage.png", image0_)
        '''
        '''
        for prob_index in range(85, 101):
            tmp_prob = prob_index/100
            tmp = image0.copy()
            for k in range(len(self.prob)):
                if self.prob[k] >= tmp_prob:
                    cv2.rectangle(tmp, self.x1[k], self.x2[k], (255, 0, 0), 10)
            cv2.imwrite(save_path + "/output/" + self.in2_png_pure2[j] + "_" + str(tmp_prob) + ".png", tmp)
        '''

        # save png
        print("save")
        output_path = os.path.join(save_path, "output", f"{self.in2_pure[j]}_{prob}.png")
        success = write_image_with_unicode_path(output_path, print_image)
        if success:
            print(f"圖片成功保存至: {output_path}")
        else:
            print(f"保存圖片失敗: {output_path}")
        #cv2.imwrite(save_path + "\\output\\" + self.in2_pure[j] + "_" + str(prob) + ".png", print_image)
        self.computation_thread.signal_callback(process_, process_2)
        '''
        # save txt
        fp = open(save_path + "/" + self.in2_png_pure2[j] + "_Not_found.txt", "w")
        for i in range(len(self.in_png_find)):
            if (self.in_png_find[i] == 0):
                print(self.in_pure[i]+".pdf", file=fp)
        fp.close()
        '''

        del print_image, image0, image__, image_,
        self.prob.clear()
        self.x1.clear()
        self.x2.clear()

    #process_ = str(index_i+1) + "/" + str(index_j+1)
    #self.info_panel.setPlainText("Finish ... " + process_)

class ComputationThread(QThread):
    update_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    
    
    def __init__(self, external_func, main_window):
        super().__init__()
        self.is_running = True
        self.external_func = external_func
        self.main_window = main_window

    def run(self):     
        self.external_func(self, self.main_window) #, update_signal, progress_signal

    def stop(self):
        self.is_running = False

    def signal_callback(self, message, progress):
        self.update_signal.emit(message)
        self.progress_signal.emit(progress)

class MyWidget(QtWidgets.QWidget): 
    def __init__(self):
        super().__init__()
        self.setWindowTitle('大版對圖')
        self.resize(800, 610)
        self.computation_thread = None

        self.fileList = []
        self.in_pdf = []
        self.in_pure = []
        self.in_png_pure2 = []
        self.in2_pdf = []
        self.in2_pure = []
        self.in2_png_pure2 = []
        self.out_list = []

        self.in_png = []
        self.in_png_len = 0
        self.in_page = []
        self.in_page_len = 0
        self.in2_png = []
        self.in2_png_len = 0
        self.in2_page = []
        self.in2_page_len = 0

        self.in_png_find = []
        self.in_png_notfound = []

        self.prob = []
        self.x1 = []
        self.x2 = []
        self.scale = []

        self.btn1 = QtWidgets.QPushButton(self)
        self.btn1.setText('選擇檔案')
        self.btn1.setGeometry(20,5,80,35)
        self.btn1.clicked.connect(self.open1)

        self.btn2 = QtWidgets.QPushButton(self)
        self.btn2.setText('選擇檔案')
        self.btn2.setGeometry(420,5,80,35)
        self.btn2.clicked.connect(self.open2)

        label1 = QtWidgets.QLabel(self)   # 在 Form 裡加入標籤
        label1.setText('鎖外框')     # 設定標籤文字
        label1.setGeometry(170,5,80,35)

        self.input = QtWidgets.QTextEdit(self)
        self.input.setGeometry(20,45,360,500)
        self.input.setLineWrapMode(self.input.LineWrapMode.NoWrap)

        label4 = QtWidgets.QLabel(self)   # 在 Form 裡加入標籤
        label4.setText('指定頁 a-b,c')     # 設定標籤文字
        label4.setGeometry(235,5,80,35) 
        self.input_field4 = QLineEdit(self)
        self.input_field4.setGeometry(300,5,80,35)

        label2 = QtWidgets.QLabel(self)   # 在 Form 裡加入標籤
        label2.setText('大版')     # 設定標籤文字
        label2.setGeometry(590,5,80,35)

        self.input2 = QtWidgets.QTextEdit(self)
        self.input2.setGeometry(420,45,360,500)
        self.input2.setLineWrapMode(self.input2.LineWrapMode.NoWrap)

        label5 = QtWidgets.QLabel(self)   # 在 Form 裡加入標籤
        label5.setText('指定頁 a-b,c')     # 設定標籤文字
        label5.setGeometry(635,5,80,35)
        self.input_field5 = QLineEdit(self)
        self.input_field5.setGeometry(700,5,80,35) #700,5,80,35

        self.pushButton = QtWidgets.QPushButton(self)
        self.pushButton.setText("start")
        self.pushButton.setGeometry(360, 550, 80, 35)
        self.pushButton.clicked.connect(self.call_all)

        self.stop_button = QtWidgets.QPushButton(self)
        self.stop_button.setText("reset")
        self.stop_button.setGeometry(260, 550, 80, 35)
        self.stop_button.clicked.connect(self.stop_computation)
        #self.stop_button.setEnabled(False)

        # 創建文本編輯器來顯示資訊
        self.info_panel = QtWidgets.QTextEdit(self)
        self.info_panel.setReadOnly(True)
        self.info_panel.setGeometry(460, 550, 320, 40)

        # 創建輸入框
        label3 = QtWidgets.QLabel(self)   # 在 Form 裡加入標籤
        label3.setText('出血 (建議60)')     # 設定標籤文字
        label3.setGeometry(20,550,80,35)
        self.input_field = QLineEdit(self)
        self.input_field.setGeometry(100,550,80,35)

        # 進度條
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setGeometry(20, 590, 760, 20)
        self.progress_bar.setRange(0, 100)

    def open1(self):
        filePath , filterType = QtWidgets.QFileDialog.getOpenFileNames()
        #print(filePath , filterType)
        for i in filePath:
            file_ext = os.path.splitext(i)[1].lower()
            if i not in self.in_pdf:
                if file_ext == '.pdf':
                    self.in_pdf.append(i)
                    # pure file name
                    temp2 = os.path.splitext(os.path.basename(i))[0]
                    #print(temp2)
                    self.in_pure.append(temp2)

        output = '\n'.join(self.in_pdf)
        self.input.setText(output)

        QApplication.processEvents()

    def open2(self):
        global pdf_check
        filePath , filterType = QtWidgets.QFileDialog.getOpenFileNames()
        #print(filePath , filterType)
        for i in filePath:
            file_ext = os.path.splitext(i)[1].lower()
            if i not in self.in2_pdf:
                if file_ext == '.pdf' and (pdf_check == 0 or pdf_check == 1):
                    if pdf_check == 0: pdf_check = 1
                    self.in2_pdf.append(i)
                    temp2 = os.path.splitext(os.path.basename(i))[0]
                    self.in2_pure.append(temp2)
                elif file_ext in ['.jpg', '.jpeg', '.png'] and (pdf_check == 0 or pdf_check == 2):
                    if pdf_check == 0: pdf_check = 2
                    self.in2_pdf.append(i)
                    temp2 = os.path.splitext(os.path.basename(i))[0]
                    self.in2_pure.append(temp2)
                elif file_ext in ['.heic'] and (pdf_check == 0 or pdf_check == 3):
                    if pdf_check == 0: pdf_check = 3
                    self.in2_pdf.append(i)
                    temp2 = os.path.splitext(os.path.basename(i))[0]
                    self.in2_pure.append(temp2)

        output2 = '\n'.join(self.in2_pdf)
        self.input2.setText(output2)

        QApplication.processEvents()

    def call_all(self):
        global bleed, bleed_check, test_len, pdf_check, hidden_pixel
        print("start")
        self.pushButton.setEnabled(False)
        self.btn1.setEnabled(False)
        self.btn2.setEnabled(False)
        self.stop_button.setEnabled(True)

        input_text = self.input_field.text()
        if input_text != "":
            temp = re.split(" ", input_text)
            bleed = int(temp[0])
            print(bleed)
            if (len(temp) == 2 and temp[-1] == "test"):
                bleed_check = True
            elif (len(temp) == 2 and temp[-1] != "test"):
                hidden_pixel = int(temp[1])
            if (len(temp) == 3 and temp[-1] == "test"):
                bleed_check = True
                #test_len = int(temp[2])
                hidden_pixel = int(temp[1])

        # input pdf2png and save into tmp
        #print(len(self.in_pdf))
        for i in range(len(self.in_pdf)):
            #print(self.in_pdf[i])
            #print(self.in_pure[i])
            pdf_file = fitz.open(self.in_pdf[i])
            j = 1
            for page in pdf_file:
                pix = page.get_pixmap(alpha=False, colorspace=fitz.csRGB, dpi=dpi_index)
                tmp = self.in_pure[i]
                if (j == 1):
                    fn = os.path.join(save_path, "tmp", tmp + ".png")
                    fn2 = tmp
                else:
                    fn = os.path.join(save_path, "tmp", tmp + "-" + str(j) + ".png")
                    #fn = save_path + "\\tmp\\" + tmp + "-" + str(j) + ".png"
                    fn2 = tmp + "-" + str(j)
                
                #print(fn)

                self.in_png.append(fn)
                self.in_png_find.append(0)
                self.in_png_pure2.append(fn2)
                pix.save(fn)
                j += 1

                if (bleed_check == True): break
            if (bleed_check == True): break

        # input2 pdf2png and save into tmp2
        #print(len(self.in2_pdf))
        if pdf_check == 1:
            for i in range(len(self.in2_pdf)):
                #print(self.in2_pdf[i])
                #print(self.in2_pure[i])
                pdf_file = fitz.open(self.in2_pdf[i])
                j = 1
                for page in pdf_file:
                    pix = page.get_pixmap(alpha=False, colorspace=fitz.csRGB, dpi=dpi_index)
                    tmp = self.in2_pure[i]
                    if (j == 1):
                        fn = os.path.join(save_path, "tmp2", tmp + ".png")
                        fn2 = tmp
                    else:
                        fn = os.path.join(save_path, "tmp2", tmp + "-" + str(j) + ".png")
                        #fn = save_path + "\\tmp2\\" + tmp + "-" + str(j) + ".png"
                        fn2 = tmp + "-" + str(j)

                    #print(fn)

                    self.in2_png.append(fn)
                    self.in2_png_pure2.append(fn2)
                    pix.save(fn)
                    j += 1
        elif pdf_check == 2:
            for i in range(len(self.in2_pdf)):
                tmp = self.in2_pure[i]
                fn = os.path.join(save_path, "tmp2", tmp)
                fn2 = tmp[:-4]
                shutil.copy2(self.in2_pdf[i], fn)

                self.in2_png.append(fn)
                self.in2_png_pure2.append(fn2)
        elif pdf_check == 3:
            for i in range(len(self.in2_pdf)):
                tmp = self.in2_pure[i]
                fn = os.path.join(save_path, "tmp2", tmp + ".png")
                fn2 = tmp
                convert_heic(self.in2_pdf[i], fn, 'png')

                self.in2_png.append(fn)
                self.in2_png_pure2.append(fn2)

        self.in_png_len = len(self.in_png)
        self.in2_png_len = len(self.in2_png)

        input_text = self.input_field4.text()
        if (input_text != "" and len(self.in_pdf) ==  1):
            temp0 = input_text.replace(" ", "")
            temp = re.split(",", temp0)
            for i in range(len(temp)):
                if (temp[i] != ""):
                    if("-" in temp[i]):
                        temp1 = re.split("-", temp[i])
                        for j in range(int(temp1[0])-1, int(temp1[1])):
                            self.in_page.append(j)
                    else:
                        self.in_page.append(int(temp[i])-1)
        else:
            for j in range(self.in_png_len):
                self.in_page.append(j)

        input_text = self.input_field5.text()
        if (input_text != "" and len(self.in2_pdf) ==  1):
            temp0 = input_text.replace(" ", "")
            temp = re.split(",", temp0)
            for i in range(len(temp)):
                if("-" in temp[i]):
                    temp1 = re.split("-", temp[i])
                    for j in range(int(temp1[0])-1, int(temp1[1])):
                        self.in2_page.append(j)
                else:
                    self.in2_page.append(int(temp[i])-1)
        else:
            for j in range(self.in2_png_len):
                self.in2_page.append(j)

        self.in_page_len = len(self.in_page)
        self.in2_page_len = len(self.in2_page)

        self.computation_thread = ComputationThread(compare_process, self)
        self.computation_thread.update_signal.connect(self.update_text_area)
        self.computation_thread.progress_signal.connect(self.update_progress_bar)
        self.computation_thread.start()

    def stop_computation(self):
        global bleed_check, find_check, pdf_check
        '''
        self.computation_thread.signal_callback("", 0)
        self.info_panel.clear()
        self.progress_bar.reset()
        '''

        #if self.computation_thread and self.computation_thread.isRunning():
        try:
            if self.computation_thread.isRunning():
                self.computation_thread.stop()
                self.computation_thread.wait()
        except:
            pass

        bleed_check = False
        find_check = False
        pdf_check = 0

        #self.computation_thread.data.clear()  # 清理數據
        self.fileList.clear()
        self.in_pdf.clear()
        self.in_pure.clear()
        self.in_png_pure2.clear()
        self.in2_pdf.clear()
        self.in2_pure.clear()
        self.in2_png_pure2.clear()
        self.out_list.clear()

        self.in_png.clear()
        self.in_png_len = 0
        self.in_page.clear()
        self.in_page_len = 0
        self.in2_png.clear()
        self.in2_png_len = 0
        self.in2_page.clear()
        self.in2_page_len = 0

        self.in_png_find.clear()

        self.prob.clear()
        self.x1.clear()
        self.x2.clear()
        self.scale.clear()

        self.input.clear()
        self.input2.clear()
        self.input_field4.clear()
        self.input_field5.clear()

        #self.update_signal.emit(message)
        #self.progress_signal.emit(progress)

        '''
        self.info_panel.clear()
        self.progress_bar.reset()
        #self.info_panel.setPlainText("stop and restart")

        QApplication.processEvents()
        '''
        self.pushButton.setEnabled(True)
        self.btn1.setEnabled(True)
        self.btn2.setEnabled(True)
        self.stop_button.setEnabled(False)

    def update_text_area(self, message):
        cursor = self.info_panel.textCursor()
        cursor.movePosition(cursor.End)
        cursor.select(cursor.LineUnderCursor)
        cursor.removeSelectedText()
        cursor.insertText(message)

    def update_progress_bar(self, value):
        self.progress_bar.setValue(value)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    Form = MyWidget()
    Form.show()
    sys.exit(app.exec_())