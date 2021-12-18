import cv2
import numpy as np
import os
import time
import pickle
import gzip
import math
from itertools import combinations
from RESNET_MODEL import *
import tensorflow as tf

def return_label():
    return ['NG', 'OK', 'BK', 'IN']


class labeling:
    def __init__(self, source, dataset_dir, label, IMG_SIZE):
        print(os.listdir(dataset_dir))
        print(label,'\n')

        MODEL = RESNET50_model(IMG_SIZE, 5)
        self.classifier = MODEL.call_model()
        self.classifier.load_weights("best_model_xx.h5")

        self.dataset_dir = dataset_dir
        self.label = label
        self.source_dir = source
        self.img = None
        self.sequence = 0  # next img 글자 깜박이기 위한 트리거
        self.next_page = 0 # 다음장으로 넘기기 위함임. 0이면 가만히, 1이면 넘기기
        self.mode = 'Label mode' #'Viewer mode'
        self.idx_for_save_step = 0

        # dataset 폴더내에 label이 없으면 해당 폴더 생성
        self.make_label_dir()

    def make_label_dir(self):
        for list in range(len(self.label)):
            for combi in combinations(self.label, list+1):
                print(combi,'------')
                full_name = ''
                for idx, name in enumerate(combi):
                    if idx == 0:
                        full_name = full_name + name
                    else:
                        full_name = full_name + '_' + name
                print(full_name)

                if full_name not in os.listdir(dataset_dir):
                    os.mkdir(os.path.join(self.dataset_dir, full_name))
                    print('dataset folder 내 {}폴더를 생성하였습니다'.format(full_name))


    def load_pickle(self, name):
        with gzip.open(name, 'rb') as f:
            data = pickle.load(f)
        return np.array(data)

    def find_pickle(self, dir):
        list = os.listdir(dir)
        list.sort()
        return [word for word in list if '.pickle' in word]

    def show_label_img(self, data):
        self.img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
        self.x, self.y, self.z = data.shape
        self.x_sum = self.x * self.z
        self.top_box_height = 50
        self.bottom_box_height = 100


        top_box = np.zeros((self.top_box_height, self.x_sum, 3), np.uint8)
        #print(self.img.shape, self.x_sum)
        self.img = cv2.vconcat([top_box, self.img])

        bottom_box = np.zeros((self.bottom_box_height, self.x_sum, 3), np.uint8)
        self.img = cv2.vconcat([self.img, bottom_box])

        self.UI_scene()
        self.UI_Viewer_mode()
        self.UI_labeling_mode()
        self.img = cv2.line(self.img, (180, self.top_box_height+self.y+20),
                            (180, self.top_box_height+self.y+75), (153, 153, 153), 1)
        self.UI_main_contents()

    def UI_main_contents(self):
        #self.mode = 'Viewer2 mode'
        if self.mode == 'Viewer mode':
            if self.sequence < 21:
                color, size, bold = (102, 255, 255), .61, 1
            else:
                color, size, bold  = (255, 255, 255), .6, 2
            self.img = cv2.putText(self.img, 'Next Image', (266, self.y + self.top_box_height + 35),
                                   cv2.FONT_HERSHEY_SIMPLEX, size, color, bold)

            self.sequence += 1
            if self.sequence > 40:
                self.sequence = 0

            #왼쪽에 작은 상자 그리기
            self.img = cv2.rectangle(self.img, (190, self.y + self.top_box_height + 20),
                                     (240, self.y + self.top_box_height + 40), (153, 153, 153), -1)
            # 오른쪽에 큰 상자 그리기
            self.img = cv2.rectangle(self.img, (400, self.y + self.top_box_height + 20),
                                     (1500, self.y + self.top_box_height + 40), (103, 103, 103), -1)
            #현재 페이지 수 표시
            #self.img = cv2.putText(self.img, '{}/{} page'.format(self.now_count, self.total_count), (426, self.y + self.top_box_height + 35),
            #                       cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1)

        else:
            idx = 0
            #self.one_hot_encoder = [1,1,0]
            for idx, label in enumerate(self.label):
                if self.one_hot_encoder[idx] == 0:
                    color = (255,255,255)
                else:
                    color = (0,0,255)
                self.img = cv2.putText(self.img, label, (idx*170+290, self.y + self.top_box_height + 70),
                                       cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)

            # 왼쪽에 작은 상자 그리기
            self.img = cv2.rectangle(self.img, (190, self.y + self.top_box_height + 55),
                                     (240, self.y + self.top_box_height + 75), (153, 153, 153), -1)
            # 오른쪽에 큰 상자 그리기
            self.img = cv2.rectangle(self.img, (idx*170+430, self.y + self.top_box_height + 55),
                                     (1500, self.y + self.top_box_height + 75), (103, 103, 103), -1)
            # 저장하기
            self.img = cv2.putText(self.img, 'Transfer data or Next page', (idx*170+430, self.y + self.top_box_height + 35),
                                   cv2.FONT_HERSHEY_SIMPLEX, .61, (255, 255, 255), 2)

            if 1 not in self.one_hot_encoder:
                self.img = cv2.putText(self.img, 'There is no selected. it will move to the next page.',
                                       (idx * 170 + 435, self.y + self.top_box_height + 69),
                                       cv2.FONT_HERSHEY_SIMPLEX, .5, (150, 150, 150), 1)
            else:
                self.img = cv2.putText(self.img,
                                       'Label was selected. it will save this image.',
                                       (idx * 170 + 435, self.y + self.top_box_height + 69),
                                       cv2.FONT_HERSHEY_SIMPLEX, .5, (150, 150, 150), 1)

            #현재 페이지 수 표시
            #self.img = cv2.putText(self.img, '{}/{} page'.format(self.now_count, self.total_count), (1400, self.y + self.top_box_height + 35),
            #                       cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1)

    def UI_scene(self):
        #맨 상위 scene1,2,3,...표시
        for i in range(self.z):
            text = 'SCENE.'+ str(i+1)
            self.img = cv2.putText(self.img, text, (i*300+5, 33), cv2.FONT_HERSHEY_SIMPLEX,
                              .35, (102, 255, 255), 1)

            if i != 0:
                self.img = cv2.line(self.img, (i*300,self.top_box_height), (i*300,self.x+30), (0,0,0), 4)

    def UI_Viewer_mode(self):
        # viewer mode 표시
        if self.mode == 'Viewer mode':
            self.img = cv2.putText(self.img, 'Viewer mode', (30, self.y + self.top_box_height+35),
                              cv2.FONT_HERSHEY_SIMPLEX, .6, (102, 255, 255), 1)
        else:
            self.img = cv2.putText(self.img, 'Viewer mode', (30, self.y +self.top_box_height + 35),
                              cv2.FONT_HERSHEY_SIMPLEX, .6,
                              (153, 153, 153), 1)

    def UI_labeling_mode(self):
        # labeling mode 표시
        if self.mode == 'Label mode':
            self.img = cv2.putText(self.img, 'Label  mode', (30, self.y + self.top_box_height + 70),
                              cv2.FONT_HERSHEY_SIMPLEX, .6, (102, 255, 255), 1)
        else:
            self.img = cv2.putText(self.img, 'Label  mode', (30, self.y + self.top_box_height + 70),
                              cv2.FONT_HERSHEY_SIMPLEX, .6,
                              (153, 153, 153), 1)

    def mark_folder(self):
        tray = []
        for idx, i in enumerate(self.one_hot_encoder):
            if i == 1:
                tray.append(self.label[idx])

        full_name = ''
        for idx, folder in enumerate(tray):
            if idx == 0:
                full_name = folder
            else:
                full_name = full_name + '_' + folder

        return full_name

    def mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.LBUTTONDOWN_x, self.LBUTTONDOWN_y = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            self.LBUTTONUP_x, self.LBUTTONUP_y = x, y
            if math.fabs((self.LBUTTONUP_x - self.LBUTTONDOWN_x)) > 20 or \
                    math.fabs((self.LBUTTONUP_y - self.LBUTTONDOWN_y)) > 20:
                print('not selected...')
            else:
                if self.mode == 'Viewer mode':
                    self.view_action()
                else:
                    self.label_action()

    def view_action(self):
        print('view mode')
        print('LBUTTONUP position : ', self.LBUTTONUP_x, self.LBUTTONUP_y)

        if 240 < self.LBUTTONUP_x < 400 and \
                370 < self.LBUTTONUP_y < 395:
            print('next page')
            self.next_page = 1
        elif 32 < self.LBUTTONUP_x < 150 and 400 < self.LBUTTONUP_y < 430:
            print('to label mode change...')
            self.mode = 'Label mode'

    def label_action(self):
        idx = 0
        print('labeling mode')
        print('LBUTTONUP position : ', self.LBUTTONUP_x, self.LBUTTONUP_y)

        #모드 체인지
        if 30 < self.LBUTTONUP_x < 150 and 370 < self.LBUTTONUP_y < 390:
            print('to view mode change...')
            self.mode = 'Viewer mode'

        #label 체인지
        for idx, _ in enumerate(self.label):
            if 290+idx*170 < self.LBUTTONUP_x < 380+idx*170 and 370 < self.LBUTTONUP_y < 420:
                if self.one_hot_encoder[idx] == 1:
                    self.one_hot_encoder[idx] = 0
                else:
                    self.one_hot_encoder[idx] = 1
                print(self.one_hot_encoder)

        # 다음장으로 넘기기
        if 260 + (idx+1) * 170 < self.LBUTTONUP_x < 550 + (idx+1) * 170 and 370 < self.LBUTTONUP_y < 410:
            self.next_page = 1

    def append(self, data):
        return np.array(data).astype('float32') / 255, np.array(self.one_hot_encoder)

    def save_file(self, data):
        self.idx_for_save_step += 1
        if self.idx_for_save_step == 10 and self.one_hot_encoder != [0,0,0,0]:
            folder = self.mark_folder()
            print('{}폴더에 저장합니다.'.format(folder))

            num_tray = ['0', '0', '0', '0', '0', '0', '0', '0', '.', '0']
            before_num = list(str(len(os.listdir('dataset/' + folder)))) #'0',  '123',  '3234'

            tray = []
            for i in reversed(before_num):
                tray.append(i)

            for idx, i in enumerate(tray):
                num_tray[idx] = i

            text = ''
            for idx, i in enumerate(num_tray):
                text = i + text

            file_name = 'dataset/' + folder + '/' + text[2:] + '.pickle'

            # save and compress.
            with gzip.open(file_name, 'wb') as f:
                pickle.dump(data, f)

            self.idx_for_save_step = 0

        
    def predict_and_mark(self, data):
        IMG_SIZE = 250
        data = np.array(cv2.resize(data, (IMG_SIZE, IMG_SIZE)))
        img_array = np.array(data).astype('float32') / 255
        img_array = img_array.reshape(1, IMG_SIZE, IMG_SIZE, 5)

        A = self.classifier.predict(img_array)[0]
        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        print(A,'\n')
        for i in range(4):
            if A[i] > .8 :
                self.one_hot_encoder[i] = 1
                    
        return A

    def pickle2img(self, data):
        num = 0
        self.img = None

        for i in cv2.split(data):
            if num == 0 :
                self.img = i
            else:
                self.img = cv2.hconcat([i, self.img])
            num += 1

    def draw_graph(self, data):
        height, width, _ = self.img.shape
        #print(height, width) #450 1500

        img = np.zeros((500, width, 3), np.uint8)


        return img



    def main(self):
        cap = cv2.VideoCapture(0)
        
        IMG_SIZE_to_show = 300
        cv2.namedWindow('Image')
        cv2.setMouseCallback("Image", self.mouse_event)

        tray = []  # 이미지를 임시저장하기위한 공간
        save_count = 0
        start = 0  # 초당 30장 인식

        while True:
            self.one_hot_encoder = [0] * len(self.label)

            _, frame = cap.read()
            frame = cv2.flip(frame, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = np.array(cv2.resize(frame, (IMG_SIZE_to_show, IMG_SIZE_to_show)))

            if start < 151:
                tray.append(frame)

            elif start > 150:
                save_count = 0
                tray = tray[:-1]
                tray.insert(0, frame)

                step = 20
                point = 0
                data = cv2.merge((tray[point], tray[step*1], tray[step*2], tray[step*3], tray[step*4]))

                '''리스트 형태를 연속 5장짜리 이미지로 변환'''
                self.pickle2img(data)

                '''5프레임 이후부터는 딥러닝 예측 가동하여 ont_hot_encoder에 기록'''
                result = self.predict_and_mark(data)

                '''화면 출력을 위한 화면 구성'''
                self.show_label_img(data)
                info_img = self.draw_graph(data)
                self.img = cv2.vconcat([self.img, info_img])
                '''30프레임 간격으로 폴더에 pickle파일 저장'''
                self.save_file(data)

                cv2.imshow('Image', self.img)


                '''단축키 설정'''
                key = cv2.waitKey(1)

                if key == 27:
                    break

            save_count += 1

            if start > 150:
                start = 151
            else:
                start += 1


        cv2.destroyAllWindows()
        cap.release()


if __name__ == '__main__':
    dataset_dir = 'dataset'
    label = return_label()
    source = 'img'
    img_size = 250

    L = labeling(source, dataset_dir, label, img_size)
    L.main()
