import cv2
import numpy as np
import os
import pickle
import gzip
import math
from itertools import combinations

def return_label():
    return ['NG', 'OK', 'BK', 'IN']


class labeling:
    def __init__(self, source, dataset_dir, label):
        print(os.listdir(dataset_dir))
        print(label,'\n')

        self.dataset_dir = dataset_dir
        self.label = label
        self.source_dir = source
        self.img = None
        self.sequence = 0  # next img 글자 깜박이기 위한 트리거
        self.next_page = 0 # 다음장으로 넘기기 위함임. 0이면 가만히, 1이면 넘기기
        self.mode = 'Viewer mode'

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

    def pickle2img(self, data):
        num = 0
        self.img = None

        for i in cv2.split(data):
            if num == 0 :
                self.img = i
            else:
                self.img = cv2.hconcat([i, self.img])
            num += 1

    def show_label_img(self, data):
        self.img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
        self.x, self.y, self.z = data.shape
        self.x_sum = self.x * self.z
        self.top_box_height = 50
        self.bottom_box_height = 100


        top_box = np.zeros((self.top_box_height, self.x_sum, 3), np.uint8)
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
            self.img = cv2.putText(self.img, '{}/{} page'.format(self.now_count, self.total_count), (426, self.y + self.top_box_height + 35),
                                   cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1)

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
            self.img = cv2.putText(self.img, '{}/{} page'.format(self.now_count, self.total_count), (1400, self.y + self.top_box_height + 35),
                                   cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1)

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

    def move_file(self, data, folder, file_name2):
        dir = folder.split('/')[:-1]
        file_name = 'dataset/' + folder + '/' + str(len(os.listdir('dataset/' + folder))) + '.pickle'

        # save and compress.
        with gzip.open(file_name, 'wb') as f:
            pickle.dump(data, f)

        os.remove(file_name2)

    def main(self):
        cv2.namedWindow('Image')
        cv2.setMouseCallback("Image", self.mouse_event)

        quit = 'off'
        self.total_count = len(self.find_pickle(self.source_dir))
        self.now_count = 1

        for file_name in self.find_pickle(self.source_dir):
            self.next_page = 0
            data = self.load_pickle(os.path.join(self.source_dir, file_name))
            print('file : ', file_name, 'shape : ', data.shape)
            self.one_hot_encoder = [0]*len(self.label)

            while True:
                self.pickle2img(data)
                self.show_label_img(data)

                #cv2.namedWindow('Image', cv2.WND_PROP_FULLSCREEN)
                #cv2.moveWindow('Image', 0, 0)
                cv2.imshow('Image', self.img)

                '''
                단축키 설정
                '''

                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    quit = 'on'
                    break

                elif key & 0xFF == ord('1'):
                    if self.one_hot_encoder[0] == 0:
                        self.one_hot_encoder[0] = 1
                    else:
                        self.one_hot_encoder[0] = 0

                elif key & 0xFF == ord('2'):
                    if self.one_hot_encoder[1] == 0:
                        self.one_hot_encoder[1] = 1
                    else:
                        self.one_hot_encoder[1] = 0

                elif key & 0xFF == ord('3'):
                    if self.one_hot_encoder[2] == 0:
                        self.one_hot_encoder[2] = 1
                    else:
                        self.one_hot_encoder[2] = 0

                elif key & 0xFF == ord('4'):
                    if self.one_hot_encoder[3] == 0:
                        self.one_hot_encoder[3] = 1
                    else:
                        self.one_hot_encoder[3] = 0

                elif key & 0xFF == ord('n'):
                    self.next_page = 1

                if self.next_page == 1:
                    if 1 in self.one_hot_encoder:
                        folder = self.mark_folder()
                        print('{}폴더에 저장할겁니다.'.format(folder))
                        for_save_data = self.append(data)
                        self.move_file(data, folder, os.path.join(self.source_dir, file_name))

                    break

            self.now_count += 1
            if quit == 'on':
                break

        cv2.destroyAllWindows()


if __name__ == '__main__':
    dataset_dir = 'dataset'
    label = return_label()
    source = 'img'

    L = labeling(source, dataset_dir, label)
    L.main()
