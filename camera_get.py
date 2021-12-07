import cv2
import numpy as np
import os
import pickle, gzip


def save_data(data):
    name = str(len(os.listdir('img')))
    file_name = 'img/Not_checked_' + name + '.pickle'
    
    # save and compress.
    with gzip.open(file_name, 'wb') as f:
        pickle.dump(data, f)

    print('{}장의 이미지 획득 완료..'.format(name))


if __name__ == '__main__':
    # Create a VideoCapture object
    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if cap.isOpened() == False: 
        print("Unable to read camera feed")

    save_count = 0
    tray = []
    IMG_SIZE = 100
    
    while(True):
        img_final = None
        
        ret, frame = cap.read()
        
        if ret == True: 

            # Display the resulting frame    
            cv2.imshow('Image', frame)
            
            # Press Q on keyboard to quit program
            key = cv2.waitKey(1)
            
            if key & 0xFF == ord('q'):
                break
            
            elif key & 0xFF == ord('s'):
                
                if save_count < 5:  #5장의 이미지를 모읍니다
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    img = np.array(cv2.resize(frame, (IMG_SIZE, IMG_SIZE)))
                    if save_count == 0:
                        img_merge1 = img
                    elif save_count == 1:
                        img_merge2 = img
                    elif save_count == 2:
                        img_merge3 = img
                    elif save_count == 3:
                        img_merge4 = img
                    elif save_count == 4:
                        img_merge5 = img
                        
                    save_count += 1
                    print(str(save_count), '번째 이미지 임시저장...')

                else:
                    
                    #5장의 이미지를 합치기 합니다
                    img_final = cv2.merge((img_merge1,img_merge2,img_merge3,img_merge4,img_merge5))
                    print('\n5장 이미지({})가 모여서 이미지 합치기를 진행합니다...'.format(np.array(img_final).shape))
                    #합쳐진 이미지를 pickle로 저장합니다
                    save_data(img_final)

                    #새로운 사진을 획득 하기 위해 메모리를 초기화 합니다
                    save_count = 0
                    img_merge1 = None
                    img_merge2 = None
                    img_merge3 = None
                    img_merge4 = None
                    img_merge5 = None
                    
                    print('다시 저장할 준비가 되었습니다\n')
                    
                    '''
                    (a,b,c,d,e) = cv2.split(img_final)
                    cv2.imshow('Image_mergea', a)
                    cv2.imshow('Image_mergeb', b)
                    cv2.imshow('Image_mergec', c)
                    '''

        
        # Break the loop
        else:
            break 
        
    # When everything done, release the video capture and video write objects
    cap.release()
        
    # Closes all the frames
    cv2.destroyAllWindows() 