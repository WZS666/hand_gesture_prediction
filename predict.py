import cv2
import json
import random
import pygame
import sys, os
import operator
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json

# 使用pygame來建一個預測視窗
pygame.init()
screen = pygame.display.set_mode((400,400),pygame.RESIZABLE)
CLIP_X1,CLIP_Y1,CLIP_X2,CLIP_Y2 = 160,140,400,360

# 讀訓練好的model
with open('model_trained.json','r') as f:
    model_json = json.load(f)
loaded_model = model_from_json(model_json)
loaded_model.load_weights('model_trained.h5')

cap = cv2.VideoCapture(0) # 擷取鏡頭畫面
i = 0 # 用來記錄之後要新增的image
wzs = 158 # 調整二值化的閥值變數
image_q = cv2.THRESH_BINARY # 調整二值化的模式

while True:
    _, FrameImage = cap.read() # 讀取鏡頭畫面
    FrameImage = cv2.flip(FrameImage, 1) # 圖像水平翻轉
    cv2.imshow("", FrameImage) # 顯示鏡頭畫面
    cv2.rectangle(FrameImage, (CLIP_X1, CLIP_Y1), (CLIP_X2, CLIP_Y2), (0,255,0) ,1) # 框出ROI位置

    ROI = FrameImage[CLIP_Y1:CLIP_Y2, CLIP_X1:CLIP_X2] # ROI的大小
    ROI = cv2.resize(ROI, (128, 128))  # ROI resize
    ROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY) # ROI 轉灰階
    _, output = cv2.threshold(ROI, wzs, 255, image_q) # Threshold Binary：即二值化，將大於閾值的灰度值設為最大灰度值，小於閾值的值設為0。
    
    SHOWROI = cv2.resize(ROI, (256, 256)) # ROI resize
    _, output2 = cv2.threshold(SHOWROI, wzs, 255, image_q) # Black Background is better for prediction
    cv2.imshow("ROI", output2)

    # 需要存dataset時，可以使用下面這三行進行取樣
    # cv2.imwrite('./test/handdata'+str(i)+'.jpg',output2)
    # i += 1
    # cv2.waitKey(100)

    
    result = loaded_model.predict(output.reshape(1,128, 128, 1)) # 若是訓練彩色，則須把1改成3個channel (1,128,128,3) 配合model輸入的dimension
    predict =   { '1':    result[0][0],
                  '2':    result[0][1],    
                  '3':    result[0][2],
                  '4':    result[0][3],
                  '5':    result[0][4],
                  }
    print(predict)
    predict = sorted(predict.items(), key=operator.itemgetter(1), reverse=True) # 分數較高者會sort至第一位

    # 這邊是取對應預測的image，沒有則是顯示nosign
    if(predict[0][1] == 1.0):
        predict_img  = pygame.image.load(os.getcwd() + '/dataset/' + predict[0][0] + '.jpg')
    else:
        predict_img  = pygame.image.load(os.getcwd() + '/dataset/nosign.png')
    predict_img = pygame.transform.scale(predict_img, (400, 400))
    screen.blit(predict_img, (0,0))
    pygame.display.flip()

    # 在ROI上觸及英數字，來反饋效果
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == ord('l'): # lower wzs quality
      wzs = wzs - 5
    elif interrupt & 0xFF == ord('u'): # upper wzs quality
      wzs = wzs + 5
    elif interrupt & 0xFF == ord ('s'): # save dataset
      cv2.imwrite('handdata'+str(random.randint(1,9999))+'.jpg',output2)
    elif interrupt & 0xFF == ord ('c'): # change THRESH_BINARY TO THRESH_BINARY_INV
      if image_q == cv2.THRESH_BINARY_INV:
        image_q = cv2.THRESH_BINARY
      else:
        image_q = cv2.THRESH_BINARY_INV
    if interrupt & 0xFF == ord('q'): # esc key
        break
            
pygame.quit()
cap.release()
cv2.destroyAllWindows()
