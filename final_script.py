import sys

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.layers import Activation
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras import backend as K
from selenium.webdriver.common.by import By


from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from PIL import ImageDraw, ImageFont
import random
import matplotlib.pyplot as plt
import io
import numpy as np    
from PIL import Image
import time
import cv2
import pandas as pd

import tensorflow as tf
import multiprocessing as mp

batch_size = 4
num_classes = 36 
epochs = 100 

#IIN = "941218350109"
#FIO = "ПОРОХНЯ ВАСИЛИЙ АНДРЕЕВИЧ"
#FIO = "Жилибаев Серик Бибиталиевич"
#FIO = "ШУАКОВА\ ЖАНАРКУЛ\ МАДКЕРИМОВНА"
#FIO = "ВАСИЛЬЕВ\ ЮЛИАН\ ВИКТОРОВИЧ"
#FIO = "КАНСЕИТОВ\ ТАЛГАТ\ СУЮНДУКОВИЧ"
# Create Model
type_inp = sys.argv[1]
text = sys.argv[2]


def base_model():

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32,(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    
    sgd = SGD(lr = 0.001, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model
cnn_n = base_model()

#Load Pretrained Model On Colab
cnn_n.load_weights("ckpt")


WINDOW_SIZE = "1920,1080"

options = webdriver.ChromeOptions()
options.add_argument('--ignore-ssl-errors=yes')
options.add_argument('--ignore-certificate-errors')
options.add_argument("headless")
options.add_argument("--window-size=%s" % WINDOW_SIZE)
driver = webdriver.Chrome("./chromedriver", options=options)
driver.get("https://aisoip.adilet.gov.kz/public/faces/debtorsReestr.jspx?_afrLoop=17617203587200948&_afrWindowMode=0&_adf.ctrl-state=18fvl8dqch_4")

c = 0
rois = []
ww = 32
hh = 32
checker = False
while not checker:
    err = False
    while True:
        time.sleep(2)
        img = None
        try:
            img = driver.find_element_by_xpath("//img[@title='Капча']")
            y = img.location['y']
            driver.execute_script('window.scrollTo(0,'+str(y)+')');
            element_png = img.screenshot_as_png
            print("Captcha copied")
        except:
            err = True
            print("Error with loading Captcha. Start Again...")
            break

        image = Image.open(io.BytesIO(element_png))
        arr = np.asarray(image)
        img = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)
        ctrs, hier = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
        c = 0
        rois = []
        for i, ctr in enumerate(sorted_ctrs):
            x, y, w, h = cv2.boundingRect(ctr)
            area = w*h
            if 120 < area < 500:
                rois.append(img[y-3:y + h +3, x-1:x + w + 2])
                c += 1
        if c == 4:
            try:
                for i in range(len(rois)):
                    rois[i] = cv2.resize(rois[i], dsize=(ww, hh), interpolation=cv2.INTER_CUBIC)
            except:
                err = True
                print("Error with detecting bounding boxes of letters. Starting Again...")
                break
                
                
        if not c == 4:
            time.sleep(3)
            print("Not all characters recognized. Starting Again...")
            driver.refresh()

        else:
            break
    if err:
        driver.refresh()
        continue
    w=10
    h=10
    fig=plt.figure(figsize=(8, 8))
    columns = 4
    rows = 1
    pred = np.array(rois, "float64")
    letters = "abcdefghijklmnopqrstuvwxyz0123456789"
    output = cnn_n.predict(pred)
    res = ""
    print(output)
    for i in output:
        res += letters[int(np.where(i == 1)[0])]
    print(res)

    inp = driver.find_elements(By.XPATH, '//input[@id="pt1:it2::content"]')[0]  
    inp.send_keys(res)
    time.sleep(1)
    try:
	    if type_inp == 'iin':
	    	inp2 = driver.find_element_by_xpath("//input[@id='pt1:itIin::content']")
	    	time.sleep(1)
	    	inp2.clear()
	    	time.sleep(1)
	    	inp2.send_keys(text)
	    elif type_inp == 'fio':
		    inp3 = driver.find_element_by_xpath("//input[@id='pt1:itFion::content']")
		    time.sleep(1)
		    inp3.clear()
		    time.sleep(1)
		    inp3.send_keys(text)
    except:
	    if type_inp == 'iin':
	    	inp2 = driver.find_element_by_xpath("//input[@id='pt1:itIin::content']")
	    	time.sleep(1)
	    	inp2.clear()
	    	time.sleep(1)
	    	inp2.send_keys(text)
	    elif type_inp == 'fio':
		    inp3 = driver.find_element_by_xpath("//input[@id='pt1:itFion::content']")
		    time.sleep(1)
		    inp3.clear()
		    time.sleep(1)
		    inp3.send_keys(text)

    time.sleep(1)
    button = driver.find_element_by_xpath("//button[@id='pt1:buttonSearch']")
    button.click()
    
    time.sleep(2)
    #tds = driver.find_elements(By.XPATH, '//span[@style="color:red;font-size:small;"]')
    try:
        el = driver.find_element_by_xpath("//span[@style='color:red;font-size:small;']")
        print(el.get_attribute("innerHTML"))
        print("Captcha Wrong Entry. Start Again")
        driver.refresh()
    except:
        checker = True
        
    #
    #
    #
time.sleep(1)
tables=WebDriverWait(driver,20).until(EC.presence_of_all_elements_located((By.XPATH,"//span[@id='pt1:pgl4']")))[0]

df = pd.read_html(tables.get_attribute('innerHTML'))
df = df[0]

driver.close()

print(df.to_string())