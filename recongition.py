import cv2
import numpy as np
import tensorflow as tf

model =  tf.keras.models.load_model('my_model.h5')

def fix_dimension(img): 
    new_img = np.zeros((28,28,3))
    for i in range(3):
        new_img[:,:,i] = img
    return new_img
  
def show_results(char):
    dic = {}
    characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i,c in enumerate(characters):
        dic[i] = c

    output = []
    for i,ch in enumerate(char): 
        img_ = cv2.resize(ch, (28,28), interpolation=cv2.INTER_AREA)
        img = fix_dimension(img_)
        img = img.reshape(1,28,28,3) 
        y_ = model.predict(img)[0]
        character = dic[np.argmax(y_)]
        output.append(character)
        
    plate_number = ''.join(output)
    
    return plate_number