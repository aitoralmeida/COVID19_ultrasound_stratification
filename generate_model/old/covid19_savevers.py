import os
from random import shuffle, sample

from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
import tensorflow.keras.applications.mobilenet as mobilenet

import numpy as np
from sklearn import metrics

PATH = '/croppedi2p0/'
TEST_SET_PATIENTS = ['04_','09_','18_','21_','27_','36_','38_','41_','51_','55_','59_','60_'] 
train_test_divide = 0.75
EPOCHS = 10

def create_sets(positive, negative):
  files_covid= os.listdir(PATH)
  total_files = len(files_covid)
  print ('Total files in disk:', total_files)
  
  #randomize the files
  shuffle(files_covid)

  #sample to avoid mem problems (temporal)
  # files_covid = sample(files_covid, 6500)

  #find positive and negative files
  print('*'*10)
  print('Separating posititive and negative files...')
  print('Positive token:', positive)
  print('Negative token', negative)
  positive_files = []
  negative_files = []
  for name in files_covid:
    if negative in name:
      negative_files.append(name) 
    elif positive in name:
      positive_files.append(name) 
  total_positive = len(positive_files)
  print ('Total positive files:', total_positive)
  total_negative = len(negative_files)
  print ('Total negative files:', total_negative)
  #sanity check
  print('>>>>>Sanity check...')
  print ('Expected total files:', total_files)
  print ('Total files positive+negative:', total_positive+total_negative)

  #calculating splits
  #train
  total_train_pos = int(total_positive * train_test_divide)
  total_train_neg = int(total_negative * train_test_divide)
  print('*'*10) 
  print('Calculating splits...')
  print('Training positive:', total_train_pos)
  print('Training positive percentage:', float(total_train_pos/(total_train_pos+total_train_neg)))
  print('Training negative:', total_train_neg)
  print('Training negative percentage:', float(total_train_neg/(total_train_pos+total_train_neg)))
  total_train = total_train_pos+total_train_neg
  print('Training total:', total_train)
  #val
  test_pos = total_positive - total_train_pos
  test_neg = total_negative - total_train_neg
  test_total = test_pos + test_neg
  print('Test positive:', test_pos)
  print('Test positive percentage:', float(test_pos/test_total))
  print('Test negative:', test_neg)
  print('Test negative percentage:', float(test_neg/test_total))
  print('Test total:', test_total)
  #sanity check
  print('>>>>>Sanity check...')
  print('Target divide perecentage:', train_test_divide)
  print('Train percentage', (float)(total_train/(total_train+test_total)))
  print('Test percentage', (float)(test_total/(total_train+test_total)))
  print ('Expected total files::', total_files)
  print ('Total files train+val:', total_train+test_total)

  #HASTA AQUI BIEN
  
  print('*'*10)
  print('Loading file names...')
  print('Total positive', len(positive_files))
  print('Total negative', len(negative_files))
  print('Expected train pos:', total_train_pos)
  print('Expected train neg:', total_train_neg)
  #train
  train_positive_filenames = positive_files[:total_train_pos]
  train_negative_filenames = negative_files[:total_train_neg]
  train_files = train_positive_filenames + train_negative_filenames
  #sanity check
  print('>>>>>Sanity check...')
  print('Expected train positive:', total_train_pos)
  print('Actual train positive:', len(train_positive_filenames))
  print('Expected train negative:', total_train_neg)
  print('Actual train negative:', len(train_negative_filenames))
  print('Expected train:', total_train)
  print('Actual files in train_files:', len(train_files))
  #val
  val_positive_filenames = positive_files[total_train_pos:]
  val_negative_filenames = negative_files[total_train_neg:]
  val_files = val_positive_filenames + val_negative_filenames
  #sanity check
  print('>>>>>Sanity check...')
  print('Expected val positive:', test_pos)
  print('Actual val positive:', len(val_positive_filenames))
  print('Expected val negative:', test_neg)
  print('Actual val negative:', len(val_negative_filenames))
  print('Expected val:', test_total)
  print('Actual files in val_files:', len(val_files))
  
    #train_files = positive_files[:total_train_pos] + negative_files[:total_train_neg]
  #val_files = positive_files[total_train_pos:]  + negative_files[total_train_neg:]

    
  shuffle(train_files)
  shuffle(val_files)
  #loading images
  print('Loading train and val images...')
  # Train
  print ('Processing training data...')
  X_train = []
  X_train_names = []
  y_train = []
  fail_train = []
  file_processed = 0
  for filename in train_files:
    file_processed += 1
    if file_processed % 300 == 0:
      print('Processing ', file_processed, 'of', len(train_files))
    if positive in filename: 
      y_train.append([1,0])
    elif negative in filename:
      y_train.append([0,1])
    else: #wrong filename
      fail_train.append(filename)
    img = image.load_img(PATH+filename, target_size=(224, 224)) #mobilenet
    x = image.img_to_array(img)
    x = mobilenet.preprocess_input(x) #mobilenet
    X_train.append(x)
  #sanity check
  print('Sanity check...')
  print('X_train total:', len(X_train))
  print('y_train total:', len(y_train))
  print('fail_train total:', len(fail_train))
  print(fail_train)

  #val
  print ('Processing validation data...')
  X_val = []
  X_val_names = []
  y_val = []
  fail_val = []
  file_processed = 0
  for filename in val_files:
    file_processed += 1
    if file_processed % 300 == 0:
      print('Processing ', file_processed, 'of', len(val_files))
    if positive in filename: 
      y_val.append([1,0])
    elif negative in filename:
      y_val.append([0,1])
    else: #wrong filename
      fail_val.append(filename)
    img = image.load_img(PATH+filename, target_size=(224, 224)) #mobilenet
    x = image.img_to_array(img)
    x = mobilenet.preprocess_input(x) #mobilenet
    X_val.append(x)

  #sanity check
  print('Sanity check...')
  print('X_val total:', len(X_val))
  print('y_val total:', len(y_val))
  print('fail_val total:', len(fail_val))
  print(fail_val)

  X_train = np.array(X_train)
  y_train = np.array(y_train)
  X_val = np.array(X_val)
  y_val = np.array(y_val)  
  print('Shapes train')
  print(X_train.shape)
  print(y_train.shape)
  print('Shapes val')
  print(X_val.shape)
  print(y_val.shape)
  return X_train, y_train, X_val, y_val




def create_sets_by_patients(positive, negative):
  files_covid= os.listdir(PATH)
  total_files = len(files_covid)
  print ('Total files in disk:', total_files)

  train_files = []
  val_files = []

  for filename in files_covid:
    if any(x in filename for x in TEST_SET_PATIENTS):
      val_files.append(filename)
    else:
      train_files.append(filename)

  print('Total train files:', len(train_files))
  print('Total test files:', len(val_files))
 
  #loading images
  print('Loading train and val images...')
  # Train
  print ('Processing training data...')
  X_train = []
  X_train_names = []
  y_train = []
  fail_train = []
  file_processed = 0
  for filename in train_files:
    file_processed += 1
    if file_processed % 300 == 0:
      print('Processing ', file_processed, 'of', len(train_files))
    if positive in filename: 
      y_train.append([1,0])
    elif negative in filename:
      y_train.append([0,1])
    else: #wrong filename
      fail_train.append(filename)
    img = image.load_img(PATH+filename, target_size=(224, 224)) #mobilenet
    x = image.img_to_array(img)
    x = mobilenet.preprocess_input(x) #mobilenet
    X_train.append(x)
    X_train_names.append(filename)
  #sanity check
  print('Sanity check...')
  print('X_train total:', len(X_train))
  print('y_train total:', len(y_train))
  print('fail_train total:', len(fail_train))
  print(fail_train)
  
  #val
  print ('Processing validation data...')
  X_val = []
  X_val_names = []
  y_val = []
  fail_val = []
  file_processed = 0
  test_pos_total = 0
  test_neg_total = 0
  for filename in val_files:
    file_processed += 1
    if file_processed % 300 == 0:
      print('Processing ', file_processed, 'of', len(val_files))
    if positive in filename: 
      y_val.append([1,0])
      test_pos_total += 1
    elif negative in filename:
      y_val.append([0,1])
      test_neg_total += 1
    else: #wrong filename
      fail_val.append(filename)
    img = image.load_img(PATH+filename, target_size=(224, 224)) #mobilenet
    x = image.img_to_array(img)
    x = mobilenet.preprocess_input(x) #mobilenet
    X_val.append(x)
    X_val_names.append(filename)

  #sanity check
  print('Sanity check...')
  print('X_val total:', len(X_val))
  print('y_val total:', len(y_val))
  print('fail_val total:', len(fail_val))
  print(fail_val)
  print('Test positive examples:', test_pos_total)
  print((float)(test_pos_total/len(y_val)))
  print('Test negative examples:', test_neg_total)
  print((float)(test_neg_total/len(y_val)))


  X_train = np.array(X_train)
  y_train = np.array(y_train)
  X_val = np.array(X_val)
  y_val = np.array(y_val)  
  print('Shapes train')
  print(X_train.shape)
  print(y_train.shape)
  print('Shapes val')
  print(X_val.shape)
  print(y_val.shape)
  return X_train, y_train, X_val, y_val
  
# get the data
print('***** Load files...')
X_train, y_train, X_val, y_val = create_sets('N0', 'N1')
# get the model without the denses
base_model = mobilenet.MobileNet(weights='imagenet', include_top='false')
new_dense = base_model.output
# add the new denses to classify the hate images
new_dense = Dense(1024, activation='relu')(new_dense)
predictions = Dense(2, activation='softmax')(new_dense)
model = Model(inputs=base_model.input, outputs=predictions)
# we will only train the new denses for the baseline
for layer in base_model.layers:
  layer.trainable = False
model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ["accuracy"])
results = model.fit(X_train, y_train, epochs= EPOCHS, batch_size = 16, validation_data = (X_val, y_val))
model.save('/results/covid19_model')
