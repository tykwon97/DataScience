'''
MNIST: 미국표준기술연구소에 공개한 필기체 숫자에 대한 데이터베이스
'''

import numpy as np
import matplotlib.pyplot as plt
    
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten
from keras.utils import to_categorical
from keras.datasets import mnist  # mnist 불러오기


def plot_loss_curve(history):

    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 10))

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()   

def train_mnist_model():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    '''
    print("X_train.shape=", X_train.shape)
    print("y_train.shape", y_train.shape)
    print("X_test.shape=", X_test.shape)
    print("y_test.shape", y_test.shape)
    print(y_train[1])
    plt.imshow(X_train[1], cmap='gray') #글자 확인
    '''

    #cnn포맷에 맞게 3차원으로 바꿔준다 + 채널이 1개다 (rgb인 경우 1이 아닌 3)
    X_train = X_train.reshape(60000, 28, 28, 1)
    X_test = X_test.reshape(10000, 28, 28, 1) 
    
    #print(y_train[0])
    '''
    softmax layer -> output=10개의 노드. 각각이 0부터 9까지 숫자를 대표하는 클래스 
                  -> 확률로 표현 총 합은 1
    이를 위해서 y값을 one-hot encoding 표현법으로 변환
    0: 1,0,0,0,0,0,0,0,0,0
    1: 0,1,0,0,0,0,0,0,0,0
    ...
    5: 0,0,0,0,0,1,0,0,0,0
    '''
    
    y_train = to_categorical(y_train) # one-hot encoding, 10개의 노드로 만들기
    y_test = to_categorical(y_test)
    
    #print(y_train[0])
    
    
    model = Sequential([
                Input(shape=(28,28,1), name='input_layer'),
                
                # n_filters * (filter_size + 1) = 32*(9+1) = 320
                Conv2D(32, kernel_size=3, activation='relu', name='conv_layer1'), #kernel개수 32 , filter 3x3 size
                
                
                #Conv2D(64, kernel_size=3, activation='relu', name='conv_layer1'),
                
                #Dropout(0.5) #50%만 loop돌게 한다. + 오버피팅 완화
                MaxPooling2D(pool_size=2), # 2x2
                Flatten(), # 다차원을 1차원으로 바꿔준다. 13 * 13 * 32 = 5408
                #Dense(20, activation='softmax', name='output_layer')
                Dense(10, activation='softmax', name='output_layer')
            ])

    model.summary()    
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #multi class classification (loss) = categorical_crossentropy
    
    
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=10, epochs=3)
    #bath_size 몇개의 data를 보고 편미분값을 업데이트 할지
    #60000개 데이터 batch_size-10이면 한번에 6000번 돈다.
    
    plot_loss_curve(history.history)
    print(history.history)
    print("train loss=", history.history['loss'][-1])
    print("validation loss=", history.history['val_loss'][-1])    
    
    model.save('mnist.model') # keras에 파일로 모델 저장하는 방법
    
    return model

def predict_image_sample(model, X_test, y_test, test_id=-1):
    if test_id < 0:
        from random import randrange
        test_sample_id = randrange(10000)
    else:
        test_sample_id = test_id
        
    test_image = X_test[test_sample_id]
    
    plt.imshow(test_image, cmap='gray')
    
    test_image = test_image.reshape(1,28,28,1)

    y_actual = y_test[test_sample_id]
    print("y_actual number=", y_actual)
    
    y_pred = model.predict(test_image)
    print("y_pred=", y_pred) #10차원 확률 값
    y_pred = np.argmax(y_pred, axis=1)[0] # argmax, 첫번째 축에서 최대가 되는 인덱스 값 읽기
    print("y_pred number=", y_pred)
    
    '''#틀린 번호 메모장에 저장
    if y_pred != y_actual:
        print("sample %d is wrong!" %test_sample_id)
        with open("wrong_samples.txt", "a") as errfile:
            print("%d"%test_sample_id, file=errfile)
    else:
        print("sample %d is correct!" %test_sample_id)
    '''    

if __name__ == '__main__':
    #train_mnist_model()
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    model = load_model('mnist.model') #미리 저장한 모델 로딩(디렉터리 이름)
    predict_image_sample(model, X_test, y_test)
    #predict_image_sample(model, X_test, y_test,test_id=947)
    #for i in range(500):
    #    predict_image_sample(model, X_test, y_test)
    