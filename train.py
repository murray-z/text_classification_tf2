# -*- coding: utf-8 -*-

from model import *
from data_helper import *
import matplotlib.pyplot as plt
import sys

model_dict = {'cnn': TextCnn,
              'mulit_kernel_cnn': TextMultiKernalCnn,
              'bilstm': TextBiLSTM,
              'bigru': TextBiGRU,
              'cnn_lstm': TextCnnLSTM,
              'cnn_gru': TextCnnGRU,
              'bilstm_attention': TextBilstmAttention,
              'cnn_attention': TextCnnAttention,
              'transformer': Transformer}

model_type = 'cnn_attention'

if model_type not in model_dict:
    print("model_type must in {}".format(list(model_dict.keys())))
    sys.exit()

max_seq_len=100
vocab_size=6000
embed_size=100
class_num=10
checkpoint_save_path = "./checkpoint/model.ckpt"
batch_size=50
epochs=3

x_train_out = "./data/x_train.npy"
y_train_out = "./data/y_train.npy"
x_test_out = "./data/x_test.npy"
y_test_out = "./data/y_test.npy"
x_dev_out = "./data/x_dev.npy"
y_dev_out = "./data/y_dev.npy"
word2id_path = "./data/word2id.json"
label2id_path = "./data/label2id.json"

# 加载cnews数据
x_train, y_train, x_test, y_test, x_dev, y_dev = np.load(x_train_out), np.load(y_train_out), np.load(x_test_out),\
    np.load(y_test_out), np.load(x_dev_out), np.load(y_dev_out)

if model_type == 'transformer':
    model = model_dict[model_type](maxlen=max_seq_len, vocab_size=vocab_size, embed_size=embed_size, num_heads=10, ff_dim=32, class_num=class_num)
else:
    model = model_dict[model_type](vocab_size=vocab_size, embed_size=embed_size, class_num=class_num)

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=False,
                                                 save_best_only=True,
                                                 monitor='val_loss',
                                                 verbose=1)

history = model.fit(x_train, y_train,
                    validation_data=(x_dev, y_dev),
                    batch_size=batch_size, epochs=epochs,  callbacks=[cp_callback])

model.evaluate(x_test, y_test, batch_size=batch_size)

model.summary()



# 显示训练集和验证集的acc和loss曲线
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
plt.savefig("./train.png")