from model import *
from sklearn.metrics import classification_report
import numpy as np
from data_helper import *
import matplotlib.pyplot as plt


word2id_path = "./data/word2id.json"
label2id_path = "./data/label2id.json"
max_seq_len = 200
batch_size = 50
epochs = 30
checkpoint_save_path = "./checkpoint_trans/model.ckpt"
train_num = 100000000000

res = {}
print("TRAIN NUM: {}".format(train_num))
# 加载需要迁移学习的数据数据
x_train, y_train, x_test, y_test, x_dev, y_dev = transfor_cnews()

# 加载预训练模型
model_path = "./checkpoint/model.ckpt"
model = TextCnn(vocab_size=6000, embed_size=300)
model.load_weights(model_path)
len_layers = len(model.layers)

# 更改模型输出层，冻结其他层
l_layer = len(model.layers)
new_model = tf.keras.models.Sequential(model.layers[0:(l_layer-1)])
new_output = tf.keras.layers.Dense(33, activation='softmax')
new_model.add(new_output)

# 控制哪些层重新训练
# for i in range(l_layer-1):
#     new_model.layers[i].trainable = False

new_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=False,
                                                 save_best_only=True,
                                                 monitor='val_loss',
                                                 verbose=1)

history = new_model.fit(x_train, y_train,
                    validation_data=(x_dev, y_dev),
                    batch_size=batch_size, epochs=epochs,  callbacks=[cp_callback])

new_model.summary()
# new_model.evaluate(x_test, y_test)
pred_prob = new_model.predict(x_test)
pred_prob = np.argsort(pred_prob, axis=1).tolist()

top1 = []
top3 = []
for p, t in zip(pred_prob, y_test.tolist()):
    top1.append(p[-1])
    if t in p[-3:]:
        top3.append(t)
    else:
        top3.append(p[-1])

print("top1: ")
t_top1 = classification_report(y_test.tolist(), top1)
print(t_top1)

print("top3: ")
t_top3 = classification_report(y_test.tolist(), top3)
print(t_top3)


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
# plt.show()
plt.savefig("./train_trans.png")