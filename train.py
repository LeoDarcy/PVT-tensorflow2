# # from __future__ import print_function
import os
os.environ['CUDA_VISIBLE_DEVICES']='1' 
import argparse
import tensorflow as tf
import logging
logging.getLogger().setLevel(logging.INFO)
from utils.lr_scheduler import get_lr_scheduler
from model.model_builder import get_model
from generator.generator_builder import get_generator
from utils.optimizers import get_optimizers
import sys
import os

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def parse_args(args):
    parser = argparse.ArgumentParser(description='Simple training script for using PVT .')

    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--batch-size', default=4, type=int)
    parser.add_argument('--dataset', default='laval', type=str, help="choices=['cifar10,cifar100,laval']")#dataset/RockPaperScissor
    parser.add_argument('--model', default='PVT-tiny', type=str, help="choices=['PVT-tiny','PVT-small','PVT-medium','PVT-large',"
                                                                      "'ResNet50','ResNet101','EfficientNetB0','CifarResNet']")
    parser.add_argument('--pretrain', default=None, help="choices=[None,'imagenet','resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5']")
    parser.add_argument('--img-size', default=32, type=int)
    parser.add_argument('--augment', default='custom_augment', type=str, help="choices=['rand_augment','auto_augment','cutmix','mixup','custom_augment',]")
    parser.add_argument('--concat-max-and-average-pool', default=False, type=bool,help="Use concat_max_and_average_pool layer in model")
    parser.add_argument('--lr-scheduler', default='cosine', type=str, help="choices=['step','cosine']")
    parser.add_argument('--init-lr', default=1e-3, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--lr-decay', default=0.1, type=float)
    parser.add_argument('--lr-decay-epoch', default=[80, 150, 180], type=int)
    parser.add_argument('--warmup-lr', default=1e-4, type=float)
    parser.add_argument('--warmup-epochs', default=50, type=int)
    parser.add_argument('--weight-decay', default=0, type=float)
    parser.add_argument('--optimizer', default='adam', help="choices=['adam','sgd']")
    parser.add_argument('--checkpoints', default='./checkpoints')
    return parser.parse_args(args)

def main(args):

    train_generator, val_generator = get_generator(args)
    '''
    for epoch in range(2):
        for batch_number, (x, l1, l2, l10) in enumerate(train_generator):
            print('正在进行第{} batch'.format(batch_number))
            if batch_number < 10:
                print('x_batch:', x.shape)#batch_size, 192, 256, 3
                print('y_batcxh:', l1.shape, l2.shape,l10.shape)
                assert(1==0)
        print('一个epoch结束=============================')
        assert(1==0)
    '''
    model = get_model(args)
    loss_object = {'out1': 'mse','out2': 'mse',
          'out10': 'mse'}
    #loss_object = tf.keras.losses.CategoricalCrossentropy()
    optimizer = get_optimizers(args)
    lr_scheduler = get_lr_scheduler(args)
    if not os.path.exists(args.checkpoints):
        os.makedirs(args.checkpoints)
    # lr_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20, verbose=1, min_lr=0)
    lr_cb = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
    model_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(filepath=args.checkpoints+'/best_weight_{epoch}_{val_loss:.3f}.h5',
                                                             monitor='val_loss',mode='min',
                                                             verbose=1,save_best_only=True,save_weights_only=True)
    tensorbd = tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1)
    #earlystop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
    cbs=[lr_cb,
         model_checkpoint_cb,
         tensorbd#,earlystop
         ]
    #model.load_weights('/root/bjy/SHNet/PVT-tensorflow2/checkpoints/best_weight_20_0.024.h5')
    model.compile(optimizer,loss_object, metrics=['mse'])
    model.fit(train_generator,
                          validation_data=val_generator,
                          epochs=args.epochs,
                          callbacks=cbs,
                          verbose=1,
                          )

if __name__== "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
