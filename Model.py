""" common model for DCGAN """
import logging

import cv2
import neuralgym as ng
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope

from neuralgym.models import Model
from neuralgym.ops.summary_ops import scalar_summary, images_summary
from neuralgym.ops.summary_ops import gradients_summary
from neuralgym.ops.layers import flatten, resize
from neuralgym.ops.gan_ops import gan_wgan_loss, gradients_penalty
from neuralgym.ops.gan_ops import random_interpolates

from inpaint_ops import gen_conv, gen_deconv, dis_conv
from inpaint_ops import random_bbox, bbox2mask, local_patch
from inpaint_ops import spatial_discounting_mask
from inpaint_ops import resize_mask_like, contextual_attention,attention,str2bool
from predictor import  resfcn256
from util.util import *

logger = logging.getLogger()


class InpaintCAModel(Model):
    def __init__(self):
        super().__init__('InpaintCAModel')

    def build_inpaint_net(self, x, mask, config=None, reuse=False,
                          training=True, padding='SAME', name='inpaint_net'):
        """Inpaint network.

        Args:
            x: incomplete image, [-1, 1]
            mask: mask region {0, 1}
        Returns:
            [-1, 1] as predicted image
        """
        xin = x #shape=(16,256,256,3)
        offset_flow = None
        ones_x = tf.ones_like(x)[:, :, :, 0:1] #shape=(16,256,256,1)
        x = tf.concat([x, ones_x, ones_x*mask], axis=3)#shape=(16,256,256,5)

        # two stage network
        cnum = 32
        with tf.variable_scope(name, reuse=reuse), \
                arg_scope([gen_conv, gen_deconv],
                          training=training, padding=padding):
            # stage1
            x = gen_conv(x, cnum, 5, 1, name='conv1')#shape =(16,256,256,32)
            x = gen_conv(x, 2*cnum, 3, 2, name='conv2_downsample')#shape=(16,128,128,64)
            x = gen_conv(x, 2*cnum, 3, 1, name='conv3')#shape=(16,128,128,64)
            x = gen_conv(x, 4*cnum, 3, 2, name='conv4_downsample')#shape=(16,64,64,128)
            x = gen_conv(x, 4*cnum, 3, 1, name='conv5')#shape=(16,64,64,128)
            x = gen_conv(x, 4*cnum, 3, 1, name='conv6')#shape=(16,64,64,128)
            mask_s = resize_mask_like(mask, x)#shape=(1,64,64,1)
            x = gen_conv(x, 4*cnum, 3, rate=2, name='conv7_atrous')#shape=(16,64,64,128)
            x = gen_conv(x, 4*cnum, 3, rate=4, name='conv8_atrous')#shape=(16,64,64,128)
            x = gen_conv(x, 4*cnum, 3, rate=8, name='conv9_atrous')#shape=(16,64,64,128)
            x = gen_conv(x, 4*cnum, 3, rate=16, name='conv10_atrous')#shape=(16,64,64,128)
            x = gen_conv(x, 4*cnum, 3, 1, name='conv11')#shape=(16,64,64,128)
            x = gen_conv(x, 4*cnum, 3, 1, name='conv12')#shape=(16,64,64,128)
            x = gen_deconv(x, 2*cnum, name='conv13_upsample')#shape=(16,128,128,64)
            x = gen_conv(x, 2*cnum, 3, 1, name='conv14')#shape=(16,128,128,64)
            x = gen_deconv(x, cnum, name='conv15_upsample')#shape=(16,256,256,128)
            x = gen_conv(x, cnum//2, 3, 1, name='conv16')#shape=(16,256,256,16)
            x = gen_conv(x, 3, 3, 1, activation=None, name='conv17')#shape=(16,256,256,3)
            x = tf.clip_by_value(x, -1., 1.)#shape=(16,256,256,3)
            x_stage1 = x#shape=(16,256,256,3)
            # return x_stage1, None, None

            # stage2, paste result as input
            # x = tf.stop_gradient(x)
            x = x*mask + xin*(1.-mask)#shape=(16,256,256,3)
            x.set_shape(xin.get_shape().as_list())#shape=(16,256,256,3)
            # conv branch
            xnow = tf.concat([x, ones_x, ones_x*mask], axis=3)#shape=(16,256,256,5)
            x = gen_conv(xnow, cnum, 5, 1, name='xconv1')#shape=(16,256,256,32)
            x = gen_conv(x, cnum, 3, 2, name='xconv2_downsample')#shape=(16,128,128,32)
            x = gen_conv(x, 2*cnum, 3, 1, name='xconv3')#shape=(16,128,128,64)
            x = gen_conv(x, 2*cnum, 3, 2, name='xconv4_downsample')#shape=(16,64,64,64)
            x = gen_conv(x, 4*cnum, 3, 1, name='xconv5')#shape=(16,64,64,128)
            x = gen_conv(x, 4*cnum, 3, 1, name='xconv6')#shape=(16,64,64,128)
            x = gen_conv(x, 4*cnum, 3, rate=2, name='xconv7_atrous')#shape=(16,64,64,128)
            x = gen_conv(x, 4*cnum, 3, rate=4, name='xconv8_atrous')#shape=(16,64,64,128)
            x = gen_conv(x, 4*cnum, 3, rate=8, name='xconv9_atrous')#shape=(16,64,64,128)
            x = gen_conv(x, 4*cnum, 3, rate=16, name='xconv10_atrous')#shape=(16,64,64,128)
            x_hallu = x #16*64*64*128
            # attention branch
            x = gen_conv(xnow, cnum, 5, 1, name='pmconv1') #16*256*256*32
            x = gen_conv(x, cnum, 3, 2, name='pmconv2_downsample') #16*128*128*32
            x = gen_conv(x, 2*cnum, 3, 1, name='pmconv3')#16*128*128*64
            x = gen_conv(x, 4*cnum, 3, 2, name='pmconv4_downsample')#16*64*64*128
            x = gen_conv(x, 4*cnum, 3, 1, name='pmconv5')#16*64*64*128
            x = gen_conv(x, 4*cnum, 3, 1, name='pmconv6',#16*64*64*128
                        activation=tf.nn.relu)
            x, offset_flow = contextual_attention(x, x, mask_s, 3, 1, rate=2)
            # x = attention(x_hallu,128,sn=str2bool,scope='attention')#16*64*64*128  #128
            x = gen_conv(x, 4*cnum, 3, 1, name='pmconv9')#16*64*64*128
            x = gen_conv(x, 4*cnum, 3, 1, name='pmconv10')#16*64*64*128
            pm = x
            #prior branch
            with tf.variable_scope('prior'):
                y = conv2d(x_stage1, 'conv1', 64, bn=True, is_training= False, activation=True, ksize=7, stride=2)#shape=(16,128,128,64)
                for i in range(3):
                    y = resblock(y,128, False,'res'+str(i))
                y = hour_glass(y,128,4,False,name='hourglass1')#(16,128,128,128)
                y = conv2d(y,'conv2',128, bn=True, is_training=False, activation=True)#16*128*128*128
                y = hour_glass(y,128,4,False,name='hourglass2')#16*128*128*128
                y1 = conv2d(y,'conv3',68,ksize=1)#16*128*128*68
                y2 = conv2d(y,'conv4',11,ksize=1)#16*128*128*11
                y = tf.concat([y1,y2],axis=-1)#16*128*128*79
                y = gen_conv(y, 79, 3, 2, name='prior_downsample')  # shape=(16,64,64,64)


            x = tf.concat([x_hallu, pm, y], axis=3)#16*64*64*256
            with tf.variable_scope('concat'):
                x = gen_conv(x, 4*cnum, 3, 1, name='allconv11')#16*64*64*128
            x = gen_conv(x, 4*cnum, 3, 1, name='allconv12')#16*64*64*128
            x = gen_deconv(x, 2*cnum, name='allconv13_upsample')#16*128*128*64
            # deconve prior
            y2_1 = gen_conv(y2,cnum,3,1, name = 'prior_decoder_conv1')
            y2_2 = gen_conv(y2_1, 2 * cnum, 3, 1, name='prior_decoder_conv2')
            y2_3 = gen_conv(y2_2, cnum, 3, rate=2, name='prior_decoder_conv3')
            y2_4 = gen_conv(y2_3, 2 * cnum, 3, rate=4, name='prior_decoder_conv4')
            y2_5 = gen_conv(y2_2, cnum, 3, rate=2, name='prior_decoder_conv5')
            y2_6 = gen_conv(y2_5, 2 * cnum, 3, rate=4, name='prior_decoder_conv6')
            x = tf.multiply(x,y2_4) + y2_6 + x

            x = gen_conv(x, 2*cnum, 3, 1, name='allconv14')#16*128*128*64
            x = gen_deconv(x, cnum, name='allconv15_upsample')#16*256*256*32
            x = gen_conv(x, cnum//2, 3, 1, name='allconv16')#16*256*256*16
            x = gen_conv(x, 3, 3, 1, activation=None, name='allconv17')#16*256*256*3
            x_stage2 = tf.clip_by_value(x, -1., 1.)#16*256*256*128
            offset_flow = None
        return x_stage1, x_stage2, offset_flow

    def build_wgan_local_discriminator(self, x, reuse=False, training=True):
        with tf.variable_scope('discriminator_local', reuse=reuse):
            cnum = 64
            x = dis_conv(x, cnum, name='conv1', training=training)
            x = dis_conv(x, cnum*2, name='conv2', training=training)
            x = dis_conv(x, cnum*4, name='conv3', training=training)
            x = dis_conv(x, cnum*8, name='conv4', training=training)
            x = flatten(x, name='flatten')
            return x

    def build_wgan_global_discriminator(self, x, reuse=False, training=True):
        with tf.variable_scope('discriminator_global', reuse=reuse):
            cnum = 64
            x = dis_conv(x, cnum, name='conv1', training=training)
            x = dis_conv(x, cnum*2, name='conv2', training=training)
            x = dis_conv(x, cnum*4, name='conv3', training=training)
            x = dis_conv(x, cnum*4, name='conv4', training=training)
            x = flatten(x, name='flatten')
            return x

    def build_wgan_discriminator(self, batch_local, batch_global,
                                 reuse=False, training=True):
        with tf.variable_scope('discriminator', reuse=reuse):
            dlocal = self.build_wgan_local_discriminator(
                batch_local, reuse=reuse, training=training)
            dglobal = self.build_wgan_global_discriminator(
                batch_global, reuse=reuse, training=training)
            dout_local = tf.layers.dense(dlocal, 1, name='dout_local_fc')
            dout_global = tf.layers.dense(dglobal, 1, name='dout_global_fc')
            return dout_local, dout_global

    def build_graph_with_losses(self, batch_data, config, training=True,
                                summary=False, reuse=False):
        batch_pos = batch_data / 127.5 - 1.
        # generate mask, 1 represents masked point
        bbox = random_bbox(config)
        mask = bbox2mask(bbox, config, name='mask_c')
        batch_incomplete = batch_pos*(1.-mask)
        x1, x2, offset_flow = self.build_inpaint_net(
            batch_incomplete, mask, config, reuse=reuse, training=training,
            padding=config.PADDING)
        if config.PRETRAIN_COARSE_NETWORK:
            batch_predicted = x1
            logger.info('Set batch_predicted to x1.')
        else:
            batch_predicted = x2
            logger.info('Set batch_predicted to x2.')
        losses = {}
        # apply mask and complete image
        batch_complete = batch_predicted*mask + batch_incomplete*(1.-mask)
        # local patches
        local_patch_batch_pos = local_patch(batch_pos, bbox)
        local_patch_batch_predicted = local_patch(batch_predicted, bbox)
        local_patch_x1 = local_patch(x1, bbox)
        local_patch_x2 = local_patch(x2, bbox)
        local_patch_batch_complete = local_patch(batch_complete, bbox)
        local_patch_mask = local_patch(mask, bbox)

        #3Dloss
        batch_data_255 = (batch_data+1)/2
        batch_complete_255 = (batch_complete + 1) /2
        network = resfcn256(256, 256)
        pos_pos = network(batch_data_255, is_training=False)#gt的uv图
        pos_neg = network(batch_complete_255,is_training=False)#预测结果的uv图

        pos_pos_left_eye_part_y, pos_pos_right_eye_part_y, pos_pos_nose_part_y, pos_pos_nose_part_x, pos_pos_mouth_part_y = self.find_eye_nose_mouth(
            pos_pos,config)
        pos_neg_left_eye_part_y, pos_neg_right_eye_part_y, pos_neg_nose_part_y, pos_neg_nose_part_x, pos_neg_mouth_part_y = self.find_eye_nose_mouth(
            pos_neg,config)

        left_eye_gradient = tf.abs(pos_pos_left_eye_part_y - pos_neg_left_eye_part_y)
        right_eye_gradient = tf.abs(pos_pos_right_eye_part_y - pos_neg_right_eye_part_y)
        nose_gradient_y = tf.abs(pos_pos_nose_part_y - pos_neg_nose_part_y )
        nose_gradient_x = tf.abs(pos_pos_nose_part_x - pos_neg_nose_part_x )
        mouth_gradient = tf.abs(pos_pos_mouth_part_y -pos_neg_mouth_part_y )
        losses['nose_loss'] = 0.1*tf.reduce_mean(left_eye_gradient)+0.1*tf.reduce_mean(right_eye_gradient)+tf.reduce_mean(nose_gradient_x)+tf.reduce_mean(nose_gradient_y)+0.1*tf.reduce_mean(mouth_gradient)
        # left_eye_gradient = pos_pos_left_eye_part_y / (pos_neg_left_eye_part_y + 0.000000001)
        # right_eye_gradient = pos_pos_right_eye_part_y / (pos_neg_right_eye_part_y + 0.00000001)
        # nose_gradient_y = pos_pos_nose_part_y / (pos_neg_nose_part_y + 0.00000001)
        # nose_gradient_x = pos_pos_nose_part_x / (pos_neg_nose_part_x + 0.00000001)
        # mouth_gradient = pos_pos_mouth_part_y / (pos_neg_mouth_part_y + 0.00000001)
        # losses['nose_loss'] = 0.01 * (tf.reduce_mean(tf.abs(left_eye_gradient - 1.))) + 0.01 * (
        #     tf.reduce_mean(tf.abs(right_eye_gradient - 1.))) + (tf.reduce_mean(tf.abs(nose_gradient_y - 1.) )) + (
        #                                   tf.reduce_mean(tf.abs(nose_gradient_x- 1.))) + 0.01 * (
        #                                   tf.reduce_mean(tf.abs(mouth_gradient- 1.)))

        '''
        row_up = 10
        row_down = 20
        column = 20
        #3D_nose loss
        pos_pos_z =pos_neg[:,:,:,0]
        pos_nose_part = []
        pos_pos_z = tf.split(pos_pos_z,4,axis=0)
        for pos_pos_z_i in pos_pos_z:
            # pos_pos_z_i = pos_pos_z[i, :, :]
            pos_row_max = tf.reduce_max(pos_pos_z_i, axis=1)
            pos_row = tf.argmax(pos_row_max, axis=1)
            pos_column_max = tf.reduce_max(pos_pos_z_i, axis=2)
            pos_column = tf.argmax(pos_column_max, axis=1)
            pos_row = pos_row[0]
            pos_column = pos_column[0]
            pos_row_row = pos_pos_z_i[:, pos_row - row_up:pos_row + row_down, :]
            def f1():
                return pos_row_row
            def f2():
                return pos_pos_z_i[:, pos_row:pos_row + row_up + row_down, :]
            def f3():
                return pos_pos_z_i[:, pos_row - row_down - row_up:pos_row, :]
            pos_row_row = tf.cond(tf.greater_equal(pos_row,row_up), true_fn=f1,false_fn=f2)
            pos_row_row = tf.cond(tf.greater_equal(pos_row+row_down,255),true_fn=f3,false_fn=f1)
            pos_nose = pos_row_row[:, : , pos_column-column:pos_column+column]
            def f4():
                return pos_nose
            def f5():
                return pos_row_row[:,:,pos_column:pos_column+2*column]
            def f6():
                return pos_row_row[:,:,pos_column-2*column:pos_column]
            pos_nose = tf.cond(tf.greater_equal(pos_column,column),true_fn=f4,false_fn=f5)
            pos_nose = tf.cond(tf.greater_equal(pos_column+column,255),true_fn=f6,false_fn=f4)
            pos_nose_part.append(pos_nose)
        pos_nose_part = tf.concat(pos_nose_part,axis=0)
        pos_nose_part = tf.expand_dims(pos_nose_part,-1)
        (pos_nose_gradient_y,pos_nose_gradient_x) = tf.image.image_gradients(pos_nose_part)
        nose_gradient = (2.-pos_nose_gradient_y)*(2.-pos_nose_gradient_x)/2
        losses['nose_loss'] = tf.reduce_mean(nose_gradient)
        '''


        l1_alpha = config.COARSE_L1_ALPHA
        losses['l1_loss'] = l1_alpha * tf.reduce_mean(tf.abs(local_patch_batch_pos - local_patch_x1)*spatial_discounting_mask(config))
        if not config.PRETRAIN_COARSE_NETWORK:
            losses['l1_loss'] += tf.reduce_mean(tf.abs(local_patch_batch_pos - local_patch_x2)*spatial_discounting_mask(config))
        losses['ae_loss'] = l1_alpha * tf.reduce_mean(tf.abs(batch_pos - x1) * (1.-mask))
        if not config.PRETRAIN_COARSE_NETWORK:
            losses['ae_loss'] += tf.reduce_mean(tf.abs(batch_pos - x2) * (1.-mask))
        losses['ae_loss'] /= tf.reduce_mean(1.-mask)
        if summary:
            scalar_summary('losses/l1_loss', losses['l1_loss'])
            scalar_summary('losses/ae_loss', losses['ae_loss'])
            viz_img = [batch_pos, batch_incomplete, batch_complete]
            if offset_flow is not None:
                viz_img.append(
                    resize(offset_flow, scale=4,
                           func=tf.image.resize_nearest_neighbor))
            images_summary(
                tf.concat(viz_img, axis=2),
                'raw_incomplete_predicted_complete', config.VIZ_MAX_OUT)

        # gan
        batch_pos_neg = tf.concat([batch_pos, batch_complete], axis=0)
        # local deterministic patch
        local_patch_batch_pos_neg = tf.concat([local_patch_batch_pos, local_patch_batch_complete], 0)
        if config.GAN_WITH_MASK:
            batch_pos_neg = tf.concat([batch_pos_neg, tf.tile(mask, [config.BATCH_SIZE*2, 1, 1, 1])], axis=3)
        # wgan with gradient penalty
        if config.GAN == 'wgan_gp':
            # seperate gan
            pos_neg_local, pos_neg_global = self.build_wgan_discriminator(local_patch_batch_pos_neg, batch_pos_neg, training=training, reuse=reuse)
            pos_local, neg_local = tf.split(pos_neg_local, 2)
            pos_global, neg_global = tf.split(pos_neg_global, 2)
            # wgan loss
            g_loss_local, d_loss_local = gan_wgan_loss(pos_local, neg_local, name='gan/local_gan')
            g_loss_global, d_loss_global = gan_wgan_loss(pos_global, neg_global, name='gan/global_gan')
            losses['g_loss'] = config.GLOBAL_WGAN_LOSS_ALPHA * g_loss_global + g_loss_local
            losses['d_loss'] = d_loss_global + d_loss_local
            # gp
            interpolates_local = random_interpolates(local_patch_batch_pos, local_patch_batch_complete)
            interpolates_global = random_interpolates(batch_pos, batch_complete)
            dout_local, dout_global = self.build_wgan_discriminator(
                interpolates_local, interpolates_global, reuse=True)
            # apply penalty
            penalty_local = gradients_penalty(interpolates_local, dout_local, mask=local_patch_mask)
            penalty_global = gradients_penalty(interpolates_global, dout_global, mask=mask)
            losses['gp_loss'] = config.WGAN_GP_LAMBDA * (penalty_local + penalty_global)
            losses['d_loss'] = losses['d_loss'] + losses['gp_loss']
            if summary and not config.PRETRAIN_COARSE_NETWORK:
                gradients_summary(g_loss_local, batch_predicted, name='g_loss_local')
                gradients_summary(g_loss_global, batch_predicted, name='g_loss_global')
                scalar_summary('convergence/d_loss', losses['d_loss'])
                scalar_summary('convergence/local_d_loss', d_loss_local)
                scalar_summary('convergence/global_d_loss', d_loss_global)
                scalar_summary('gan_wgan_loss/gp_loss', losses['gp_loss'])
                scalar_summary('gan_wgan_loss/gp_penalty_local', penalty_local)
                scalar_summary('gan_wgan_loss/gp_penalty_global', penalty_global)

        if summary and not config.PRETRAIN_COARSE_NETWORK:
            # summary the magnitude of gradients from different losses w.r.t. predicted image
            gradients_summary(losses['g_loss'], batch_predicted, name='g_loss')
            gradients_summary(losses['g_loss'], x1, name='g_loss_to_x1')
            gradients_summary(losses['g_loss'], x2, name='g_loss_to_x2')
            gradients_summary(losses['l1_loss'], x1, name='l1_loss_to_x1')
            gradients_summary(losses['l1_loss'], x2, name='l1_loss_to_x2')
            gradients_summary(losses['ae_loss'], x1, name='ae_loss_to_x1')
            gradients_summary(losses['ae_loss'], x2, name='ae_loss_to_x2')
        if config.PRETRAIN_COARSE_NETWORK:
            losses['g_loss'] = 0
        else:
            losses['g_loss'] = config.GAN_LOSS_ALPHA * losses['g_loss']
        losses['g_loss'] += config.L1_LOSS_ALPHA * losses['l1_loss']
        losses['g_loss'] += 0.03 * losses['nose_loss']
        logger.info('Set L1_LOSS_ALPHA to %f' % config.L1_LOSS_ALPHA)
        logger.info('Set GAN_LOSS_ALPHA to %f' % config.GAN_LOSS_ALPHA)
        if config.AE_LOSS:
            losses['g_loss'] += config.AE_LOSS_ALPHA * losses['ae_loss']
            logger.info('Set AE_LOSS_ALPHA to %f' % config.AE_LOSS_ALPHA)
        g_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'inpaint_net')
        d_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        return g_vars, d_vars, losses

    def build_infer_graph(self, batch_data, config, bbox=None, name='val'):
        """
        """
        config.MAX_DELTA_HEIGHT = 0
        config.MAX_DELTA_WIDTH = 0
        if bbox is None:
            bbox = random_bbox(config)
        mask = bbox2mask(bbox, config, name=name+'mask_c')
        batch_pos = batch_data / 127.5 - 1.
        edges = None
        batch_incomplete = batch_pos*(1.-mask)
        # inpaint
        x1, x2, offset_flow = self.build_inpaint_net(
            batch_incomplete, mask, config, reuse=True,
            training=False, padding=config.PADDING)
        if config.PRETRAIN_COARSE_NETWORK:
            batch_predicted = x1
            logger.info('Set batch_predicted to x1.')
        else:
            batch_predicted = x2
            logger.info('Set batch_predicted to x2.')
        # apply mask and reconstruct
        batch_complete = batch_predicted*mask + batch_incomplete*(1.-mask)
        # global image visualization
        viz_img = [batch_pos, batch_incomplete, batch_complete]
        if offset_flow is not None:
            viz_img.append(
                resize(offset_flow, scale=4,
                       func=tf.image.resize_nearest_neighbor))
        images_summary(
            tf.concat(viz_img, axis=2),
            name+'_raw_incomplete_complete', config.VIZ_MAX_OUT)
        return batch_complete

    def build_static_infer_graph(self, batch_data, config, name):
        """
        """
        # generate mask, 1 represents masked point
        bbox = (tf.constant(config.HEIGHT//2), tf.constant(config.WIDTH//2),
                tf.constant(config.HEIGHT), tf.constant(config.WIDTH))
        return self.build_infer_graph(batch_data, config, bbox, name)


    def build_server_graph(self, batch_data, reuse=False, is_training=False):
        """
        """
        # generate mask, 1 represents masked point
        batch_raw, masks_raw = tf.split(batch_data, 2, axis=2)
        masks = tf.cast(masks_raw[0:1, :, :, 0:1] > 127.5, tf.float32)

        batch_pos = batch_raw / 127.5 - 1.
        batch_incomplete = batch_pos * (1. - masks)
        # inpaint
        x1, x2, flow = self.build_inpaint_net(
            batch_incomplete, masks, reuse=reuse, training=is_training,
            config=None)
        batch_predict = x2
        # apply mask and reconstruct
        batch_complete = batch_predict*masks + batch_incomplete*(1-masks)
        return batch_complete


    def find_eye_nose_mouth(self, pos_map,config):
        pos_pos_z = tf.split(pos_map, config.BATCH_SIZE, axis=0)
        left_eye_part = []
        right_eye_part = []
        nose_part = []
        mouth_part = []
        for pos_pos_z_i in pos_pos_z:
            pos_pos_z_i = pos_pos_z_i[:, :, :, 0]
            pos_pos_z_i = tf.squeeze(pos_pos_z_i, axis=0)
            left_eye = pos_pos_z_i[51:79, 67:109]
            left_eye = tf.expand_dims(left_eye, 0)
            right_eye = pos_pos_z_i[51:79, 156:198]
            right_eye = tf.expand_dims(right_eye, 0)
            nose = pos_pos_z_i[55:116, 107:149]
            nose = tf.expand_dims(nose, 0)
            mouth = pos_pos_z_i[121:167, 102:159]
            mouth = tf.expand_dims(mouth, 0)
            left_eye_part.append(left_eye)
            right_eye_part.append(right_eye)
            nose_part.append(nose)
            mouth_part.append(mouth)
        left_eye_part = tf.concat(left_eye_part, axis=0)
        left_eye_part = tf.expand_dims(left_eye_part, -1)
        right_eye_part = tf.concat(right_eye_part, axis=0)
        right_eye_part = tf.expand_dims(right_eye_part, -1)
        nose_part = tf.concat(nose_part, axis=0)
        nose_part = tf.expand_dims(nose_part, -1)
        mouth_part = tf.concat(mouth_part, axis=0)
        mouth_part = tf.expand_dims(mouth_part, -1)
        (pos_left_eye_part_y, pos_left_eye_part_x) = tf.image.image_gradients(left_eye_part)
        (pos_right_eye_part_y, pos_right_eye_part_x) = tf.image.image_gradients(right_eye_part)
        (pos_nose_part_y, pos_nose_part_x) = tf.image.image_gradients(nose_part)
        (pos_mouth_part_y, pos_mouth_part_x) = tf.image.image_gradients(mouth_part)
        pos_left_eye_part_y = tf.abs(pos_left_eye_part_y)
        pos_right_eye_part_y = tf.abs(pos_right_eye_part_y)
        pos_nose_part_y = tf.abs(pos_nose_part_y)
        pos_nose_part_x = tf.abs(pos_nose_part_x)
        pos_mouth_part_y = tf.abs(pos_mouth_part_y)
        return pos_left_eye_part_y, pos_right_eye_part_y, pos_nose_part_y, pos_nose_part_x, pos_mouth_part_y

