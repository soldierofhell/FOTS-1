### 此处默认真实值和预测值的格式均为 bs * W * H * channels
import torch
import torch.nn as nn
from torch.nn import CTCLoss


class DetectionLoss(nn.Module):
    def __init__(self, lambdas, writer):
        super(DetectionLoss, self).__init__()
        self.lambdas = lambdas
        self.writer = writer
        return

    def forward(self, y_true_cls, y_pred_cls,
                y_true_geo, y_pred_geo,
                training_mask, global_step):
        classification_loss = self.__dice_coefficient(y_true_cls, y_pred_cls, training_mask, global_step)
        # scale classification loss to match the iou loss part
        #classification_loss *= 0.01

        # d1 -> top, d2->right, d3->bottom, d4->left
        #     d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = tf.split(value=y_true_geo, num_or_size_splits=5, axis=3)
        d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = torch.split(y_true_geo, 1, 1)
        #     d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = tf.split(value=y_pred_geo, num_or_size_splits=5, axis=3)
        d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = torch.split(y_pred_geo, 1, 1)
        area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
        area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
        w_union = torch.min(d2_gt, d2_pred) + torch.min(d4_gt, d4_pred)
        h_union = torch.min(d1_gt, d1_pred) + torch.min(d3_gt, d3_pred)
        area_intersect = w_union * h_union
        area_union = area_gt + area_pred - area_intersect
        L_AABB = -torch.log((area_intersect + 1.0) / (area_union + 1.0))
        L_theta = 1 - torch.cos(theta_pred - theta_gt)
        L_g = L_AABB + self.lambdas["lambda_theta"] * L_theta
        
        L_AABB_scalar = torch.sum(L_AABB * y_true_cls * training_mask)/torch.sum(y_true_cls * training_mask)
        L_theta_scalar = torch.sum(L_theta * y_true_cls * training_mask)/torch.sum(y_true_cls * training_mask)
        
        L_g = torch.sum(L_g * y_true_cls * training_mask)/torch.sum(y_true_cls * training_mask)
        
        self.writer.add_scalar('L_g', L_g, global_step)
        self.writer.add_scalar('L_AABB', L_AABB_scalar, global_step)
        self.writer.add_scalar('L_theta', L_theta_scalar, global_step)

        return classification_loss + self.lambdas["lambda_reg"] * L_g, classification_loss

    def __dice_coefficient(self, y_true_cls, y_pred_cls,
                           training_mask, global_step):
        '''
        dice loss
        :param y_true_cls:
        :param y_pred_cls:
        :param training_mask:
        :return:
        '''
        eps = 1e-5
        intersection = torch.sum(y_true_cls * y_pred_cls * training_mask)
        union_gt = torch.sum(y_true_cls * training_mask)
        union_pred = torch.sum(y_pred_cls * training_mask)
        union = union_gt + union_pred + eps
        loss = 1. - (2 * intersection / union)
        
        self.writer.add_scalar('intersection', intersection, global_step)
        self.writer.add_scalar('union', union, global_step)
        self.writer.add_scalar('union_gt', union_gt, global_step)
        self.writer.add_scalar('union_pred', union_pred, global_step)
        
        self.writer.add_scalar('loss', loss, global_step)
        
        #print('intersection :', intersection.item())
        #print('union :', union.item())

        return loss


class RecognitionLoss(nn.Module):

    def __init__(self):
        super(RecognitionLoss, self).__init__()
        self.ctc_loss = CTCLoss()  # pred, pred_len, labels, labels_len

    def forward(self, *input):
        gt, pred = input[0], input[1]
        loss = self.ctc_loss(pred[0], gt[0], pred[1], gt[1])
        return loss


class FOTSLoss(nn.Module):

    def __init__(self, config, writer):
        super(FOTSLoss, self).__init__()
        self.mode = config['model']['mode']
        self.lambdas = config['loss_lambdas']
        self.writer = writer
        self.detection_loss = DetectionLoss(self.lambdas, self.writer)
        self.recognition_loss = RecognitionLoss()

    def forward(self, y_true_cls, y_pred_cls,
                y_true_geo, y_pred_geo,
                y_true_recog, y_pred_recog,
                training_mask, global_step):

        recognition_loss = torch.tensor([0]).float()
        detection_loss = torch.tensor([0]).float()
        classification_loss = torch.tensor([0]).float()

        if self.mode == 'recognition':
            recognition_loss = self.recognition_loss(y_true_recog, y_pred_recog)
        elif self.mode == 'detection':
            detection_loss, classification_loss = self.detection_loss(y_true_cls, y_pred_cls,
                                                 y_true_geo, y_pred_geo, training_mask, global_step)
        elif self.mode == 'united':
            detection_loss, classification_loss = self.detection_loss(y_true_cls, y_pred_cls,
                                                y_true_geo, y_pred_geo, training_mask, global_step)
            if y_true_recog:
                recognition_loss = self.recognition_loss(y_true_recog, y_pred_recog)

        recognition_loss = recognition_loss.to(detection_loss.device)
        return detection_loss, classification_loss, recognition_loss
