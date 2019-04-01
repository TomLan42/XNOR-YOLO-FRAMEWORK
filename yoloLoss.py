#encoding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class yoloLoss(nn.Module):
    def __init__(self,S,B,l_coord,l_noobj):
        super(yoloLoss,self).__init__()
        self.S = S
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def compute_iou(self, box1, box2):
        '''Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
        Args:
          box1: (tensor) bounding boxes, sized [N,4].
          box2: (tensor) bounding boxes, sized [M,4].
        Return:
          (tensor) iou, sized [N,M].
        '''
        N = box1.size(0)
        M = box2.size(0)

        lt = torch.max(
            box1[:,:2].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:,:2].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        rb = torch.min(
            box1[:,2:].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:,2:].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        wh = rb - lt  # [N,M,2]
        wh[wh<0] = 0  # clip at 0
        inter = wh[:,:,0] * wh[:,:,1]  # [N,M]

        area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])  # [N,]
        area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter)
        return iou
    def forward(self,pred_tensor,target_tensor):
        '''
        pred_tensor: (tensor) size(batchsize,S,S,Bx5+20=30) [x,y,w,h,c]
        target_tensor: (tensor) size(batchsize,S,S,30)
        '''
        N = pred_tensor.size()[0]
        coo_mask = target_tensor[:,:,:,4] > 0
        noo_mask = target_tensor[:,:,:,4] == 0
        coo_mask = coo_mask.unsqueeze(-1).expand_as(target_tensor)
        noo_mask = noo_mask.unsqueeze(-1).expand_as(target_tensor)

        coo_pred = pred_tensor[coo_mask].view(-1,30)
        box_pred = coo_pred[:,:10].contiguous().view(-1,5)            #box_pred[[x1,y1,w1,h1,c1]
        class_pred = coo_pred[:,10:].contiguous()
                                             #         [x2,y2,w2,h2,c2]] * n个
        
        coo_target = target_tensor[coo_mask].view(-1,30)
        box_target = coo_target[:,:10].contiguous().view(-1,5)
        class_target = coo_target[:,10:].contiguous()

        # 1. 20 classes probability loss
        mse = torch.nn.MSELoss(reduction='sum')
        if not int(class_pred[1,:].max(0)[1]) == 14:
            print 'Nice! We got class' , int(class_pred[1,:].max(0)[1])
        class_loss = mse(class_pred,class_target)
        
     

        # 2. Object in grid cell AND bbox responsible. Confidence (c) Loss.
        coo_response_mask = torch.cuda.ByteTensor(box_target.size())
        coo_response_mask.zero_()
        box_target_iou = torch.zeros(box_target.size()).cuda()
        for i in range(0,box_target.size()[0],2):                       #遍历成对的bbox
            box1 = box_pred[i:i+2]
            box1_xyxy = Variable(torch.FloatTensor(box1.size()))
            box1_xyxy[:,:2] = box1[:,:2]/7. - 0.5*box1[:,2:4]
            box1_xyxy[:,2:4] = box1[:,:2]/7. + 0.5*box1[:,2:4]
            box2 = box_target[i].view(-1,5)
            box2_xyxy = Variable(torch.FloatTensor(box2.size()))
            box2_xyxy[:,:2] = box2[:,:2]/7. - 0.5*box2[:,2:4]
            box2_xyxy[:,2:4] = box2[:,:2]/7. + 0.5*box2[:,2:4]
            
            iou = self.compute_iou(box1_xyxy[:,:4],box2_xyxy[:,:4])     #尺寸[2,1]
            max_iou,max_index = iou.max(0)
            max_index = max_index.data.cuda()
            coo_response_mask[i+max_index]=1
            box_target_iou[i+max_index,torch.LongTensor([4]).cuda()] = (max_iou).data.cuda()

        box_target_iou = Variable(box_target_iou).cuda()
        box_pred_response = box_pred[coo_response_mask].view(-1,5)
        box_target_response_iou = box_target_iou[coo_response_mask].view(-1,5)
        box_target_response = box_target[coo_response_mask].view(-1,5)
        contain_loss = F.mse_loss(box_pred_response[:,4],box_target_response_iou[:,4],size_average=False)
        
        # 3. Object in grid cell AND bbox responsible. Localization Loss. 
        loc_loss = F.mse_loss(box_pred_response[:,:2],box_target_response[:,:2],size_average=False) + F.mse_loss(torch.sqrt(box_pred_response[:,2:4]),torch.sqrt(box_target_response[:,2:4]),size_average=False)
        

        # 4. Object NOT in grid cell AND bbox responsible. Confidence (c) Loss.
        noo_pred = pred_tensor[noo_mask].view(-1,30)
        noo_box_pred = noo_pred[:,:10].contiguous().view(-1,5)
        noo_target = target_tensor[noo_mask].view(-1,30)
        noo_box_target = noo_target[:,:10].contiguous().view(-1,5)
        
        noo_response_mask = torch.cuda.ByteTensor(noo_box_target.size())
        noo_response_mask.zero_()
        noo_box_target_iou = torch.zeros(noo_box_target.size()).cuda()
        for i in range(0,noo_box_target.size()[0],2):
            box1 = noo_box_pred[i:i+2]
            box1_xyxy = Variable(torch.FloatTensor(box1.size()))
            box1_xyxy[:,:2] = box1[:,:2]/7. - 0.5*box1[:,2:4]
            box1_xyxy[:,2:4] = box1[:,:2]/7. + 0.5*box1[:,2:4]
            box2 = noo_box_target[i].view(-1,5)
            box2_xyxy = Variable(torch.FloatTensor(box2.size()))
            box2_xyxy[:,:2] = box2[:,:2]/7. - 0.5*box2[:,2:4]
            box2_xyxy[:,2:4] = box2[:,:2]/7. + 0.5*box2[:,2:4]
            iou = self.compute_iou(box1_xyxy[:,:4],box2_xyxy[:,:4]) #[2,1]
            max_iou,max_index = iou.max(0)
            max_index = max_index.data.cuda()
            
            noo_response_mask[i+max_index]=1
        noo_box_target_iou = Variable(noo_box_target_iou).cuda()
        noo_box_pred_response = noo_box_pred[noo_response_mask].view(-1,5)
        noo_box_target_response_iou = noo_box_target_iou[noo_response_mask].view(-1,5)
        noo_box_target_response = noo_box_target[noo_response_mask].view(-1,5)
        noo_contain_loss = F.mse_loss(noo_box_pred_response[:,4],noo_box_target_response_iou[:,4],size_average=False)
       
        print ('Coordination Loss: %.4f, Conofidence Loss: %.4f, classes Loss: %.4f' 
            %(self.l_coord*loc_loss/N, (contain_loss + self.l_noobj*noo_contain_loss)/N,class_loss/N))
            
        return (self.l_coord*loc_loss + contain_loss + self.l_noobj*noo_contain_loss + class_loss)/N



       