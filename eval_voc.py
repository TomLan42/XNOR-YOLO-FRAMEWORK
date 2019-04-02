#encoding:utf-8
from net import vgg16_bn
import argparse
import util
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
from yoloLoss import yoloLoss

VOC_CLASSES = (    # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')
Color = [[0, 0, 0],
                    [128, 0, 0],
                    [0, 128, 0],
                    [128, 128, 0],
                    [0, 0, 128],
                    [128, 0, 128],
                    [0, 128, 128],
                    [128, 128, 128],
                    [64, 0, 0],
                    [192, 0, 0],
                    [64, 128, 0],
                    [192, 128, 0],
                    [64, 0, 128],
                    [192, 0, 128],
                    [64, 128, 128],
                    [192, 128, 128],
                    [0, 64, 0],
                    [128, 64, 0],
                    [0, 192, 0],
                    [128, 192, 0],
                    [0, 64, 128]]
def voc_ap(rec,prec,use_07_metric=False):
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0.,1.1,0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec>=t])
            ap = ap + p/11.

    else:
        # correct ap caculation
        mrec = np.concatenate(([0.],rec,[1.]))
        mpre = np.concatenate(([0.],prec,[0.]))

        for i in range(mpre.size -1, 0, -1):
            mpre[i-1] = np.maximum(mpre[i-1],mpre[i])

        i = np.where(mrec[1:] != mrec[:-1])[0]

        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap

def voc_eval(preds,target,VOC_CLASSES=VOC_CLASSES,threshold=0.5,use_07_metric=False,):
    '''
    preds {'cat':[[image_id,confidence,x1,y1,x2,y2],...],'dog':[[],...]}
    target {(image_id,class):[[],]}
    '''
    aps = []
    for i,class_ in enumerate(VOC_CLASSES):
        print class_
        pred = preds[class_] #[[image_id,confidence,x1,y1,x2,y2],...]
        if len(pred) == 0: #如果这个类别一个都没有检测到的异常情况
            ap = -1
            print('---class {} ap {}---'.format(class_,ap))
            aps += [ap]
            continue
        #print(pred)
        image_ids = [x[0] for x in pred]
        confidence = np.array([float(x[1]) for x in pred])
        BB = np.array([x[2:] for x in pred])
        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        npos = 0.
        for (key1,key2) in target:
            if key2 == class_:
                npos += len(target[(key1,key2)]) #统计这个类别的正样本，在这里统计才不会遗漏
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d,image_id in enumerate(image_ids):
            bb = BB[d] #预测框
            if (image_id,class_) in target:
                BBGT = target[(image_id,class_)] #[[],]
                for bbgt in BBGT:
                    # compute overlaps
                    # intersection
                    ixmin = np.maximum(bbgt[0], bb[0])
                    iymin = np.maximum(bbgt[1], bb[1])
                    ixmax = np.minimum(bbgt[2], bb[2])
                    iymax = np.minimum(bbgt[3], bb[3])
                    iw = np.maximum(ixmax - ixmin + 1., 0.)
                    ih = np.maximum(iymax - iymin + 1., 0.)
                    inters = iw * ih

                    union = (bb[2]-bb[0]+1.)*(bb[3]-bb[1]+1.) + (bbgt[2]-bbgt[0]+1.)*(bbgt[3]-bbgt[1]+1.) - inters
                    if union == 0:
                        print(bb,bbgt)
                    
                    overlaps = inters/union
                    if overlaps > threshold:
                        tp[d] = 1
                        BBGT.remove(bbgt) #这个框已经匹配到了，不能再匹配
                        if len(BBGT) == 0:
                            del target[(image_id,class_)] #删除没有box的键值
                        break
                fp[d] = 1-tp[d]
            else:
                fp[d] = 1
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp/float(npos)
        prec = tp/np.maximum(tp + fp, np.finfo(np.float64).eps)
        #print(rec,prec)
        ap = voc_ap(rec, prec, use_07_metric)
        print('---class {} ap {}---'.format(class_,ap))
        aps += [ap]
    print('---map {}---'.format(np.mean(aps)))

def test_eval():
    preds = {'cat':[['image01',0.9,20,20,40,40],['image01',0.8,20,20,50,50],['image02',0.8,30,30,50,50]],'dog':[['image01',0.78,60,60,90,90]]}
    target = {('image01','cat'):[[20,20,41,41]],('image01','dog'):[[60,60,91,91]],('image02','cat'):[[30,30,51,51]]}
    voc_eval(preds,target,VOC_CLASSES=['cat','dog'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Pytorch XNOR-YOLO Evaluation')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    default=False, help='use pre-trained model')
    parser.add_argument('-ct', '--confidence_threshold', default=0.001, type=float,
                    metavar='CT', help='first round filtering')
    parser.add_argument('--pt', '--prob_threshold', default=0.01, type=float,
                    metavar='PT', help='second round filtering')
    args = parser.parse_args()

    #test_eval()
    from predict import *
    from collections import defaultdict
    from tqdm import tqdm

    target =  defaultdict(list)
    preds = defaultdict(list)
    image_list = [] #image path list

    f = open('./meta/voc2007test.txt')
    lines = f.readlines()
    file_list = []
    for line in lines:
        splited = line.strip().split()
        file_list.append(splited)
    f.close()
    print('---prepare target---')
    for index,image_file in enumerate(file_list):
        image_id = image_file[0]

        image_list.append(image_id)
        num_obj = (len(image_file) - 1) // 5
        for i in range(num_obj):
            x1 = int(image_file[1+5*i])
            y1 = int(image_file[2+5*i])
            x2 = int(image_file[3+5*i])
            y2 = int(image_file[4+5*i])
            c = int(image_file[5+5*i])
            class_name = VOC_CLASSES[c]
            target[(image_id,class_name)].append([x1,y1,x2,y2])
    
    print('---start test---')
    '''Create model.'''
    if args.pretrained:
        model = vgg16_bn(pretrained = False)
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
        model.load_state_dict(torch.load('./experiment/vgg16fp/checkpoint.pth'))
    else:
        model = model_list.vgg(pretrained = False)
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
        checkpoint = torch.load('./experiment/vgg16xnor/model_best.pth.tar')
        model.load_state_dict(checkpoint['state_dict'])

        bin_op = util.BinOp(model)
        bin_op.binarization()
    

    model.eval()
    count = 0
    for image_path in tqdm(image_list):
        result = predict_gpu(model,image_path,root_path='/mnt/lustre/share/DSK/datasets/VOC07+12/JPEGImages/') #result[[left_up,right_bottom,class_name,image_path],]
        for (x1,y1),(x2,y2),class_name,image_id,prob in result: 
            preds[class_name].append([image_id,prob,x1,y1,x2,y2])

    print('---start evaluate---')
    voc_eval(preds,target,VOC_CLASSES=VOC_CLASSES)