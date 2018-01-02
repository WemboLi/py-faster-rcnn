# change matplotlib settings

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import matplotlib
matplotlib.use('Agg')

CLASSES = ('__background__', # always index 0
         'adidas', 'aldi', 'apple', 'becks', 'bmw',
         'carlsberg', 'chimay', 'cocacola', 'corona',
         'dhl', 'erdinger', 'esso', 'fedex', 'ferrari',
         'ford', 'fosters', 'google', 'guinness', 'heineken',
         'HP', 'joy', 'milka', 'nvidia', 'paulaner', 'pepsi', 'rittersport',
         'shell', 'singha', 'starbucks', 'stellaartois', 'texaco',
         'tsingtao', 'ups') 


NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}

# improve the threshold for high confidence of the bbox
def vis_detections(im, class_name, dets, image_name, thresh):
    class_path = os.path.join('output', 'demo', class_name)
    
    if not os.path.isdir(class_path):
        os.mkdir(class_path)
        print "create class_path {}".format(class_path)

    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    dis_im = im[:, :, (2, 1, 0)].copy()
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(dis_im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        print '{}th bbox: {}'.format(i, bbox)
        
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    # save the image with bounding box and class_name         
    plt.savefig(os.path.join(class_path, image_name))
    plt.close()

def demo(net, image_name, verbose = False):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo_logos', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
       cls_ind += 1 # because we skipped background
       cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
       cls_scores = scores[:, cls_ind]
       dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
       keep = nms(dets, NMS_THRESH, True)
       if verbose:
          for i in range(len(keep)):
               for j in range(len(keep)):
                   if i == j:
                      continue

                   if dets[i, -1] >= CONF_THRESH and dets[j, -1] >= CONF_THRESH:
                        over = overlap(dets[keep[i], :4], dets[keep[j], :4])
                        assert over < NMS_THRESH, "nms thresh error"
                        print 'overlap between {:d} and {:d} is {:f}'.format(i, j, over)

       dets = dets[keep, :]
       vis_detections(im, cls, dets, image_name, thresh=CONF_THRESH)

def overlap(bbox1, bbox2):
    area1 = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)  
    area2 = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)

    xx1 = max(bbox1[0], bbox2[0])
    yy1 = max(bbox1[1], bbox2[1])
    xx2 = min(bbox1[2], bbox2[2])
    yy2 = min(bbox1[3], bbox2[3])
    w = max(0.0, xx2 - xx1 + 1)
    h = max(0.0, yy2 - yy1 + 1)
    inter = w * h
    ovr = inter / (area1 + area2 - inter)

    return ovr

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    # prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
    #                        'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    #caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              #NETS[args.demo_net][1])
    prototxt = os.path.join('models', 'logos33', NETS[args.demo_net][0],
                            'faster_rcnn_end2end', 'test.prototxt')
   
    caffemodel = os.path.join('output', 'faster_rcnn_end2end', 'trainval2017', 
                        'vgg16_faster_rcnn_logos_iter_70000.caffemodel')
 
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    # im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
                # '001763.jpg', '004545.jpg']
    im_path = os.path.join(cfg.DATA_DIR, 'demo_logos')
    im_names = [f for f in os.listdir(im_path) if f.endswith('.jpg')]     

    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        demo(net, im_name, True)

#    plt.show()
