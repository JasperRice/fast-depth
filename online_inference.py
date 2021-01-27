import time

import cv2
import torch


if __name__ == '__main__':
    alpha = 0.3
    output_directory = 'results/'
    checkpoint = torch.load(
        'results/mobilenet-nnconv5dw-skipadd-pruned.pth.tar')
    if isinstance(checkpoint, dict):
        start_epoch = checkpoint['epoch']
        best_result = checkpoint['best_result']
        model = checkpoint['model']
    else:
        start_epoch = 0
        model = checkpoint
    model.eval()

    end = time.time()
    capture = cv2.VideoCapture(0)
    while capture.isOpened():
        ret, bgr = capture.read()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = torch.from_numpy(rgb)
        data_time = time.time() - end

        end = time.time()
        with torch.no_grad():
            pred = model(rgb)
        gpu_time = time.time() - end

        heat = cv2.applyColorMap(pred.data, cv2.COLORMAP_JET)
        bgr = cv2.addWeighted(heat, alpha, bgr, 1-alpha, 0)

        cv2.imshow('Depth', bgr)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
