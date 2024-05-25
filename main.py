import queue
import cv2
import numpy as np
import argparse
import onnxruntime as rt
import threading
import datetime

q = queue.Queue()

class main():
    def __init__(self, confThreshold=0.3, nmsThreshold=0.4):
        self.classes = list(map(lambda x: x.strip(), open('coco.names',
                                                          'r').readlines()))
        self.inpWidth = 512
        self.inpHeight = 512
        providers = ['CPUExecutionProvider']

        self.sess = rt.InferenceSession('model.onnx', providers=providers, provider_options=[{'device_type' : 'GPU_FP32'}])

        self.input_name = self.sess.get_inputs()[0].name
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.H, self.W = 32, 32
        self.grid = self._make_grid(self.W, self.H)
        self.detectTime = 0

    def _make_grid(self, nx=20, ny=20):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)

    def postprocess(self, frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        classIds = []
        confidences = []
        boxes = []
        for detection in outs:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId] * detection[0]
            if confidence > self.confThreshold:
                center_x = int(detection[1] * frameWidth)
                center_y = int(detection[2] * frameHeight)
                width = int(detection[3] * frameWidth)
                height = int(detection[4] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        if len(indices) == 0:
            return frame
        indices = np.array(indices).flatten()

        for i in indices:
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]

            frame = self.drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)

        return frame

    def drawPred(self, frame, classId, conf, left, top, right, bottom):

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=2)

        label = '%.2f' % conf
        label = '%s:%s' % (self.classes[classId], label)


        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=1)
        return frame

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def detect(self, srcimg):

        blob = cv2.dnn.blobFromImage(srcimg, 1 / 255.0, (self.inpWidth, self.inpHeight))
        t1 = cv2.getTickCount()
        pred = self.sess.run(None, {self.input_name: blob})[0]
        t2 = cv2.getTickCount()
        self.detectTime = (t2 - t1) / cv2.getTickFrequency()
        pred[:, 3:5] = self.sigmoid(pred[:, 3:5])
        pred[:, 1:3] = (np.tanh(pred[:, 1:3]) + self.grid) / np.tile(np.array([self.W,self.H]), (pred.shape[0], 1)) ###cx,cy

        srcimg = self.postprocess(srcimg, pred)
        now2 = datetime.datetime.now()
        now2_str = now2.strftime("%Y-%m-%d %H:%M:%S")
        print(f"{now2_str}本轮计算时间: %.2f ms" % (self.detectTime * 1000))
        return srcimg


def ReceiveAndDisplay(model):
    try:
        now = datetime.datetime.now()
        now_str = now.strftime("%Y-%m-%d %H:%M:%S")
        print(f"{now_str}: 开始接受数据流")

        stream = input("输入1后开始")
        if stream == '1':
            cap = cv2.VideoCapture("在此处填写你的输入源")
        else:
            now = datetime.datetime.now()
            now_str = now.strftime("%Y-%m-%d %H:%M:%S")
            print(f"{now_str}：别乱输，让你选1就选1")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = model.detect(frame)
            if frame is not None:
                cv2.imshow("FTD-cv-gz-B4", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except Exception as e:
        print(e)
    finally:
        cap.release()
        cv2.destroyAllWindows()




if __name__ == '__main__':
    now1 = datetime.datetime.now()
    now1_str = now1.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{now1_str}: 正在准备环境")


    parser = argparse.ArgumentParser()
    parser.add_argument('--confThreshold', default=0.5, type=float, help='class confidence')
    parser.add_argument('--nmsThreshold', default=0.35, type=float, help='nms iou thresh')
    args = parser.parse_args()
    model = main(confThreshold=args.confThreshold, nmsThreshold=args.nmsThreshold)
    now2 = datetime.datetime.now()
    now2_str = now2.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{now2_str}: 新线程已启动")
    p1 = threading.Thread(target=ReceiveAndDisplay, args=(model,))
    p1.start()