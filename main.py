import numpy
import cv2
import os
import time
import pygame
import queue
import threading

# 无缓存读取视频流类
class VideoCapture:

  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    self.q = queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  # 帧可用时立即读取帧，只保留最新的帧
  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      if not self.q.empty():
        try:
          self.q.get_nowait()   # 删除上一个（未处理的）帧
        except queue.Empty:
          pass
      self.q.put(frame)

  def read(self):
    return self.q.get()



weightsPath = os.path.join('yolov3-tiny_10000.weights')  # 权重文件
configPath = os.path.join('yolov3-tiny.cfg')  # 配置文件
labelsPath = os.path.join('mask.txt')  # label名称windows darknet python
#imgPath = os.path.join(yolo_dir, 'test.jpg')  # 测试图像
CONFIDENCE = 0.8  # 过滤弱检测的最小概率
THRESHOLD = 0.4  # 非最大值抑制阈值

count = 0
countt = 0
yes=0
no=0



net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
print("[INFO] loading YOLO from disk...")  # # 可以打印下信息


cameraCapture = VideoCapture(0)  # 打开编号为0的摄像头

frame = cameraCapture.read()
while 1:  
    # 加载图片、转为blob格式、送入网络输入层
    blobImg = cv2.dnn.blobFromImage(frame, 1.0 / 255.0, (416, 416), None, True,
                                False)  # # net需要的输入是blob格式的，用blobFromImage这个函数来转格式
    net.setInput(blobImg)  # # 调用setInput函数将图片送入输入层

    # 获取网络输出层信息（所有输出层的名字），设定并前向传播
    outInfo = net.getUnconnectedOutLayersNames()  # # 前面的yolov3架构也讲了，yolo在每个scale都有输出，outInfo是每个scale的名字信息，供net.forward使用
    start = time.time()
    layerOutputs = net.forward(outInfo)  # 得到各个输出层的、各个检测框等信息，是二维结构。
    end = time.time()
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))  # # 可以打印下信息

    # 拿到图片尺寸
    (H, W) = frame.shape[:2]
    # 过滤layerOutputs
    # layerOutputs的第1维的元素内容: [center_x, center_y, width, height, objectness, N-class score data]
    # 过滤后的结果放入：
    boxes = []  # 所有边界框（各层结果放一起）
    confidences = []  # 所有置信度
    classIDs = []  # 所有分类ID

    # # 1）过滤掉置信度低的框框
    for out in layerOutputs:  # 各个输出层
        for detection in out:  # 各个框框
            # 拿到置信度
            scores = detection[5:]  # 各个类别的置信度
            classID = numpy.argmax(scores)  # 最高置信度的id即为分类id
            confidence = scores[classID]  # 拿到置信度

            # 根据置信度筛查
            if confidence > CONFIDENCE:
                box = detection[0:4] * numpy.array([W, H, W, H])  # 将边界框放会图片尺寸
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # # 2）应用非最大值抑制(non-maxima suppression，nms)进一步筛掉
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)  # boxes中，保留的box的索引index存入idxs
    # 得到labels列表
    with open(labelsPath, 'rt') as f:
        labels = f.read().rstrip('\n').split('\n')
    # 应用检测结果
    numpy.random.seed(42)
    COLORS = numpy.random.randint(0, 255, size=(len(labels), 3),
                                  dtype="uint8")  # 框框显示颜色，每一类有不同的颜色，每种颜色都是由RGB三个值组成的，所以size为(len(labels), 3)
    if len(idxs) > 0:
        for i in idxs.flatten():  # indxs是二维的，第0维是输出层，所以这里把它展平成1维
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)  # 线条粗细为2px
            text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])

            print(classIDs[i]) #输出1，对应unmasked

            if classIDs[i] == 1: #如果没带口罩
                count+=1
            else:
                countt+=1
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,
                        2)  # cv.FONT_HERSHEY_SIMPLEX字体风格、0.5字体大小、粗细2px

    if count >= 1:
        if no==0:
            pygame.mixer.init()
            track = pygame.mixer.music.load('478.mp3')
            pygame.mixer.music.play()
            # yes,no变量用于保证只有第一次识别到没戴/戴了口罩才发出提示音
            no=1
            yes=0

        count -= 1
    if countt >= 1:
        if yes==0:
            pygame.mixer.init()
            track = pygame.mixer.music.load('13538.mp3')
            pygame.mixer.music.play()
            yes=1
            no=0

        countt -= 1
    frame = cameraCapture.read()  # 摄像头获取下一帧

cameraCapture.release()
