# yolo11 + bytetrack ncnn

## 依赖
* opencv (windows端运行，因此不需要android opencv，在官网自行下载即可)
* ncnn 
* eigen

## 编译
使用cmake-gui编译


## 模型生成
* 训练一个模型（此处省略）

* 训练完模型之后，打开head.py，修改检测头中的forward方法，让我们拿到的推理结果是模型的三个检测头直接输出的结果（没有经过后处理）:
```python
    # ncnn
    def forward(self, x):
        z = []
        for i in range(self.nl):
            boxes = self.cv2[i](x[i]).permute(0, 2, 3, 1)
            scores = self.cv3[i](x[i]).permute(0, 2, 3, 1)
            feat = torch.cat((boxes, scores), dim=-1)
            z.append(feat)

        return tuple(z)
```

* 导出我们训练完后保存的模型：
```python
from ultralytics import YOLO

# Load the model
model = YOLO(r'yolo11n.pt')

model.export(format='ncnn', imgsz=640, batch=1, dynamic=False)
```


## Debug 模式下的报错
在Debug模式下有可能会生成报错，那是因为cmakelists解析的时候没有成功把opencvxxxd.dll和ncnnd.dll注册到我们的附加依赖项中，我们只需要手动的打开属性页中的输入，附加依赖项，然后分别在这两个注册项后面加上d即可：
<img width="659" height="275" alt="image" src="https://github.com/user-attachments/assets/7127807a-62e5-4a26-9808-77723ec214f2" />
