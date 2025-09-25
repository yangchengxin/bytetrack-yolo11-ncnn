# yolo11 + bytetrack ncnn

## 依赖
* opencv (windows端运行，因此不需要android opencv，在官网自行下载即可)
* ncnn 
* eigen

## 编译
使用cmake-gui编译


## Debug 模式下的报错
在Debug模式下有可能会生成报错，那是因为cmakelists解析的时候没有成功把opencvxxxd.dll和ncnnd.dll注册到我们的附加依赖项中，我们只需要手动的打开属性页中的输入，附加依赖项，然后分别在这两个注册项后面加上d即可：
<img width="659" height="275" alt="image" src="https://github.com/user-attachments/assets/7127807a-62e5-4a26-9808-77723ec214f2" />
