作用：用卷积做预测类别；
输入：每个像素（feature_map)， 锚框数量， 类别数量
输出：对每一个锚框的类别都进行预测（类似独热编码（0, 1, 0, ..., 0）
```
def cls_predictor(num_inputs, num_anchors, num_classes):
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1),
                     kernel_size=3, padding=1)
```
`num_anchors * (num_classes + 1)`：`+1`是背景
`kernel_size=3, padding=1`：保证了形状不变


**ps：这里的函数参数知识为了配置卷积层的超参数，而输入的张量实际形状为（batch_size, num_inputs, H, W）；与函数参数无关**
作用：对每个锚框和真实边框的偏移的预测size为（4，）
输入张量形状：`(batch_size, num_inputs, H, W)`
输出张量形状：`(batch_size, num_anchors*4, H, W)`
```
def bbox_predictor(num_inputs, num_anchors):
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)
```