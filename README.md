# Not-a-Diagnosis
## Pipline
数据预处理
```
DICOM (CT slices)
   ↓
重建 3D 体数据（HU）
   ↓
根据 LIDC 标注定位结节
   ↓
裁剪固定尺寸 3D patch
   ↓
HU clip + normalize
   ↓
保存为 .npy
```
```
LIDC-IDRI CT (DICOM)
        │
        ▼
  pylidc 读取结节标注
        │
        ▼
结节中心点 + 半径
        │
        ▼
3D patch crop (64×64×64)
        │
        ▼
HU 归一化 / 标准化
        │
        ▼
3D CNN（ResNet）
        │
        ▼
P(malignant)
```