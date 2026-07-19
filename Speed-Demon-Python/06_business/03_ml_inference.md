# 机器学习推理性能优化

在生产环境中，机器学习模型的推理性能直接影响服务的响应速度和吞吐量。本文将系统介绍7种主流的推理性能优化技术。

## 性能对比概览

| 优化技术 | 速度提升 | 精度损失 | 实现复杂度 |
|---------|---------|---------|-----------|
| 基线 (FP32) | 1x | - | - |
| FP16 半精度 | 1.5~2x | 极小 | 低 |
| INT8 量化 | 2~4x | 小 | 中 |
| ONNX Runtime | 1.5~3x | 极小 | 低 |
| 模型剪枝 (50%) | 1.2~2x | 小 | 中 |
| 模型蒸馏 | 1.5~5x | 小 | 高 |
| TensorRT | 3~10x | 小 | 中 |
| 批处理 (batch=32) | 5~20x | - | 低 |

---

## 1. 模型量化

量化通过减少模型权重和激活的数值精度来加速推理。

### 1.1 PyTorch INT8 量化

```python
import torch
import torchvision.models as models

# 加载预训练模型
model = models.resnet50(pretrained=True)
model.eval()

# 动态量化（训练后量化）
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear, torch.nn.Conv2d},
    dtype=torch.qint8
)

# 静态量化（需要校准数据）
model.qconfig = torch.quantization.get_default_qconfig('x86')
torch.quantization.prepare(model, inplace=True)

# 校准数据
calibration_data = torch.randn(100, 3, 224, 224)
with torch.no_grad():
    for batch in calibration_data.chunk(10):
        model(batch)

torch.quantization.convert(model, inplace=True)
```

### 1.2 TensorFlow 量化

```python
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 动态范围量化
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# 全整数量化（需要校准数据集）
def representative_data_gen():
    for input_value in x_train.take(100):
        yield [np.expand_dims(input_value, axis=0)]

converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
tflite_quant_model = converter.convert()
```

**性能数据**：ResNet50 在 CPU 上
- FP32: 100ms/张
- INT8: 35ms/张
- 模型大小: 97MB → 25MB

---

## 2. 模型剪枝

剪枝通过移除模型中不重要的连接来减小模型大小。

```python
import torch
import torch.nn.utils.prune as prune

model = models.resnet50(pretrained=True)

# 对 Conv2d 层进行结构化剪枝
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.ln_structured(
            module,
            name='weight',
            amount=0.5,  # 剪枝 50%
            n=2,
            dim=0
        )

# 移除剪枝掩码，使剪枝永久化
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.remove(module, 'weight')

# 微调剪枝后的模型以恢复精度
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
for epoch in range(5):
    train_one_epoch(model, optimizer)
```

**性能数据**：
- 剪枝 50% 后：模型大小减少 40%，推理速度提升 15~30%
- 精度损失通常 < 1%（经过微调后）

---

## 3. ONNX Runtime 加速

ONNX Runtime 提供跨平台的高性能推理引擎。

```python
import onnxruntime as ort
import torch
import numpy as np

# 导出 PyTorch 模型为 ONNX
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    "resnet50.onnx",
    opset_version=12,
    input_names=['input'],
    output_names=['output']
)

# 创建 ONNX Runtime 会话
session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# 使用所有可用的优化
session = ort.InferenceSession(
    "resnet50.onnx",
    sess_options=session_options,
    providers=['CPUExecutionProvider']  # 或 'CUDAExecutionProvider'
)

# 推理
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
outputs = session.run(None, {'input': input_data})
```

**优化配置**：
```python
# 启用所有优化
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session_options.enable_profiling = True

# 设置线程数
session_options.intra_op_num_threads = 8
session_options.inter_op_num_threads = 4
```

**性能数据**：BERT-base 推理
- 原始 PyTorch: 45ms
- ONNX Runtime: 20ms
- 启用量化后: 12ms

---

## 4. 批处理推理

批处理充分利用硬件的并行计算能力。

```python
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time

class BatchProcessor:
    def __init__(self, model, batch_size=32, max_wait_time=0.01):
        self.model = model
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.request_queue = []
        self.results = {}
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.executor.submit(self._process_loop)

    def _process_loop(self):
        while True:
            if len(self.request_queue) >= self.batch_size:
                self._process_batch()
            elif self.request_queue:
                time.sleep(self.max_wait_time)
                if self.request_queue:
                    self._process_batch()

    def _process_batch(self):
        batch = self.request_queue[:self.batch_size]
        self.request_queue = self.request_queue[self.batch_size:]

        inputs = torch.cat([req['input'] for req in batch], dim=0)
        with torch.no_grad():
            outputs = self.model(inputs)

        for i, req in enumerate(batch):
            self.results[req['id']] = outputs[i:i+1]

    def predict(self, x):
        request_id = id(x)
        self.request_queue.append({'id': request_id, 'input': x})
        
        while request_id not in self.results:
            time.sleep(0.001)
        
        result = self.results.pop(request_id)
        return result
```

**性能对比**：ResNet50 在 V100 GPU 上

| Batch Size | 吞吐量 (img/s) | 延迟 (ms/img) |
|-----------|---------------|--------------|
| 1 | 300 | 3.3 |
| 8 | 1500 | 5.3 |
| 32 | 3500 | 9.1 |
| 64 | 4200 | 15.2 |

---

## 5. 模型蒸馏

用大模型（教师）指导小模型（学生）训练，获得紧凑高效的模型。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def __init__(self, temperature=3.0, alpha=0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, student_logits, teacher_logits, labels):
        # 蒸馏损失
        distillation_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1),
            reduction='batchmean'
        ) * (self.temperature ** 2)

        # 真实标签损失
        classification_loss = F.cross_entropy(student_logits, labels)

        return self.alpha * distillation_loss + (1 - self.alpha) * classification_loss

# 蒸馏训练示例
teacher_model = models.resnet50(pretrained=True)
student_model = models.resnet18()

criterion = DistillationLoss(temperature=3.0, alpha=0.7)
optimizer = torch.optim.Adam(student_model.parameters())

teacher_model.eval()
for epoch in range(20):
    student_model.train()
    for inputs, labels in dataloader:
        with torch.no_grad():
            teacher_logits = teacher_model(inputs)
        
        student_logits = student_model(inputs)
        loss = criterion(student_logits, teacher_logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

**典型效果**：
- 教师模型 (ResNet50): 76.1% acc, 97MB, 100ms
- 学生模型 (ResNet18): 74.5% acc, 46MB, 45ms
- 蒸馏后学生模型: 75.5% acc, 46MB, 45ms

---

## 6. GPU 推理优化 (TensorRT)

TensorRT 是 NVIDIA 的深度学习推理优化器。

```python
import torch
import torch_tensorrt

# 编译模型为 TensorRT
trt_model = torch_tensorrt.compile(
    model,
    inputs=[torch_tensorrt.Input((1, 3, 224, 224))],
    enabled_precisions={torch.float32, torch.half},  # 启用 FP16
    workspace_size=1 << 30,  # 1GB 工作空间
    truncate_long_and_double=True
)

# 保存和加载
torch.jit.save(trt_model, "resnet50_trt.ts")
loaded_model = torch.jit.load("resnet50_trt.ts")

# 推理
with torch.no_grad():
    output = loaded_model(input_tensor)
```

**多精度优化**：
```python
# 启用 INT8 量化（需要校准）
calibrator = torch_tensorrt.ptq.DataLoaderCalibrator(
    calibration_dataloader,
    cache_file="./calibration.cache",
    use_cache=False,
    algo_type=torch_tensorrt.ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2
)

trt_model_int8 = torch_tensorrt.compile(
    model,
    inputs=[torch_tensorrt.Input((1, 3, 224, 224))],
    enabled_precisions={torch.int8},
    calibrator=calibrator
)
```

**性能对比**：ResNet50 在 A100 上

| 精度 | 延迟 | 吞吐量 |
|-----|------|--------|
| FP32 | 1.8ms | 556 img/s |
| FP16 | 0.8ms | 1250 img/s |
| INT8 (TensorRT) | 0.4ms | 2500 img/s |

---

## 7. 推理服务部署

使用 FastAPI 结合批处理队列构建高性能推理服务。

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
import torch
import uuid
from typing import List, Dict
import time

app = FastAPI()

class InferenceRequest(BaseModel):
    data: List[float]
    shape: List[int]

class BatchInferenceService:
    def __init__(self, model, batch_size=32, max_wait_ms=10):
        self.model = model
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms
        self.request_queue: asyncio.Queue = asyncio.Queue()
        self.results: Dict[str, torch.Tensor] = {}
        self.worker_task = None

    async def start(self):
        self.worker_task = asyncio.create_task(self._batch_worker())

    async def stop(self):
        if self.worker_task:
            self.worker_task.cancel()

    async def predict(self, data: torch.Tensor) -> torch.Tensor:
        request_id = str(uuid.uuid4())
        future = asyncio.Future()
        
        await self.request_queue.put({
            'id': request_id,
            'data': data,
            'future': future
        })
        
        return await future

    async def _batch_worker(self):
        while True:
            batch = []
            start_time = time.time()
            
            # 收集批次，直到达到批量大小或超时
            while len(batch) < self.batch_size:
                elapsed = (time.time() - start_time) * 1000
                if elapsed >= self.max_wait_ms and batch:
                    break
                
                try:
                    item = await asyncio.wait_for(
                        self.request_queue.get(),
                        timeout=0.001
                    )
                    batch.append(item)
                except asyncio.TimeoutError:
                    if batch:
                        break
            
            if not batch:
                await asyncio.sleep(0.001)
                continue

            # 执行批处理推理
            tensors = torch.cat([item['data'] for item in batch], dim=0)
            with torch.no_grad():
                outputs = self.model(tensors)

            # 返回结果
            for i, item in enumerate(batch):
                item['future'].set_result(outputs[i:i+1])

# 服务初始化
model = models.resnet50(pretrained=True).eval()
inference_service = BatchInferenceService(model, batch_size=32, max_wait_ms=10)

@app.on_event("startup")
async def startup_event():
    await inference_service.start()

@app.on_event("shutdown")
async def shutdown_event():
    await inference_service.stop()

@app.post("/predict")
async def predict(request: InferenceRequest):
    try:
        tensor = torch.tensor(request.data).reshape(request.shape)
        result = await inference_service.predict(tensor)
        return {"predictions": result.flatten().tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

**服务性能优化技巧**：

1. **动态批处理**：根据请求量自动调整批大小
2. **请求合并**：将多个小请求合并处理
3. **结果缓存**：对相同输入缓存结果
4. **异步处理**：使用 asyncio 提高并发
5. **连接池**：复用数据库等外部连接

**典型部署架构**：

```
客户端 → 负载均衡 (Nginx) → FastAPI 服务实例 × N
                          ↓
                      批处理队列
                          ↓
                      GPU/CPU 推理
                          ↓
                      结果返回
```

---

## 综合优化策略

在实际项目中，通常需要组合多种优化技术：

### 推荐优化路径

1. **基础优化**：
   - 使用 ONNX Runtime
   - 启用 FP16 半精度
   - 实现批处理推理

2. **进阶优化**：
   - 模型量化 (INT8)
   - 模型剪枝 + 微调
   - 使用 TensorRT (GPU)

3. **极致优化**：
   - 模型蒸馏
   - 算子融合
   - 自定义 CUDA 核

### 注意事项

- 始终在真实数据和硬件上做性能基准测试
- 监控精度变化，权衡速度和精度的平衡
- 考虑优化成本（开发时间 vs 性能提升）
- 定期重新评估优化策略（硬件和软件都在快速迭代）

---

## 参考工具

- **性能分析**：PyTorch Profiler, TensorRT Profiler, NVIDIA Nsight
- **优化框架**：ONNX Runtime, TensorRT, OpenVINO, TVM
- **部署框架**：FastAPI, TorchServe, TensorFlow Serving, Triton
- **监控工具**：Prometheus, Grafana, OpenTelemetry
