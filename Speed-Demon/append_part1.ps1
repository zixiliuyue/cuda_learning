$content = @'

for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
        for (int k = 0; k < N; k++)
            C[i][j] += A[i][k] * B[k][j];

// 性能好，可读性差（tiling + 循环展开 + 向量化）
__shared__ float smA[32][32], smB[32][32];
// ... 50行复杂的优化代码
```

我的平衡策略：
- 关键热点代码：可以牺牲可读性，写详细注释
- 其他代码：保持简洁，优先可维护性
- 避免"为了优化而优化"

**3. 性能 vs 通用性/灵活性**

| 方案 | 性能 | 通用性 |
|------|------|--------|
| 硬编码常量 | 快 | 只能处理特定大小 |
| 模板参数 | 较快 | 编译时确定 |
| 动态参数 | 较慢 | 完全灵活 |

**4. 精度 vs 性能**

- FP32 → FP16：性能提升2-4倍，精度损失可接受
- FP16 → INT8：性能再提升2倍，需要量化校准
- 需要根据业务SLA选择合适的精度

'@

Add-Content -Path "F:\code\cuda_learning\Speed-Demon\15_interview_simulation\README.md" -Value $content -Encoding UTF8
