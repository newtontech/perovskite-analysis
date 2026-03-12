# Issue #9 验收检查点清单

> 研究时间：2026-03-12  
> 目标：确保项目质量可控

---

## 一、数据质量检查点

### ✅ 检查项
- [ ] **数据量** ≥ 50条有效记录
- [ ] **关键字段完整率** > 95%
- [ ] **异常值标注** 已完成
- [ ] **数据分布** 无严重偏斜
- [ ] **物理一致性** 无违背基础物理的数据

### 验证方法
```python
def check_data_quality(data):
    checks = {
        'min_samples': len(data) >= 50,
        'completeness': data[key_fields].isna().mean() < 0.05,
        'outliers_labeled': 'outlier_flag' in data.columns,
        'distribution': check_distribution(data),
        'physics_consistency': check_physics(data)
    }
    return all(checks.values())
```

---

## 二、模型性能检查点

### ✅ 检查项
- [ ] **R² ≥ 0.85**（测试集）
- [ ] **MAE < 1%**
- [ ] **MAPE < 8%**
- [ ] **无过拟合**（训练-测试差距 < 10%）
- [ ] **交叉验证稳定性**（CV标准差 < 5%）

### 验证方法
```python
def check_model_performance(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    metrics = {
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'mape': mean_absolute_percentage_error(y_test, y_pred),
        'no_overfitting': check_overfitting(model),
        'cv_stability': check_cv_stability(model)
    }
    
    return metrics
```

---

## 三、可解释性检查点

### ✅ 检查项
- [ ] **Top 5特征与文献一致**
- [ ] **SHAP图可解读**
- [ ] **特征方向符合物理直觉**
- [ ] **无"异常"发现（或已解释）**
- [ ] **因果关系已验证**

### 验证方法
1. 对比SHAP排序与文献中的关键参数
2. 请领域专家审核SHAP图
3. 检查每个特征的SHAP值方向

4. 调查任何"意外"发现

---

## 四、实际效用检查点

### ✅ 检查项
- [ ] **预测配方实验验证** ≥ 3组
- [ ] **预测误差 < 10%**（80%以上）
- [ ] **实验次数减少** ≥ 40%
- [ ] **发现新材料** ≥ 1个候选
- [ ] **用户满意度** > 80%

### 验证方法
1. 选择3-5个预测最优配方
2. 实验室合成测试
3. 对比预测与实测PCE
4. 统计实验次数对比

---

## 五、代码质量检查点

### ✅ 检查项
- [ ] **测试覆盖率** ≥ 80%
- [ ] **代码规范** 通过lint
- [ ] **文档完整** 所有函数有docstring
- [ ] **类型提示** 关键函数有type hints
- [ ] **可复现** 随机种子固定

### 验证方法
```bash
# 测试覆盖率
pytest --cov=src --cov-fail-under=80

# 代码规范
pylint src/

# 文档检查
pydocstyle src/
```

---

## 六、交付物检查点

### ✅ 必须交付
- [ ] **研究报告** 完整的Markdown文档
- [ ] **代码仓库** 可运行的Python代码
- [ ] **模型文件** 训练好的模型
- [ ] **使用文档** README和示例
- [ ] **测试报告** 评估结果

---

## 七、最终验收流程

```
1. 自检 → 各检查项通过
2. 内审 → 团队成员审核
3. 外审 → 领域专家评审
4. 实验验证 → 实验室确认
5. 最终批准 → 项目负责人签字
```

---

*研究完成时间：2026-03-12*
