import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV

# ===============================
# 1. 训练 + 校准模型
# ===============================
train = pd.read_csv(
    r'D:/临床病例/胃癌/数据汇总/训练数据/东营+省医-训练数据-0.65-smote800-train.csv',
    encoding='gbk'
)

X_train = train.drop(columns=['label']).values
y_train = train['label'].values

base_model = LGBMClassifier(
    learning_rate=0.1,
    n_estimators=165,
    verbose=-1
)

model = CalibratedClassifierCV(
    base_model,
    method='isotonic',
    cv=3
)

model.fit(X_train, y_train)

# ===============================
# 2. 净效益函数（GC vs non-GC）
# ===============================
def calculate_net_benefit(y_true, y_prob, threshold):
    tp = np.sum((y_prob >= threshold) & (y_true == 1))
    fp = np.sum((y_prob >= threshold) & (y_true == 0))
    n = len(y_true)
    return (tp / n) - (fp / n) * (threshold / (1 - threshold))


# ===============================
# 3. 外部 DCA → 只生成表格
# ===============================
def external_dca_table(
    X, y,
    save_excel,
    gc_label=4
):
    # 二分类标签
    y_true = (y == gc_label).astype(int)
    y_prob = model.predict_proba(X)[:, gc_label]

    thresholds = np.arange(0.01, 1.00, 0.01)

    # 模型净效益
    net_benefit_model = [
        calculate_net_benefit(y_true, y_prob, t)
        for t in thresholds
    ]

    # All / None
    prevalence = np.mean(y_true)
    net_benefit_all = [
        prevalence - (1 - prevalence) * (t / (1 - t))
        for t in thresholds
    ]
    net_benefit_none = [0.0] * len(thresholds)

    # 保存为 Excel
    df = pd.DataFrame({
        'Threshold': thresholds,
        'NetBenefit_Model': net_benefit_model,
        'NetBenefit_All': net_benefit_all,
        'NetBenefit_None': net_benefit_none
    })

    df.to_excel(save_excel, index=False)
    print(f'DCA 表格已保存：{save_excel}')


# ===============================
# 4. 外部验证 ① 省医
# ===============================
predict1 = pd.read_csv(
    r'D:/临床病例/胃癌/数据汇总/外部验证/省医-外部验证.csv'
)

X1 = predict1.drop(columns=['label']).values
y1 = predict1['label'].values

external_dca_table(
    X1,
    y1,
    save_excel=r'D:/临床病例/胃癌/数据汇总/Results/DCA_external_shengyi.xlsx'
)

# ===============================
# 5. 外部验证 ② 东营（胜利油田）
# ===============================
predict2 = pd.read_csv(
    r'D:/临床病例/胃癌/数据汇总/胜利油田/东营-test1.csv',
    encoding='gbk'
)

X2 = predict2.drop(columns=['label']).values
y2 = predict2['label'].values

external_dca_table(
    X2,
    y2,
    save_excel=r'D:/临床病例/胃癌/数据汇总/Results/DCA_external_dongying.xlsx'
)
