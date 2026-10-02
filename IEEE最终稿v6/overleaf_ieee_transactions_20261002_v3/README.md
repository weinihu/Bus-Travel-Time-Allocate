# IEEE Transactions 中英文修订稿（2026-10-02，v3）

v3同步了主文实验预算、修复了优化器确认表，并在补充材料报告该组配对结果及其条件行程Bootstrap区间。摘要、结果与结论明确区分点估计、配对重复和独立测试；主文按预算验证选参结果保持不变。

主文五节：Introduction、Related Work、Methodology、Experiments、Conclusion。没有Discussion或占位章节。`main_zh.tex`为中文阅读稿，`main.tex`为英文稿；补充材料分别为`supplementary_zh.tex`与`supplementary.tex`。使用官方未修改的IEEEtran类和参考文献样式，中文采用ctex/Fandol字体。

中文题目为“已知总时长与同段信息共享下的分段运行时间估计”。中英文摘要、引言、方法、实验解释和结论已同步重写；引言使用三段行程的动机例子，引出同段共享和当前行程修正。正文采用训练、验证、评估等通用术语，随机数编号与文件状态仅保留在复现记录。修改前源码备份为相邻的`overleaf_ieee_transactions_20261001_before_submission_revision`。

框架图位置按老师修改意见暂留空白，保留图号与引用；原框架图仅作本地备份，不显示在当前正文中。替换前版本另存于相邻的`overleaf_ieee_transactions_20261001_before_framework_slot`目录。

Overleaf上传本包后，将编译器设为XeLaTeX，选择对应主文件。未覆盖原版或在线Overleaf工程。作者、单位及目标期刊尚未提供，因此作者字段为空，不虚构个人信息。

此前新增27次训练包括5%预算下两种神经方法各九次及修正参数化控制九次，并计算六次简单基线。本轮再补36次10%多布局消融与参数化训练：三个布局×三个配对初始化，共九次观测/条件；独立审计门已关闭。关闭行程修正和打乱日历分别使MAE增加0.405秒和0.377秒，九组配对方向一致；去除日历字段的MAE略低0.062秒，参数化差异仍很小。既有覆盖和训练预算结果继续单独报告。

方法对比图的柱内纹理已移除。消融改为配对误差差值图，以零线表示无变化，同时保留逐次观测、均值与样本标准差；绝对误差仍完整保留在表格中。可比表格按同预算、同条件加粗最优值。

优化配置筛选包括8次筛选和54次配对确认训练；后续按验证集分别选择三个预算的修正幅度，并确认ExtraTrees和固定权重组合。更新后的Ours/直接预测MAE为32.607/33.186、29.383/29.457、28.776/28.481秒；配对九次中Ours分别胜8、6、0次。固定权重Ours+Direct组合的MAE为31.785、28.783、28.065秒，但这是单独的集成结果，不代表Ours单模型性能。1%时同段均值仍更优；Ours在10%时不及直接预测，长区间MAE三个预算均较高。以上均为开发集结果，不能据此声称Ours总体最优或测试集泛化。

`evidence/`保存结果审计报告与封存状态。`figure_data/`含绘图数据和生成程序，`figures/`含论文图。`MANIFEST.json`校验交付文件，`VALIDATION.json`记录本轮编译与版面检查。训练预测、检查点、配置及源码快照保存在实验仓库的`results/submission_revision_20261001`、`results/submission_ablation_extension_20261002`和`results/paper_performance_20261002`，不附原始轨迹。

研究结果属于探索性评价，本文方法尚无相对同输入直接预测的稳定精度优势。独立测量运行总时长、独立外部测试、完整同任务基线及Singapore来源许可仍需补充；作者和单位字段待作者填写。框架图按用户要求预留空位，替换后再生成投稿PDF。

官方模板来源：
https://journals.ieeeauthorcenter.ieee.org/wp-content/uploads/sites/7/IEEE-Transactions-LaTeX2e-templates-and-instructions.zip

主要数据来源：
https://zenodo.org/records/15769359

本地复现入口（实验仓库）：

```bash
python -m pytest tests/test_submission_revision_20261001.py -q
python scripts/run_submission_revision_20261001.py budget --device cuda:2
python scripts/audit_submission_revision_20261001.py budget --device cuda:2
python scripts/run_submission_revision_20261001.py mechanism --device cuda:2
python scripts/audit_submission_revision_20261001.py mechanism --device cuda:2
python scripts/audit_submission_ablation_extension_20261002.py --device cuda:2
python scripts/build_submission_figures_20261001.py
python -m scripts.build_submission_local_cases_20261001
```

模型编号、随机数状态及哈希在JSON中记录，用于复现，不在论文中逐项罗列。已有完整结果会校验配置和源码，真正重跑应使用新输出目录。

写作参考使用K-Dense的`scientific-writing`规范，侧重证据与表述一致、段落主题、方法和结果对应，未使用AI检测规避服务。规范来源为https://github.com/K-Dense-AI/scientific-agent-skills/tree/main/skills/scientific-writing 。该技能库的参考论文：Kassis, T., Agarwal, V., He, Y., Patel, D., and Brueckner, A. M., *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*, 2026, https://doi.org/10.48550/arXiv.2609.00065 。使用技能不构成文字作者身份或检测器结果的保证。
