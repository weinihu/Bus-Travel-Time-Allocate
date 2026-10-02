# 原始文献核查与相关工作改写

核查日期：2026-10-01。范围：当前 IEEE 中英文稿的相关工作与书目。本报告区分原文方法、当前问题条件与写作判断，不提供新的方法排名。

## 1. 当前问题与比较依据

查询输入为已知有序有向公交站间区间、允许的上下文和本次行程总时长。站间区间可以跨多个道路链路，不能直接等同于原文的原子 road link。当前模型共享区间身份与上下文对应的条件基量，在具体行程内施加相对基量有界、加和为零的修正，最后按比例缩放到给定总时长。总时长只进入最终缩放。拟合与模型选择分别使用训练集、验证集中有限可见的单区间标签；当前查询没有局部真值输入。

比较按四项展开：查询时已知什么、局部监督从何而来、跨行程共享什么、输出的一致性如何保证。经典分解已经处理已知总量，概率 TTE 已经学习共享链路表示。因此本文的定位是特定观测条件下的分配设计与实证分析，不能据此声称首次提出总量分解或普遍精度优势。

## 2. 核查来源与书目信息

| 文献及引用键 | 原始来源与核验信息 | 阅读范围与定位 |
|---|---|---|
| Hellinga, Izadpanah, Takada, Fu (2008), `hellinga2008` | *Decomposing travel times measured by probe-based traffic monitoring systems to individual road segments*. Transportation Research Part C, 16(6), 768–782. DOI [10.1016/j.trc.2008.04.002](https://doi.org/10.1016/j.trc.2008.04.002). [作者所在大学 PDF](https://www.civil.uwaterloo.ca/itss/papers/2008-1%20%28Decomposing%20travel%20times%29.pdf). | 已阅读本地原文 `/tmp/hellinga2008-source.txt`。重点：§3.1–3.3，pp.770–775，式(3)–(16)；§4例子；§5–7实验、敏感性与结论。 |
| Hofleitner, Bayen (2011), `hofleitner2011` | *Optimal decomposition of travel times measured by probe vehicles using a statistical traffic flow model*. IEEE ITSC, 815–821. DOI [10.1109/ITSC.2011.6083050](https://doi.org/10.1109/ITSC.2011.6083050). [作者团队 PDF](https://bayen.berkeley.edu/sites/default/files/06083050.pdf). | 已阅读下载原文 `/tmp/hofleitner2011-source.pdf` 及其文本。重点：§II交通流假设与命题1–2；§III式(1)–(6)；§IV算法与 NGSIM 验证。 |
| Xu, Wang, Sun (2025), `xu2025probeta` | *Link Representation Learning for Probabilistic Travel Time Estimation*. IEEE TITS, **26(11), 21149–21161**. DOI [10.1109/TITS.2025.3590075](https://doi.org/10.1109/TITS.2025.3590075). [作者最新稿](https://arxiv.org/abs/2407.05895), [PDF](https://arxiv.org/pdf/2407.05895). [出版社提交的元数据](https://api.crossref.org/works/10.1109/TITS.2025.3590075). | 已阅读本地 `/tmp/probtte-source.txt`，首页为 arXiv v2，2026-01-26，模型名 **ProbETA**。重点：§III–IV，式(2)–(18)；§V表II–IV、图4–5与条件预测实验。 |
| Wang, Zhang, Cao, Li, Zheng (2018), `wang2018deeptte` | *When Will You Arrive? Estimating Travel Time Based on Deep Neural Networks*. AAAI, 32(1), 2500–2507. DOI [10.1609/aaai.v32i1.11877](https://doi.org/10.1609/aaai.v32i1.11877). [出版方论文页](https://ojs.aaai.org/index.php/AAAI/article/view/11877), [PDF](https://ojs.aaai.org/index.php/AAAI/article/download/11877/11736). | 已阅读 `/tmp/deeptte2018-source.pdf` 与文本。重点：Preliminary，p.2501；Local Path定义，p.2502；Multi-task Learning，p.2503；Model Training式(6)–(8)，p.2504；p.2506局部任务消融。 |
| Zhang, Charoenphakdee, Wu, Sugiyama (2020), `zhang2020aggregate` | *Learning from Aggregate Observations*. NeurIPS 33, **7993–8005**. [出版方全文](https://papers.nips.cc/paper/2020/file/5b0fa0e4c041548bb6289e15d865a696-Paper.pdf), [出版方 BibTeX](https://papers.nips.cc/paper/2020/file/5b0fa0e4c041548bb6289e15d865a696-Bibtex.bib). 无 DOI 条目，保留官方 URL。 | 已阅读 `/tmp/aggregate2020-source.pdf` 的问题定义、假设、似然、等价关系理论及均值回归：§2–5，尤其§2.2条件独立假设、§3式(2)–(4)、§5.3式(9)–(11)。不把其定理直接转用于公交顺序区间。 |
| Wickramasuriya, Athanasopoulos, Hyndman (2019), `wickramasuriya2019mint` | *Optimal Forecast Reconciliation for Hierarchical and Grouped Time Series Through Trace Minimization*. JASA **114(526), 804–819**. DOI [10.1080/01621459.2018.1448825](https://doi.org/10.1080/01621459.2018.1448825). [作者稿](https://robjhyndman.com/papers/mint.pdf), [期刊元数据](https://api.crossref.org/works/10.1080/01621459.2018.1448825). | 阅读作者稿的引言与§2：§2.1加和矩阵和预测协调式(1)–(2)；§2.3误差协方差、无偏性与定理1。作者稿日期早于正式2019卷期，引用采用正式书目信息。 |
| Athanasopoulos, Hyndman, Kourentzes, Petropoulos (2017), 未新增引用 | *Forecasting with Temporal Hierarchies*. EJOR 262(1), 60–74. DOI [10.1016/j.ejor.2017.02.046](https://doi.org/10.1016/j.ejor.2017.02.046). [作者稿](https://robjhyndman.com/papers/temporalhierarchies.pdf). | 阅读作者稿引言、§3–4：非重叠时间聚合、不同频率的独立预测、式(5)–(8)协调。本文没有多频率未来时间序列，正文采用更直接的 MinT 比较，避免第三篇新增引用。 |
| Yin, Wang, Wang, Adams (2015), `yin2015entryexit` | *Link travel time inference using entry/exit information of trips on a network*. TR-B 80, 303–321. DOI [10.1016/j.trb.2015.07.007](https://doi.org/10.1016/j.trb.2015.07.007). [出版方摘要](https://www.sciencedirect.com/science/article/abs/pii/S019126151500154X), [出版社元数据](https://api.crossref.org/works/10.1016/j.trb.2015.07.007). | 此次核验出版方摘要与作者/题名/卷页，全文未取得。正文只保留摘要明确列出的高斯似然、未知路径混合估计与一般分布 trip splitting，不补写未读算法公式。 |
| Sun, Zhao, Zhang, Chen, Yu (2022), `sun2022prltte` | *PR-LTTE: Link travel time estimation based on path recovery from large-scale incomplete trip data*. Information Sciences 589, 34–45. DOI [10.1016/j.ins.2021.12.091](https://doi.org/10.1016/j.ins.2021.12.091). [出版方摘要](https://www.sciencedirect.com/science/article/pii/S0020025521013128), [出版社元数据](https://api.crossref.org/works/10.1016%2Fj.ins.2021.12.091). | 此次核验出版方摘要与书目信息，全文未取得。只据摘要表述路径恢复依赖行程距离/耗时，与非负最小二乘交替进行。 |
| Mao et al. (2025), `mao2025dutytte` | *DutyTTE: Deciphering Uncertainty in Origin-Destination Travel Time Estimation*. AAAI 39(12), 12390–12398. DOI [10.1609/aaai.v39i12.33350](https://doi.org/10.1609/aaai.v39i12.33350). [出版方论文页](https://ojs.aaai.org/index.php/AAAI/article/view/33350). | 此次核验出版方摘要和九位作者名单。正文限于摘要说明的路径对齐与上下文相关区间不确定性；不声称完成全文复现。 |

## 3. 最接近原文的具体技术差异

### Hellinga 2008：拥堵与停止时间的似然加权分配

原文假定地图匹配与路径识别已经完成，首尾位置可以位于部分链路内部。自由流时间由长度与自由流速度给出；其余时间分为拥堵与停止时间。§3.3.1假定当前短路径各链路拥堵程度相近，并利用同一车辆此前有效采样区间构造当前拥堵程度的似然，式(10)。§3.3.2以相对下游位置和拥堵程度构造停止似然，式(11)–(13)，且假定一次观测路径至多停止一次。式(14)–(15)对拥堵程度积分，并按似然权重分配停止与拥堵耗时，式(16)将各项合并。

因此，尽管摘要采用 likelihood maximization 的概括，相关工作依据实际求解公式写成“似然模型与积分分配”，不把它改写为普通独立高斯 MAP/MLE 优化。当前模型没有部分道路链路位置、自由流速度或相邻轮询拥堵参数输入；其修正也是统计分配形状，不能解释成真实停止或拥堵成分。原文仿真中的改进率不与当前数据分数作横向比较。

本地定位：`/tmp/hellinga2008-source.txt:295` 为§3.3.1，`:363` 为§3.3.2，pp.774–775为积分与合并公式。

### Hofleitner 2011：交通流混合分布与条件凸子问题

§II假定周期性、短时平稳的干道交通、三角基本图、驾驶自由流 pace 的对数凹分布，并推导延迟分布。卷积后，每个部分链路时间是至多三个对数凹分量的混合。§III假定链路时间相互独立，在已知路径总时间下最大化分配的似然。整体混合目标非凸；固定延迟模式或 EM 权重后，M步子问题凸。枚举遍历模式，Hard EM 每次选择最可能模式；EM与Hard EM使用多起点，只有局部收敛保证。Given stop 算法另需停止位置观测。

原文要求道路网络与信号位置，但不要求先验提供信号配时、自由流速度或流量，这些分布参数可以由历史数据估计。不能将“不要求先验提供”写成“没有交通流参数”。当前方法通过有限单区间标签学习上下文基量和零和修正，没有原文的交通流混合分布、延迟模式推断或 EM；NNLS-Ridge也不构成该论文复现。

本地定位：`/tmp/hofleitner2011-source.txt:105` 为不适定性例子，`:131` 为假设，`:230` 为混合分布，`:319` 为求解算法，`:342` 为实验。正式页码815–821；方法集中在816–819。

### ProbETA：共享均值、相关随机效应与完成行程条件预测

最新作者稿§IV-B式(2)定义 `t = shared mean + day effect + trip effect`；日期效应描述行程间相关，行程效应描述一次行程内的相关。§IV-C将链路分布经路径关联矩阵聚合成多行程联合高斯，以边缘似然学习参数。协方差采用低秩及低秩加对角结构。不同时间窗口有对应链路表示；不是令所有时段链路均值相同。§IV-D利用GPS时间戳生成子行程，不等于直接取得每条道路链路的可靠标签。§IV-E式(17)–(18)根据同日时空邻近已完成行程更新日期效应，再推断查询行程的分布。

与当前问题的联系在于从重复局部身份共享统计信息，而非单纯“它预测总量、我们分解总量”。关键差异是：ProbETA学习分布、协方差和随机效应，推断借助其他完成行程；当前方法学习本次查询的确定性份额，以其已知总量缩放，修正有相对幅度和行程内零和约束。当前有限单区间标签与原文GPS子行程监督也不是同一标签预算。零均值随机效应是分布期望性质，不等同于每次实现都在行程内加和为零。

本地定位：`/tmp/probtte-source.txt:292` 为§IV-B，`:398` 为§IV-C，`:445` 起为时段表示与§IV-D，`:598` 附近为§IV-E；式(17)–(18)在PDF第7页。期刊书目信息使用2025年，最新作者稿上传日期不改变期刊出版年。旧v1名为ProbTTE；当前稿与论文引用统一使用ProbETA。若历史结果中的适配器使用旧仓库名称，须单独交代适配名称，不能据仓库名重命名原论文。

### DeepTTE：GPS局部窗口辅助监督与整条路径预测

模型从位置序列与行程属性学习，Geo-Conv窗口形成local path，后接LSTM。局部头学习GPS窗口端点时间戳差，整条路径头以注意力汇聚预测总耗时；两项损失加权训练。滑动局部窗口可以重叠，默认整条路径输出并非将局部预测相加。当前单区间标签是有向公交站间时间，非GPS窗口时间；已知总量输入和严格加和也需要改变其原始输出接口。因此引用可支持路线表示与局部辅助学习，不支持把经过改造的比较行称为原论文任务下的直接复现。

本地定位：`/tmp/deeptte2018-source.txt:191` 为Local Path定义，`:267` 为多任务组件，`:304` 为训练，式(6)–(8)在p.2504。

## 4. 聚合学习与一致性约束的作用位置

Zhang等的§2–3由个体条件分布推出聚合观测的似然，并据此学习个体预测器。均值回归在高斯与条件独立设定下得到组均值平方误差，§5.3式(10)；其一致性命题针对该数据生成与模型设定，不能直接转用于顺序相关的公交区间。当前输出按总量归一化后，最终总和误差恒为零；它只保证可行性，不可作为单独训练信号。正文据此明确有限单区间拟合标签和有限验证标签的作用，不使用“只有总量便能学习分解”的说法。

MinT的输入为不同层级的未来 base forecasts，利用预测误差协方差和加和矩阵协调全部预测；其最优性依赖无偏等条件。本文输入一个已观测总量，并学习其内部份额，未估计MinT误差协方差，也不具备MinT最小方差保证。时间层级论文则处理同一时间序列在非重叠聚合频率上的协调，和公交一次行程的不同站间成分不同。保留MinT这一项引用足以说明加和约束的既有研究背景。

本地定位：`/tmp/aggregate2020-source.txt:136` 为假设，`:187` 为似然，`:355` 为均值回归；`/tmp/mint2019-source.txt:352` 为MinT，`:399` 为定理1；`/tmp/temporal2017-source.txt:414` 为时间协调。

## 5. 写作结构与改写决定

原文中可借鉴的是论证组织，不是措辞。Hellinga先界定观测处理流程中分配所处的步骤，再分解需估计的时间成分；Hofleitner先给交通假设和可解结构，明确例子中的不唯一性；DeepTTE把局部与整体预测的冲突落到训练头与损失；ProbETA按共享参数、随机效应、训练似然与条件推断逐项展开；聚合学习与MinT分别交代监督如何产生梯度和约束如何改变输出。

当前相关工作沿用这种具体论证次序：

1. 已观测时间分解：说明Hellinga和Hofleitner用什么额外规律完成分配，再连接共享链路反演与已知站间序列。
2. 旅行时间表示学习：说明DeepTTE的标签粒度，重点比较ProbETA共享表示、随机效应和推断信息；保留DutyTTE的路径不确定性背景。
3. 聚合监督与加和一致性：区分会产生学习误差的聚合监督、强制输出加和、协方差预测协调；以当前同输入、同标签预算和同缩放的比较结束。

中英文按同一事实与论证组织重写，不逐字翻译。使用观测、输入、标签、分布、路径、估计、分配等具体名词，避免空泛的“先进框架”“显著创新”以及反复声明不可比。没有引用原文精度改善数字，也没有扩展到作者认证、审稿结果或全面文献首创性结论。

## 6. 已修正的问题与最终引用键

- Hellinga此前被笼统概括为“条件最大似然分配”，未体现上一采样区间及似然积分。改为按§3.3公式说明拥堵与停止分配。
- Hofleitner此前的“凸分配”容易被读为整体混合问题凸。改为整体非凸、固定模式凸子问题。
- ProbETA此前只写“内外依赖”，未说明共享均值、日期/行程随机效应、子行程监督和完成行程条件输入。补充这些直接差异。
- ProbETA书目补全TITS 26(11), 21149–21161；书目URL改为正式DOI。论文模型名保持ProbETA；不以旧v1或仓库名替代。
- DeepTTE此前只说联合局部监督。补充GPS时间戳窗口、可重叠与独立整条路径预测头。
- 添加训练和验证各自的有限单区间标签说明，以及归一化后聚合误差不产生监督的原因。

原有且保留的相关工作引用键：`hellinga2008`、`hofleitner2011`、`yin2015entryexit`、`sun2022prltte`、`wang2018deeptte`、`xu2025probeta`、`mao2025dutytte`。

仅新增两项：`zhang2020aggregate`、`wickramasuriya2019mint`。书目数据分别依据NeurIPS官方BibTeX和期刊DOI元数据。未新增时间层级第三项或软件/skill引用。

修改文件为当前Overleaf包 `sections/en/related.tex`、`sections/zh/related.tex`、`completion_references_20260919.bib`、`additional_references.bib`。报告同时复制至包根目录，方便离线核查。
