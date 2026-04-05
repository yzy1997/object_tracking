# -*- coding: utf-8 -*-
"""生成包含参数和评价指标的Word文档"""
import os
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.table import WD_TABLE_ALIGNMENT

# 创建文档
doc = Document()

# 标题
title = doc.add_heading('轨迹关联算法比较 - 参数与评价指标', 0)
title.alignment = 1  # 居中

# 1. GNN参数
doc.add_heading('1. GNN (Global Nearest Neighbor) 参数', level=1)
gnn_params = doc.add_table(rows=10, cols=2)
gnn_params.style = 'Light Grid Accent 1'

gnn_params.rows[0].cells[0].text = '参数名称'
gnn_params.rows[0].cells[1].text = '数值'

gnn_params.rows[1].cells[0].text = 'GATING_DISTANCE'
gnn_params.rows[1].cells[1].text = '75.0'

gnn_params.rows[2].cells[0].text = 'COST_UNMATCHED'
gnn_params.rows[2].cells[1].text = '1e5'

gnn_params.rows[3].cells[0].text = 'MAX_MISSED'
gnn_params.rows[3].cells[1].text = '40'

gnn_params.rows[4].cells[0].text = 'MIN_HITS_TO_CONFIRM'
gnn_params.rows[4].cells[1].text = '1'

gnn_params.rows[5].cells[0].text = 'MAX_TRACKS_KEEP'
gnn_params.rows[5].cells[1].text = '55'

gnn_params.rows[6].cells[0].text = 'PROCESS_NOISE_POS'
gnn_params.rows[6].cells[1].text = '35.0'

gnn_params.rows[7].cells[0].text = 'PROCESS_NOISE_VEL'
gnn_params.rows[7].cells[1].text = '18.0'

gnn_params.rows[8].cells[0].text = 'MEASUREMENT_NOISE_POS'
gnn_params.rows[8].cells[1].text = '40.0'

gnn_params.rows[9].cells[0].text = 'WH_SMOOTH_GNN'
gnn_params.rows[9].cells[1].text = '0.04'

# 2. MHT参数
doc.add_heading('2. MHT (Multiple Hypothesis Tracking) 参数', level=1)
mht_params = doc.add_table(rows=3, cols=2)
mht_params.style = 'Light Grid Accent 1'

mht_params.rows[0].cells[0].text = '参数名称'
mht_params.rows[0].cells[1].text = '数值'

mht_params.rows[1].cells[0].text = 'max_hypotheses'
mht_params.rows[1].cells[1].text = '80'

mht_params.rows[2].cells[0].text = 'GATING (GNN_GATING_DISTANCE * 3.5)'
mht_params.rows[2].cells[1].text = '75.0 * 3.5 = 262.5'

# 3. IMM-MHT参数
doc.add_heading('3. IMM-MHT (IMM + MHT) 参数', level=1)
imm_params = doc.add_table(rows=9, cols=2)
imm_params.style = 'Light Grid Accent 1'

imm_params.rows[0].cells[0].text = '参数名称'
imm_params.rows[0].cells[1].text = '数值'

imm_params.rows[1].cells[0].text = 'IMM_GATING'
imm_params.rows[1].cells[1].text = '12.0'

imm_params.rows[2].cells[0].text = 'IMM_Q0_POS'
imm_params.rows[2].cells[1].text = '0.05'

imm_params.rows[3].cells[0].text = 'IMM_Q0_VEL'
imm_params.rows[3].cells[1].text = '0.01'

imm_params.rows[4].cells[0].text = 'IMM_Q1_POS'
imm_params.rows[4].cells[1].text = '0.5'

imm_params.rows[5].cells[0].text = 'IMM_Q1_VEL'
imm_params.rows[5].cells[1].text = '0.2'

imm_params.rows[6].cells[0].text = 'IMM_R_POS'
imm_params.rows[6].cells[1].text = '1.0'

imm_params.rows[7].cells[0].text = 'WH_SMOOTH'
imm_params.rows[7].cells[1].text = '0.92'

imm_params.rows[8].cells[0].text = 'max_hypotheses'
imm_params.rows[8].cells[1].text = '12'

# 4. 评价指标结果
doc.add_heading('4. 评价指标结果', level=1)

# 添加结果表格
results_table = doc.add_table(rows=4, cols=6)
results_table.style = 'Light Grid Accent 1'

# 表头
results_table.rows[0].cells[0].text = '算法'
results_table.rows[0].cells[1].text = 'RMSE (m)'
results_table.rows[0].cells[2].text = '漏检率 (%)'
results_table.rows[0].cells[3].text = 'ID Switch (%)'
results_table.rows[0].cells[4].text = '误跟率 (%)'
results_table.rows[0].cells[5].text = '失跟率 (%)'

# 数据行
results_table.rows[1].cells[0].text = 'GNN'
results_table.rows[1].cells[1].text = '0.94'
results_table.rows[1].cells[2].text = '0.0'
results_table.rows[1].cells[3].text = '26.6'
results_table.rows[1].cells[4].text = '7.9'
results_table.rows[1].cells[5].text = '0.0'

results_table.rows[2].cells[0].text = 'MHT'
results_table.rows[2].cells[1].text = '0.67'
results_table.rows[2].cells[2].text = '0.0'
results_table.rows[2].cells[3].text = '7.5'
results_table.rows[2].cells[4].text = '8.9'
results_table.rows[2].cells[5].text = '0.0'

results_table.rows[3].cells[0].text = 'IMM-MHT'
results_table.rows[3].cells[1].text = '0.50'
results_table.rows[3].cells[2].text = '0.0'
results_table.rows[3].cells[3].text = '4.5'
results_table.rows[3].cells[4].text = '11.0'
results_table.rows[3].cells[5].text = '0.0'

# 5. 指标说明
doc.add_heading('5. 评价指标说明', level=1)

doc.add_paragraph('• RMSE (Root Mean Square Error): 跟踪误差的均方根，单位为米，反映定位精度')
doc.add_paragraph('• 漏检率: 未能检测到的真实目标占比')
doc.add_paragraph('• ID Switch: 目标ID发生切换的次数占比，反映跟踪稳定性')
doc.add_paragraph('• 误跟率: 错误关联的检测占比')
doc.add_paragraph('• 失跟率: 目标丢失的占比')

# 6. 结论
doc.add_heading('6. 结论', level=1)
doc.add_paragraph('IMM-MHT算法在RMSE指标上表现最佳（0.50m），明显优于GNN（0.94m）和MHT（0.67m）。')
doc.add_paragraph('GNN算法由于只考虑最近邻关联，且使用较大的gating范围，导致关联误差最大。')
doc.add_paragraph('MHT算法通过多假设生成能够部分改善跟踪效果，但仍受限于单一CV模型。')
doc.add_paragraph('IMM-MHT结合了交互式多模型（IMM）和多假设跟踪（MHT），能够自适应目标运动模式变化，因此跟踪精度最高。')

# 保存文档
output_path = r'D:\codes\object_tracking\results\association_comparison\algorithm_comparison.docx'
doc.save(output_path)
print(f"Word文档已保存至: {output_path}")