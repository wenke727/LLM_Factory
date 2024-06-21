from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

task = Tasks.token_classification
model = 'damo/mgeo_geographic_composition_analysis_chinese_base'
inputs = '浙江省杭州市余杭区阿里巴巴西溪园区'
pipeline_ins = pipeline(task=task, model=model)

res = pipeline_ins(input=inputs).get('output')


# 输出
# {'output': [{'type': 'PB', 'start': 0, 'end': 3, 'span': '浙江省'}, {'type': 'PC', 'start': 3, 'end': 6, 'span': '杭州市'}, {'type': 'PD', 'start': 6, 'end': 9, 'span': '余杭区'}, {'type': 'Entity', 'start': 9, 'end': 17, 'span': '阿里巴巴西溪园区'}]}