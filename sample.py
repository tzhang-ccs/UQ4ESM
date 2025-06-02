from framework import uq

para_name = 'para_set.json'
foo = uq(para_name)
foo.lhs_sample(0,continue_run=False,continue_id=20)
foo.analyse('lhs_sample')
