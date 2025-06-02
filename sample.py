from framework import uq

para_name = 'para_set.json'
foo = uq(para_name)

foo.lhs_sample(500,continue_run=True,continue_id=2000)

foo.analyse('scm_sample')
