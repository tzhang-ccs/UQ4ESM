from framework import uq

para_name = 'para_set.json'
foo = uq(para_name)

foo.analyse(method='TurBO')