import yaml
document = """
  a: 1
  b:
    c: 3
    d: 4
"""

print yaml.dump(yaml.load(document))

config = yaml.load(document)

print('config[\'a\']={}'.format(config['a']))
print('config[\'b\']={}'.format(config['b']))
print('config[\'b\'][\'c\']={}'.format(config['b']['c']))
print('config[\'b\'][\'d\']={}'.format(config['b']['d']))
# print('config.b.c={}'.format(config.b.c))


