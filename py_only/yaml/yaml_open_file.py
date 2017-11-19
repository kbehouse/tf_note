import yaml

with open("config.yaml", 'r') as stream:
    try:
        # print(yaml.load(stream))
        config = yaml.load(stream)

        print('config[\'a\']={}'.format(config['a']))
        print('config[\'b\']={}'.format(config['b']))
        print('config[\'b\'][\'c\']={}'.format(config['b']['c']))
        print('config[\'b\'][\'d\']={}'.format(config['b']['d']))
    except yaml.YAMLError as exc:
        print(exc)

