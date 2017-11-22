import yaml


print yaml.__version__
print yaml.__file__

cfg = []

""" For loader tuple"""
class YAMLPatch(yaml.SafeLoader):
    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))

YAMLPatch.add_constructor(u'tag:yaml.org,2002:python/tuple', YAMLPatch.construct_python_tuple)




""" Load File"""
with open("default_config.yaml", 'r') as stream:
    try:
        # cfg = yaml.safe_load(stream)
        cfg = yaml.load(stream, Loader=YAMLPatch)
        for a in cfg:
            print('{}:'.format(a))
            for b in cfg[a]:
                print('----{}:{} ({})'.format(b, cfg[a][b], type(cfg[a][b])))
        state_shape = tuple(cfg['RL']['state_shape'])
        print("state_shape={},type(state_shape)={}".format(state_shape, type(state_shape)))

    except yaml.YAMLError as exc:
        print(exc)

print('log' in cfg)