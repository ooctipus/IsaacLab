
from rl_games.common import object_factory
import torch.nn as nn
from rl_games.algos_torch.d2rl import D2RLNet

class AuxInit(nn.Module):
    def __init__(self, in_dim, aux_params):
        super().__init__()  # <<< important
        self.activation_factory = self.build_activation_factory()

        self.aux_outputs = list(aux_params['aux_outputs'].keys())  # <<< you use this in forward
        mlp_cfg = aux_params['mlp']

        self.aux_mlp = self._build_mlp(
            input_size=in_dim,
            units=mlp_cfg['units'],
            activation=mlp_cfg['activation'],
            norm_func_name=aux_params.get('normalization', None),
            dense_func=nn.Linear,
            d2rl=mlp_cfg.get('d2rl', False),
            norm_only_first_layer=mlp_cfg.get('norm_only_first_layer', False),
        )

        # one head per aux output
        self.aux_heads = nn.ModuleDict()
        for name in self.aux_outputs:
            out_size = aux_params['aux_outputs'][name]['size']
            self.aux_heads[name] = nn.Sequential(
                nn.Linear(mlp_cfg['units'][-1], out_size),
                self.activation_factory.create(mlp_cfg['out_activation']),
            )

    def forward(self, in_data):
        x = self.aux_mlp(in_data)
        self.last_aux_out = {name: head(x) for name, head in self.aux_heads.items()}
        return self.last_aux_out

    def _build_mlp(self, input_size, units, activation, dense_func,
                   norm_only_first_layer=False, norm_func_name=None, d2rl=False):
        if d2rl:
            # NOTE: correct attribute name is activation_factory (no "s")
            act_layers = [self.activation_factory.create(activation) for _ in range(len(units))]
            return D2RLNet(input_size, units, act_layers, norm_func_name)
        # plain sequential MLP
        in_size = input_size
        layers = []
        need_norm = True
        for u in units:
            layers.append(dense_func(in_size, u))
            layers.append(self.activation_factory.create(activation))
            if need_norm and norm_func_name is not None:
                if norm_only_first_layer:
                    need_norm = False
                if norm_func_name == 'layer_norm':
                    layers.append(nn.LayerNorm(u))
                elif norm_func_name == 'batch_norm':
                    layers.append(nn.BatchNorm1d(u))
            in_size = u
        return nn.Sequential(*layers)

    def build_activation_factory(self):
        af = object_factory.ObjectFactory()
        af.register_builder('relu', lambda **kw: nn.ReLU(**kw))
        af.register_builder('tanh', lambda **kw: nn.Tanh(**kw))
        af.register_builder('sigmoid', lambda **kw: nn.Sigmoid(**kw))
        af.register_builder('elu', lambda **kw: nn.ELU(**kw))
        af.register_builder('selu', lambda **kw: nn.SELU(**kw))
        af.register_builder('swish', lambda **kw: nn.SiLU(**kw))
        af.register_builder('gelu', lambda **kw: nn.GELU(**kw))
        af.register_builder('softplus', lambda **kw: nn.Softplus(**kw))
        af.register_builder('None', lambda **kw: nn.Identity())
        return af
