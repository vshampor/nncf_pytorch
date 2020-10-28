import torch

from nncf import register_operator
from nncf.dynamic_graph.graph_builder import GraphBuilder, create_dummy_forward_fn, ModelInputInfo


def my_custom_unwrapped_operator(x):
    y = x * x
    y = y + y
    return y


@register_operator
def my_custom_wrapped_operator(x):
    y = x * x
    y = y + y
    return y


class MyTestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.use_wrapped = False
        self._dummy_param = torch.nn.Parameter(torch.ones([]))

    def forward(self, x):
        if self.use_wrapped:
            return my_custom_wrapped_operator(x)
        else:
            return my_custom_unwrapped_operator(x)


def test_register_operator():
    model = MyTestModel()
    builder = GraphBuilder(create_dummy_forward_fn([ModelInputInfo([1, 1, 1, 1])]))
    graph = builder.build_graph(model)
    model.use_wrapped = True
    graph_2 = builder.build_graph(model)
    pass

