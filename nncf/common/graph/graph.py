"""
 Copyright (c) 2021 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import os
from abc import ABC
from abc import abstractmethod
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Callable, Dict, KeysView, List, Tuple, ValuesView, Type

import networkx as nx
from networkx.drawing.nx_agraph import to_agraph
import networkx.algorithms.isomorphism as iso

from nncf.common.graph.module_attributes import BaseLayerAttributes
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.operator_metatypes import get_input_metatypes
from nncf.common.graph.operator_metatypes import get_output_metatypes
from nncf.common.graph.module_attributes import Dtype
from nncf.common.utils.logger import logger as nncf_logger
from nncf.common.graph.graph_matching import Expression
from nncf.common.graph.graph_matching import NodeExpression
from nncf.common.graph.graph_matching import get_edge_boundaries
from nncf.common.graph.graph_matching import search_all

MODEL_INPUT_OP_NAME = "nncf_model_input"
MODEL_OUTPUT_OP_NAME = "nncf_model_output"

NNCFNodeName = str


class NNCFNode:
    """
    Class describing nodes used in NNCFGraph.
    """

    def __init__(self,
                 node_id: int,
                 data: dict = None):
        self.node_id = node_id
        self.data = data if data else {}

    @property
    def node_name(self) -> str:
        return self.data.get(NNCFGraph.NODE_NAME_ATTR)

    @property
    def node_type(self) -> str:
        return self.data.get(NNCFGraph.NODE_TYPE_ATTR)

    @property
    def metatype(self) -> Type[OperatorMetatype]:
        return self.data.get(NNCFGraph.METATYPE_ATTR)

    @property
    def module_attributes(self) -> BaseLayerAttributes:
        return self.data.get(NNCFGraph.MODULE_ATTRIBUTES)

    @property
    def ignored_algorithms(self) -> List[str]:
        return self.data.get(NNCFGraph.IGNORED_ALGOS_ATTR, [])

    def is_in_iteration_scope(self) -> bool:
        return self.data.get(NNCFGraph.IS_IN_ITERATION_SCOPE_NODE_ATTR, False)

    def is_integer_input(self) -> bool:
        return self.data.get(NNCFGraph.IS_INTEGER_INPUT_NODE_ATTR, False)

    def __str__(self):
        return ' '.join([str(self.node_id), self.node_name, self.node_type])

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return isinstance(other, NNCFNode) \
               and self.node_id == other.node_id \
               and self.data == other.data \
               and self.node_type == other.node_type \
               and self.module_attributes == other.module_attributes


class NNCFGraphNodeType:
    INPUT_NODE = MODEL_INPUT_OP_NAME
    OUTPUT_NODE = MODEL_OUTPUT_OP_NAME


class NNCFGraphEdge:
    def __init__(self, from_node: NNCFNode, to_node: NNCFNode, tensor_shape: List[int]):
        self.from_node = from_node
        self.to_node = to_node
        self.tensor_shape = tensor_shape

    def __str__(self):
        return str(self.from_node) + " -> " + str(self.tensor_shape) + " -> " + str(self.to_node)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return self.from_node == other.from_node and self.to_node == other.to_node \
               and self.tensor_shape == other.tensor_shape


class NNCFGraphPatternIO:
    """
    Describes the inputs and outputs of a subgraph in NNCFGraph.
    """
    def __init__(self, input_edges: List[NNCFGraphEdge], output_edges: List[NNCFGraphEdge],
                 input_nodes: List[NNCFNode],
                 output_nodes: List[NNCFNode],
                 ):
        self.input_edges = input_edges
        self.output_edges = output_edges
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes


class NNCFNodeExpression(NodeExpression):
    def __init__(self, node_type: str = None, filter_fn=None):
        node_type_fn = lambda x: x[NNCFGraph.NODE_TYPE_ATTR]
        super().__init__(node_type, filter_fn, node_type_fn=node_type_fn)


class NNCFGraph(ABC):
    """
    Wrapper over a regular directed acyclic graph that represents a control flow/execution graph of a DNN
    providing some useful methods for graph traversal.
    """

    ID_NODE_ATTR = 'id'
    KEY_NODE_ATTR = 'key'
    NODE_NAME_ATTR = 'node_name'
    NODE_TYPE_ATTR = 'type'
    METATYPE_ATTR = 'metatype'
    MODULE_ATTRIBUTES = 'module_attributes'
    ACTIVATION_SHAPE_EDGE_ATTR = 'activation_shape'
    IN_PORT_NAME_EDGE_ATTR = 'in_port'
    IGNORED_ALGOS_ATTR = 'ignored_algos'
    IS_IN_ITERATION_SCOPE_NODE_ATTR = 'is_in_iteration_scope'
    IS_INTEGER_INPUT_NODE_ATTR = 'is_integer_input'
    DTYPE_EDGE_ATTR = 'dtype'

    #pylint:disable=too-many-public-methods
    def __init__(self):
        self._nx_graph = nx.DiGraph()
        self._node_id_to_key_dict = dict()
        self._input_nncf_nodes = {}  # type: Dict[int, NNCFNode]
        self._output_nncf_nodes = {}  # type: Dict[int, NNCFNode]

    @abstractmethod
    def get_shared_nodes(self) -> List[List[NNCFNode]]:
        """Returns nodes that are associated with a single forward-ed layer in the model object"""

    @abstractmethod
    def is_shared_node(self, node: NNCFNode) -> bool:
        pass

    def get_node_by_id(self, node_id: int) -> NNCFNode:
        """
        :param node_id: Id of the node.
        :return: Node in a graph with such id.
        """
        return self.get_node_by_key(self.get_node_key_by_id(node_id))

    def get_node_by_key(self, key: str):
        """
        :param key: key (node_name) of the node.
        :return: NNCFNode in a graph with such key.
        """
        return self._nx_node_to_nncf_node(self._nx_graph.nodes[key])

    def get_input_nodes(self) -> List[NNCFNode]:
        """
        :return: List of input nodes of the graph.
        """
        return list(self._input_nncf_nodes.values())

    def get_output_nodes(self) -> List[NNCFNode]:
        """
        :return: List of output nodes of the graph.
        """
        return list(self._output_nncf_nodes.values())

    def get_nodes_by_types(self, type_list: List[str]) -> List[NNCFNode]:
        """
        :param type_list: List of types to look for.
        :return: List of nodes with provided types.
        """
        all_nodes_of_type = []
        for node_key in self.get_all_node_keys():
            nx_node = self._nx_graph.nodes[node_key]
            nncf_node = self._nx_node_to_nncf_node(nx_node)
            if nncf_node.node_type in type_list:
                all_nodes_of_type.append(nncf_node)
        return all_nodes_of_type

    def get_nodes_by_metatypes(self, metatype_list: List[Type[OperatorMetatype]]) -> List[NNCFNode]:
        """
        Return a list of nodes with provided metatypes.

        :param metatype_list: List of types to look for.
        :return: List of nodes with provided metatypes.
        """
        all_nodes_of_type = []
        for node_key in self.get_all_node_keys():
            nx_node = self._nx_graph.nodes[node_key]
            nncf_node = self._nx_node_to_nncf_node(nx_node)
            if nncf_node.metatype in metatype_list:
                all_nodes_of_type.append(nncf_node)
        return all_nodes_of_type


    def get_all_node_ids(self) -> KeysView[int]:
        """
        Returns all graph nodes' node_ids.
        """
        return self._node_id_to_key_dict.keys()

    def get_all_node_keys(self) -> ValuesView[str]:
        """
        Returns all graph nodes' keys i.e. node_names.
        """
        return self._node_id_to_key_dict.copy().values()

    def get_all_nodes(self) -> List[NNCFNode]:
        """
        Returns list of all graph nodes.
        """
        all_nodes = []
        for node_key in self.get_all_node_keys():
            nx_node = self._nx_graph.nodes[node_key]
            nncf_node = self._nx_node_to_nncf_node(nx_node)
            all_nodes.append(nncf_node)
        return all_nodes

    @staticmethod
    def _nx_node_to_nncf_node(nx_node: dict) -> NNCFNode:
        return NNCFNode(nx_node[NNCFGraph.ID_NODE_ATTR], nx_node)

    def get_node_key_by_id(self, node_id: id) -> str:
        """
        Returns node key (node_name) by provided id.

        :param node_id: Id of the node.
        :return: Key of the node with provided id.
        """
        return self._node_id_to_key_dict[node_id]

    def get_next_nodes(self, node: NNCFNode) -> List[NNCFNode]:
        """
        Returns consumer nodes of provided node.

        :param node: Producer node.
        :return: List of consumer nodes of provided node.
        """
        nx_node_keys = self._nx_graph.succ[self._node_id_to_key_dict[node.node_id]]
        return [self._nx_node_to_nncf_node(self._nx_graph.nodes[key]) for key in nx_node_keys]

    def get_previous_nodes(self, node: NNCFNode) -> List[NNCFNode]:
        """
        Returns producer nodes of provided node.

        :param node: Consumer node.
        :return: List of producers nodes of provided node.
        """

        nx_node_keys = self._nx_graph.pred[self._node_id_to_key_dict[node.node_id]]
        return [self._nx_node_to_nncf_node(self._nx_graph.nodes[key]) for key in nx_node_keys]

    def get_input_edges(self, node: NNCFNode) -> Dict[Tuple[str, str], dict]:
        """
        Returns edges of input tensors with description sorted by 'in_port'.

        :param node: Consumer node.
        :return: Dictionary of input edges for node sorted by in_port.
        """
        nx_node_key = self._node_id_to_key_dict[node.node_id]
        input_edges = sorted(list(self._nx_graph.in_edges(nx_node_key)),
                             key=lambda edge: self._nx_graph.edges[edge][NNCFGraph.IN_PORT_NAME_EDGE_ATTR])

        return OrderedDict((edge, self._nx_graph.edges[edge]) for edge in input_edges)

    def get_output_edges(self, node: NNCFNode) -> Dict[Tuple[str, str], dict]:
        """
        Returns edges of output tensors with description. Unordered.

        :param node: Producer node.
        :return: Dictionary of output edges for the node.
        """
        nx_node_key = self._node_id_to_key_dict[node.node_id]
        return {edge: self._nx_graph.edges[edge] for edge in self._nx_graph.out_edges(nx_node_key)}

    def traverse_graph(self,
                       curr_node: NNCFNode,
                       traverse_function: Callable[[NNCFNode, List[Any]], Tuple[bool, List[Any]]],
                       traverse_forward: bool = True):
        """
        Traverses graph up or down starting form `curr_node` node.

        :param curr_node: Node from which traversal is started.
        :param traverse_function: Function describing condition of traversal continuation/termination.
        :param traverse_forward: Flag specifying direction of traversal.
        :return:
        """
        output = []
        return self._traverse_graph_recursive_helper(curr_node, traverse_function, output, traverse_forward)

    def _traverse_graph_recursive_helper(self, curr_node: NNCFNode,
                                         traverse_function: Callable[[NNCFNode, List[Any]], Tuple[bool, List[Any]]],
                                         output: List[Any], traverse_forward: bool):
        is_finished, output = traverse_function(curr_node, output)
        get_nodes_fn = self.get_next_nodes if traverse_forward else self.get_previous_nodes
        if not is_finished:
            for node in get_nodes_fn(curr_node):
                self._traverse_graph_recursive_helper(node, traverse_function, output, traverse_forward)
        return output

    def add_nncf_node(self, node_name: str,
                      node_type: str,
                      node_metatype: Type[OperatorMetatype],
                      module_attributes: BaseLayerAttributes = None,
                      node_id_override: int = None,
                      ignored_algorithms: List[str] = None,
                      is_in_iteration_scope: bool = False,
                      is_integer_input: bool = False) -> NNCFNode:

        if node_id_override is not None:
            node_id = node_id_override
        else:
            node_ids = self.get_all_node_ids()
            if node_ids:
                node_id = max(self.get_all_node_ids()) + 1
            else:
                node_id = 0

        if node_id in self._node_id_to_key_dict:
            raise ValueError(f"NNCF node with id {node_id} is already in the NNCFGraph")

        node_key = f'{node_id} {node_name}'

        self._node_id_to_key_dict[node_id] = node_key
        attrs = {
            NNCFGraph.ID_NODE_ATTR: node_id,
            NNCFGraph.KEY_NODE_ATTR: node_key,
            NNCFGraph.NODE_NAME_ATTR: node_name,
            NNCFGraph.NODE_TYPE_ATTR: node_type,
            NNCFGraph.METATYPE_ATTR: node_metatype,
            NNCFGraph.IS_IN_ITERATION_SCOPE_NODE_ATTR: is_in_iteration_scope,
            NNCFGraph.IS_INTEGER_INPUT_NODE_ATTR: is_integer_input
        }
        if module_attributes is not None:
            attrs[NNCFGraph.MODULE_ATTRIBUTES] = module_attributes

        if ignored_algorithms is None:
            ignored_algorithms = []
        attrs[NNCFGraph.IGNORED_ALGOS_ATTR] = ignored_algorithms
        self._nx_graph.add_node(node_key, **attrs)

        node = NNCFNode(node_id, data=attrs)

        if node.metatype in get_input_metatypes():
            self._input_nncf_nodes[node_id] = node

        if node.metatype in get_output_metatypes():
            self._output_nncf_nodes[node_id] = node
        return node

    def add_edge_between_nncf_nodes(self, from_node_id: int, to_node_id: int,
                                    tensor_shape: List[int],
                                    input_port_id: int,
                                    dtype: Dtype):
        from_node_key = self._node_id_to_key_dict[from_node_id]
        to_node_key = self._node_id_to_key_dict[to_node_id]

        err_reason = None

        if from_node_key not in self._nx_graph.nodes:
            err_reason = f"node {from_node_key} not in NNCFGraph"
        if to_node_key not in self._nx_graph.nodes:
            err_reason = f"node {from_node_key} not in NNCFGraph"
        if from_node_id in self._output_nncf_nodes:
            err_reason = "cannot add edges *from* output nodes"
        if to_node_id in self._input_nncf_nodes:
            err_reason = "cannot add edges *to* input nodes"

        if err_reason is not None:
            raise ValueError(f"Cannot add edge from {from_node_key} to {to_node_key} - {err_reason}!")

        attrs = {
            NNCFGraph.ACTIVATION_SHAPE_EDGE_ATTR: tensor_shape,
            NNCFGraph.IN_PORT_NAME_EDGE_ATTR: input_port_id,
            NNCFGraph.DTYPE_EDGE_ATTR: dtype
        }
        self._nx_graph.add_edge(from_node_key, to_node_key, **attrs)

    def topological_sort(self) -> List[NNCFNode]:
        """
        Returns nodes in topologically sorted order, additionally sorted in ascending node ID order.
        """
        return [self._nx_node_to_nncf_node(self._nx_graph.nodes[node_name])
                for node_name in
                nx.lexicographical_topological_sort(self._nx_graph,
                                                    key=lambda x: self._nx_graph.nodes[x][NNCFGraph.ID_NODE_ATTR])]

    def dump_graph(self, path):
        nx.drawing.nx_pydot.write_dot(self.get_graph_for_structure_analysis(), path)

    def visualize_graph(self, path):
        out_graph = self._get_graph_for_visualization()
        nx.drawing.nx_pydot.write_dot(out_graph, path)
        try:
            A = to_agraph(out_graph)
            A.layout('dot')
            png_path = os.path.splitext(path)[0]+'.png'
            A.draw(png_path)
        except ImportError:
            nncf_logger.warning("Graphviz is not installed - only the .dot model visualization format will be used. "
                                "Install pygraphviz into your Python environment and graphviz system-wide to enable "
                                "PNG rendering.")

    def get_graph_for_structure_analysis(self, extended=False) -> nx.DiGraph:
        """The graph to dump has certain node attributes omitted, compared to the graph stored
         inside NNCFGraph."""
        out_graph = nx.DiGraph()
        for node_name, node in self._nx_graph.nodes.items():
            attrs_node = {
                'id': node[NNCFGraph.ID_NODE_ATTR],
                'type': node[NNCFGraph.NODE_TYPE_ATTR]
            }
            if 'color' in node:
                attrs_node['color'] = node['color']
            if 'label' in node:
                attrs_node['label'] = node['label']
            if 'style' in node:
                attrs_node['style'] = node['style']

            out_graph.add_node(node_name, **attrs_node)
        if extended:
            for u, v in self._nx_graph.edges:
                out_graph.add_edge(u, v, label=self._nx_graph.edges[u, v][NNCFGraph.ACTIVATION_SHAPE_EDGE_ATTR])
        else:
            for u, v in self._nx_graph.edges:
                out_graph.add_edge(u, v)

        return out_graph

    def _get_graph_for_visualization(self) -> nx.DiGraph:
        """A user-friendly graph .dot file, making it easier to debug the network and setup
        ignored/target scopes."""
        out_graph = nx.DiGraph()
        for node_name, node in self._nx_graph.nodes.items():
            attrs_node = {}
            attrs_node['label'] = str(node[NNCFGraph.ID_NODE_ATTR]) + ' ' + str(NNCFGraph.NODE_NAME_ATTR)
            out_graph.add_node(node_name, **attrs_node)

        for u, v in self._nx_graph.edges:
            out_graph.add_edge(u, v, label=self._nx_graph.edges[u, v][NNCFGraph.ACTIVATION_SHAPE_EDGE_ATTR])

        mapping = {k: v["label"] for k, v in out_graph.nodes.items()}
        out_graph = nx.relabel_nodes(out_graph, mapping)
        for node in out_graph.nodes.values():
            node.pop("label")

        return out_graph

    def get_node_by_name(self, name: NNCFNodeName) -> NNCFNode:
        matches = []
        for nx_node in self._nx_graph.nodes.values():
            if nx_node[NNCFGraph.NODE_NAME_ATTR] == name:
                matches.append(nx_node)
        if not matches:
            raise RuntimeError("Could not find a node {} in NNCFGraph!".format(name))
        if len(matches) > 1:
            raise RuntimeError("More than one node in NNCFGraph matches name {}:\n{}".
                               format(name,
                                      '\t\n'.join(
                                          [n[NNCFGraph.KEY_NODE_ATTR] for n in matches])))
        return self._nx_node_to_nncf_node(next(iter(matches)))

    def __eq__(self, other: 'NNCFGraph'):
        nm = iso.categorical_node_match([NNCFGraph.ID_NODE_ATTR,
                                         NNCFGraph.KEY_NODE_ATTR,
                                         NNCFGraph.MODULE_ATTRIBUTES], [None, None, None])
        em = iso.categorical_edge_match([NNCFGraph.ACTIVATION_SHAPE_EDGE_ATTR,
                                         NNCFGraph.IN_PORT_NAME_EDGE_ATTR], [None, None])
        return nx.is_isomorphic(self._nx_graph, other._nx_graph, node_match=nm, edge_match=em)

    def get_nx_graph_copy(self) -> nx.DiGraph:
        return deepcopy(self._nx_graph)

    def get_matching_nncf_graph_pattern_io_list(self, expression: Expression) -> List[NNCFGraphPatternIO]:
        matched_node_key_sequences = search_all(self._nx_graph, expression)
        pattern_ios = [self.get_nncf_graph_pattern_io_list(match) for match in matched_node_key_sequences]
        return pattern_ios

    def _get_nncf_graph_pattern_input_output(self, match: List[str]) -> NNCFGraphPatternIO:
        out_edge_boundary = list(nx.edge_boundary(self._nx_graph, match, data=True))
        complement = list(filter(lambda x: x not in match, self._nx_graph.nodes.keys()))
        in_edge_boundary = list(nx.edge_boundary(self._nx_graph, complement, data=True))
        boundary = in_edge_boundary + out_edge_boundary
        input_nncf_edges = []
        output_nncf_edges = []
        input_nncf_nodes = []
        output_nncf_nodes = []
        for key in match:
            # Currently we treat the nodes without incoming edges as "input" and the nodes without
            # outcoming edges as "output".
            # A proper way to find the input nodes would be to mark the tensors arriving at NNCFNetwork's
            # "forward" as input, then drop the marking once the first operation with an input tensor
            # has been done; the node corresponding to this operation would be "input" by definition.
            # Same with output nodes - should check the model output for TracedTensors and mark the
            # nodes from which such tensors originated as "output".
            # TODO: implement the functionality above.
            if not list(self._nx_graph.successors(key)):
                output_nncf_nodes.append(self._nx_node_to_nncf_node(self._nx_graph.nodes[key]))
            if not list(self._nx_graph.predecessors(key)):
                input_nncf_nodes.append(self._nx_node_to_nncf_node(self._nx_graph.nodes[key]))

        for nx_edge in boundary:
            from_node_key = nx_edge[0]
            to_node_key = nx_edge[1]
            data = nx_edge[2]
            nncf_edge = NNCFGraphEdge(self._nx_node_to_nncf_node(self._nx_graph.nodes[from_node_key]),
                                      self._nx_node_to_nncf_node(self._nx_graph.nodes[to_node_key]),
                                      data[NNCFGraph.ACTIVATION_SHAPE_EDGE_ATTR])
            if from_node_key in match:
                output_nncf_edges.append(nncf_edge)
            elif to_node_key in match:
                input_nncf_edges.append(nncf_edge)
            else:
                raise RuntimeError("Invalid graph expression supplied!")

        return NNCFGraphPatternIO(input_nncf_edges, output_nncf_edges,
                                  input_nncf_nodes, output_nncf_nodes)

    def get_nncf_graph_pattern_io_list(self, match: List[str]) -> NNCFGraphPatternIO:
        """Returns an NNCFGraphPatternIO object that describes the input/output nodes and edges of a
        subgraph specified by `match`.

        :param match: A list of node keys specifying a subgraph to be matched. The subgraph to be matched will
        consist of nodes with the same keys that are connected with edges in the order they are listed in the
        `match` list
        :return: NNCFGraphPatternIO object describing the inputs and outputs of the matched subgraph
        """
        in_edge_boundary, out_edge_boundary = get_edge_boundaries(match, self._nx_graph)
        boundary = in_edge_boundary + out_edge_boundary
        input_nncf_edges = []
        output_nncf_edges = []
        input_nncf_nodes = []
        output_nncf_nodes = []
        for key in match:
            node_id = self._nx_graph.nodes[key][NNCFGraph.ID_NODE_ATTR]
            if node_id in self._input_nncf_nodes:
                input_nncf_nodes.append(self._nx_node_to_nncf_node(self._nx_graph.nodes[key]))
            if node_id in self._output_nncf_nodes:
                output_nncf_nodes.append(self._nx_node_to_nncf_node(self._nx_graph.nodes[key]))

        for nx_edge in boundary:
            from_node_key = nx_edge[0]
            to_node_key = nx_edge[1]
            data = nx_edge[2]
            nncf_edge = NNCFGraphEdge(self._nx_node_to_nncf_node(self._nx_graph.nodes[from_node_key]),
                                      self._nx_node_to_nncf_node(self._nx_graph.nodes[to_node_key]),
                                      data[NNCFGraph.ACTIVATION_SHAPE_EDGE_ATTR])
            if from_node_key in match:
                output_nncf_edges.append(nncf_edge)
            elif to_node_key in match:
                input_nncf_edges.append(nncf_edge)
            else:
                raise RuntimeError("Invalid graph expression supplied!")

        return NNCFGraphPatternIO(input_nncf_edges, output_nncf_edges,
                                  input_nncf_nodes, output_nncf_nodes)

    # pylint:disable=protected-access
    def get_nx_edge(self, node_u: NNCFNode, node_v: NNCFNode):
        nx_node_u = self._nx_graph._node[self._node_id_to_key_dict[node_u.node_id]]
        nx_node_v = self._nx_graph._node[self._node_id_to_key_dict[node_v.node_id]]
        return self._nx_graph.edges[nx_node_u['key'], nx_node_v['key']]

    def get_nodes_count(self):
        return self._nx_graph.number_of_nodes()
