from abc import ABC, abstractmethod
from typing import Dict
from typing import Union

from texttable import Texttable


class Statistics(ABC):
    @abstractmethod
    def print(self) -> str:
        pass

    def get_flat_statistics_dict(self) -> Dict[str, Union[int, float, str]]:
        # Can use nncf.utils.objwalk to recursively walk through self and
        # get all data fields, or register each field for this function to work
        # in the __init__, or make a metaclass to do this for us etc., anyway,
        # it is technically possible; the string name will be built based on
        # the field's position in the statistics struct, or be assigned during
        # registration at __init__
        pass


# Note that this and the following structs are not Statistics,
# since they are only used to group the entries in the global statistic,
# but they can be made Statistics if we want to print separate groups separately


class QuantizationShareStats:
    def __init__(self):
        # Here I illustrate the approach with external field initialization
        # __init__ signature can be extended to accept arguments and default them as None
        # if strict __init__-based initialization is required
        # None marks uncollected metrics, can use a special Statistics.NOT_COLLECTED marker instead
        self.num_total_weight_quantizers = None  # type: int
        self.num_total_activation_quantizers = None  # type: int

        # I illustrated the flat field structure here, but it can
        # be made a Dict[QuantizerCounterType, int] instead,
        # where QuantizerCounterType is an enum taking values
        # such as SYMMETRIC, ASYMMETRIC, SIGNED, ...
        self.num_symmetric_wq = None  # type: int
        self.num_asymmetric_wq = None  # type: int
        self.num_symmetric_aq = None  # type: int
        self.num_asymmetric_aq = None  # type: int
        self.num_signed_aq = None  # type: int
        self.num_unsigned_aq = None  # type: int
        self.num_signed_wq = None  # type: int
        self.num_unsigned_wq = None  # type: int
        self.num_per_tensor_aq = None  # type: int
        self.num_per_tensor_wq = None  # type: int
        self.num_potential_wq = None  # type: int
        self.num_potential_aq = None  # type: int

        self.num_wq_per_bitwidth = None  # type: Dict[int, int]
        self.num_aq_per_bitwidth = None  # type: Dict[int, int]

# Another grouping-only struct
class MemoryShareStats:
    def __init__(self):
        self.weight_memory_consumption_decrease = None # type: float
        self.fp32_weight_size = None  # type: int
        self.max_fp32_activation_size = None # type: int
        self.max_compressed_activation_size = None # type: int


class QuantizationStatistics(Statistics):
    def __init__(self):
        self.ratio_of_enabled_quantizations = None  # type: float
        # The grouping-only structs can be unrolled to make a completely flat
        # structure of the stats, I think that struct-based grouping will be more
        # readable
        self.quantization_share_stats = None  # type: QuantizationShareStats
        self.memory_share_stats = None  # type: MemoryShareStats
        self.quantized_edges_in_cfg = None  # type: int
        self.total_edges_in_cfg = None  # type: int

    def print(self) -> str:
        # Use these for pretty formatting. Can be made class attributes if we
        # want to make these string constants to be available to the user.
        NAME_STR = 'NetworkQuantizationShare'

        WEIGHTS_RATIO_STR = ' WQs / All placed WQs'  # WQ - weight quantizer
        ACTIVATIONS_RATIO_STR = ' AQs / All placed AQs'  # AQ - activation quantizer
        TOTAL_RATIO_STR = ' Qs (out of total placed)'

        PARAMS_STR = 'Quantizer parameter'
        SYMMETRIC_STR = 'Symmetric'
        ASYMMETRIC_STR = 'Asymmetric'
        PER_CHANNEL_STR = 'Per-channel'
        SIGNED_STR = 'Signed'
        PER_TENSOR_STR = 'Per-tensor'
        UNSIGNED_STR = 'Unsigned'
        SHARE_WEIGHT_QUANTIZERS_STR = 'Placed WQs / Potential WQs'
        SHARE_ACTIVATION_QUANTIZERS_STR = 'Placed AQs / Potential AQs'

        ENABLED_QUANTIZATIONS_RATIO_STR = "Ratio of enabled quantizations"

        retval = ""
        retval += '\n' + ENABLED_QUANTIZATIONS_RATIO_STR + ":" + str(self.ratio_of_enabled_quantizations)
        table_with_bits_stats = Texttable()
        table_with_other_stats = Texttable()

        # ... populate Texttable with data from self.
        # Note that Texttable is only used as an intermediate helper!

        retval += '\nQuantization shares statistics:\n' + table_with_other_stats.draw()
        retval += "\nBitwidth distribution:" + table_with_bits_stats.draw()
        return retval
