# import copy
# import inspect
from typing import List, Any

# Extension of from torch.testing._internal.common_device_type import dtypes
# to support device types for Tensors and PIL Image modes
import torch
# from torch.testing._internal.common_device_type import dtypes
# from torch.testing._internal.common_device_type import device_type_test_bases


# ------- Changed dtypes ----------------------------------------------------------------------
class dtypes(object):

    # Note: *args, **kwargs for Python2 compat.
    # Python 3 allows (self, *args, device_type='all').
    def __init__(self, *args, **kwargs):
        if len(args) > 0 and isinstance(args[0], (list, tuple)):
            for arg in args:
                assert isinstance(arg, (list, tuple)), \
                    "When one dtype variant is a tuple or list, " \
                    "all dtype variants must be. " \
                    "Received non-list non-tuple dtype {0}".format(str(arg))
                # This is the change vs original dtypes implementation
                assert all(self._is_supported_dtype(dtype) for dtype in arg), "Unknown dtype in {0}".format(str(arg))
        else:
            # This is the change vs original dtypes implementation
            assert all(self._is_supported_dtype(arg) for arg in args), "Unknown dtype in {0}".format(str(args))

        self.args = args
        self.device_type = kwargs.get('device_type', 'all')

    def __call__(self, fn):
        d = getattr(fn, 'dtypes', {})
        assert self.device_type not in d, "dtypes redefinition for {0}".format(self.device_type)
        d[self.device_type] = self.args
        fn.dtypes = d
        return fn

    # This is the change vs original dtypes implementation
    @staticmethod
    def _is_supported_dtype(dtype):
        if isinstance(dtype, torch.dtype):
            return True
        return False

# ------- End of changed dtypes ----------------------------------------------------------------------

class vision_dtypes(dtypes):

    @staticmethod
    def _is_supported_dtype(dtype):
        if isinstance(dtype, str):
            # https://pillow.readthedocs.io/en/stable/handbook/concepts.html?highlight=modes#modes
            modes = ["1", "L", "P", "RGB", "RGBA", "CMYK", "YCbCr", "LAB", "HSV", "I", "F"]
            return dtype in ["PIL.{}".format(mode) for mode in modes]
        return dtypes._is_supported_dtype(dtype)


class variants:
    def __init__(self, name: str, config_list: List):
        self.name = name
        self.config_list = config_list

    def __call__(self, fn):
        if not hasattr(fn, "variants"):
            setattr(fn, "variants", {})
        fn.variants[self.name] = self.config_list
        return fn


# ------- DeviceTypeTestBase dtypes ----------------------------------------------------------------------
import threading
from functools import wraps

from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.common_device_type import _construct_test_name, TEST_WITH_ROCM, get_all_dtypes, ClassVar


class DefaultInstantiator:

    attr_dispatcher_map = {}

    @staticmethod
    def instantiate_test_helper(cls, in_name, *, in_test, dtype, op, generic_cls, kwargs_dict=None):

        # Constructs the test's name
        test_name = _construct_test_name(in_name, op, cls.device_type, dtype)

        # wraps instantiated test with op decorators
        # NOTE: test_wrapper exists because we don't want to apply
        #   op-specific decorators to the original test.
        #   Test-specific decorators are applied to the original test,
        #   however.
        if op is not None and op.decorators is not None:
            @wraps(in_test)
            def test_wrapper(*args, **kwargs):
                return in_test(*args, **kwargs)

            for decorator in op.decorators:
                test_wrapper = decorator(test_wrapper)

            test_fn = test_wrapper
        else:
            test_fn = in_test

        if kwargs_dict is None:
            kwargs_dict = {}

        # Constructs the test
        @wraps(in_test)
        def instantiated_test(self, name=in_name, test=test_fn, dtype=dtype, op=op, kw_dict=kwargs_dict):
            if op is not None and op.should_skip(generic_cls.__name__, name,
                                                 self.device_type, dtype):
                self.skipTest("Skipped!")

            device_arg: str = cls.get_primary_device()
            if hasattr(test, 'num_required_devices'):
                device_arg = cls.get_all_devices()

            # Sets precision and runs test
            # Note: precision is reset after the test is run
            guard_precision = self.precision
            try:
                self.precision = self._get_precision_override(test, dtype)
                args = (arg for arg in (device_arg, dtype, op) if arg is not None)
                result = test(self, *args, **kw_dict)
            finally:
                self.precision = guard_precision

            return result

        assert not hasattr(cls, test_name), "Redefinition of test {0}".format(test_name)
        setattr(cls, test_name, instantiated_test)

    @staticmethod
    def _handle_default_test(cls, name, test, generic_cls=None, kwargs_dict=None):
        dtypes = cls._get_dtypes(test)
        dtypes = tuple(dtypes) if dtypes is not None else (None,)
        for dtype in dtypes:
            DefaultInstantiator.instantiate_test_helper(
                cls,
                name,
                in_test=test,
                dtype=dtype,
                op=None,
                generic_cls=generic_cls,
                kwargs_dict=kwargs_dict
            )

    @classmethod
    def instantiate_test(cls, test_cls, name, test, *, generic_cls=None):
        # Handles tests using decorators
        for attr_name, method_name in cls.attr_dispatcher_map.items():
            if hasattr(test, attr_name):
                fn = getattr(cls, method_name)
                fn(test_cls, name, test, generic_cls=generic_cls)
                return
        # Handles tests that don't use the ops decorator
        DefaultInstantiator._handle_default_test(test_cls, name, test, generic_cls=generic_cls)


class OpsDecoratorInstantiator:

    attr_dispatcher_map = {
        "op_list": "_handle_ops_decorator",
    }

    @staticmethod
    def _handle_ops_decorator(cls, name, test, generic_cls=None):
        for op in test.op_list:
            # Acquires dtypes, using the op data if unspecified
            dtypes = cls._get_dtypes(test)
            if dtypes is None:
                if cls.device_type == 'cpu' and op.dtypesIfCPU is not None:
                    dtypes = op.dtypesIfCPU
                elif (cls.device_type == 'cuda' and not TEST_WITH_ROCM
                      and op.dtypesIfCUDA is not None):
                    dtypes = op.dtypesIfCUDA
                elif (cls.device_type == 'cuda' and TEST_WITH_ROCM
                      and op.dtypesIfROCM is not None):
                    dtypes = op.dtypesIfROCM
                else:
                    dtypes = op.dtypes

            # Inverts dtypes if the function wants unsupported dtypes
            if test.unsupported_dtypes_only is True:
                dtypes = [d for d in get_all_dtypes() if d not in dtypes]
            dtypes = dtypes if dtypes is not None else (None,)
            for dtype in dtypes:
                DefaultInstantiator.instantiate_test_helper(
                    cls,
                    name,
                    in_test=test,
                    dtype=dtype,
                    op=op,
                    generic_cls=generic_cls
                )


class DeviceTypeTestBase(TestCase):
    device_type: str = 'generic_device_type'

    # Precision is a thread-local setting since it may be overridden per test
    _tls = threading.local()
    _tls.precision = TestCase._precision

    instantiator_cls = DefaultInstantiator

    @property
    def precision(self):
        return self._tls.precision

    @precision.setter
    def precision(self, prec):
        self._tls.precision = prec

    # Returns a string representing the device that single device tests should use.
    # Note: single device tests use this device exclusively.
    @classmethod
    def get_primary_device(cls):
        return cls.device_type

    # Returns a list of strings representing all available devices of this
    # device type. The primary device must be the first string in the list
    # and the list must contain no duplicates.
    # Note: UNSTABLE API. Will be replaced once PyTorch has a device generic
    #   mechanism of acquiring all available devices.
    @classmethod
    def get_all_devices(cls):
        return [cls.get_primary_device()]

    # Returns the dtypes the test has requested.
    # Prefers device-specific dtype specifications over generic ones.
    @classmethod
    def _get_dtypes(cls, test):
        if not hasattr(test, 'dtypes'):
            return None
        return test.dtypes.get(cls.device_type, test.dtypes.get('all', None))

    def _get_precision_override(self, test, dtype):
        if not hasattr(test, 'precision_overrides'):
            return self.precision
        return test.precision_overrides.get(dtype, self.precision)

    # Creates device-specific tests.
    @classmethod
    def instantiate_test(cls, name, test, *, generic_cls=None):
        cls.instantiator_cls.instantiate_test(cls, name, test, generic_cls=generic_cls)


class CPUTestBase(DeviceTypeTestBase):
    device_type = 'cpu'


class CUDATestBase(DeviceTypeTestBase):
    device_type = 'cuda'
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True
    primary_device: ClassVar[str]
    cudnn_version: ClassVar[Any]
    no_magma: ClassVar[bool]
    no_cudnn: ClassVar[bool]


    def has_cudnn(self):
        return not self.no_cudnn

    @classmethod
    def get_primary_device(cls):
        return cls.primary_device

    @classmethod
    def get_all_devices(cls):
        primary_device_idx = int(cls.get_primary_device().split(':')[1])
        num_devices = torch.cuda.device_count()

        prim_device = cls.get_primary_device()
        cuda_str = 'cuda:{0}'
        non_primary_devices = [cuda_str.format(idx) for idx in range(num_devices) if idx != primary_device_idx]
        return [prim_device] + non_primary_devices

    @classmethod
    def setUpClass(cls):
        # has_magma shows up after cuda is initialized
        t = torch.ones(1).cuda()
        cls.no_magma = not torch.cuda.has_magma

        # Determines if cuDNN is available and its version
        cls.no_cudnn = not torch.backends.cudnn.is_acceptable(t)
        cls.cudnn_version = None if cls.no_cudnn else torch.backends.cudnn.version()

        # Acquires the current device as the primary (test) device
        cls.primary_device = 'cuda:{0}'.format(torch.cuda.current_device())


# Adds available device-type-specific test base classes
from torch.testing._internal.common_device_type import device_type_test_bases

device_type_test_bases.clear()

device_type_test_bases.append(CPUTestBase)
if torch.cuda.is_available():
    device_type_test_bases.append(CUDATestBase)

# ------- End of DeviceTypeTestBase dtypes ----------------------------------------------------------------------
# Downstream usage:
# - introduce variants decorator
# - usage with DeviceTypeTestBase and all derived classes
from itertools import product


class VariantsDecoratorInstantiator(DefaultInstantiator):

    attr_dispatcher_map = dict(DefaultInstantiator.attr_dispatcher_map)
    attr_dispatcher_map["variants"] = "_handle_variants_decorator"

    @staticmethod
    def _handle_variants_decorator(cls, name, test, generic_cls=None):
        test_variants = test.variants
        var_names = list(test_variants.keys())
        base_name = "_".join([k + "_{}" for k in var_names])

        conf_names = [base_name.format(*v) for v in
                      product(*[range(len(test_variants[var_name])) for var_name in var_names])]
        conf_vals = product(*[test_variants[var_name] for var_name in var_names])
        conf_val_dicts = [{n: c for n, c in zip(var_names, conf)} for conf in conf_vals]

        all_configs = []
        for n, v in zip(conf_names, conf_val_dicts):
            all_configs.append((n, v))

        for test_name, test_kwargs_dict in all_configs:
            DefaultInstantiator._handle_default_test(
                cls,
                name + "_" + test_name,
                test,
                generic_cls=generic_cls,
                kwargs_dict=test_kwargs_dict
            )


DeviceTypeTestBase.instantiator_cls = VariantsDecoratorInstantiator
