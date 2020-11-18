
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing._internal.common_device_type import dtypesIfCUDA, instantiate_device_type_tests
from torch.testing._internal.common_device_type import device_type_test_bases
from torch.testing import floating_types, floating_types_and_half, integral_types

from common_data_type import vision_dtypes, variants
# from common_data_type import instantiate_tests


class Tester(TestCase):

    @variants("interpolation", [0, 2, 3])
    @variants("size", [32, 26, (32, 26), [26, 32]])
    @vision_dtypes(*(floating_types() + integral_types() + ("PIL.RGB", "PIL.P", "PIL.F")))
    @dtypesIfCUDA(*(floating_types_and_half() + integral_types()))
    def test_resize(self, device, dtype, interpolation, size):
        # img = self.create_image(h=12, w=16, device=device, dtype=dtype)
        pass


instantiate_device_type_tests(Tester, globals())


if __name__ == '__main__':
    run_tests()
