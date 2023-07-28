import unittest
import torch as t
from torch import einsum


class MyTestCase(unittest.TestCase):
    def test_einsum(self):
        """
        爱因斯坦求和示例
        """
        x = t.randn(3, 4, 5, 6)
        y = t.randn(3, 4, 8, 6)
        b_num, h_num, i_num, d_num = x.shape
        _, _, j_num, _ = y.shape
        test = t.zeros([b_num, h_num, i_num, j_num])
        for b in range(b_num):
            for h in range(h_num):
                for i in range(i_num):
                    for j in range(j_num):
                        tmp = 0
                        for d in range(d_num):
                            tmp += x[b, h, i, d] * y[b, h, j, d]
                        test[b, h, i, j] = tmp
        real = einsum("b h i d, b h j d -> b h i j", x, y)
        self.assertEqual(real.equal(test), True)


if __name__ == '__main__':
    unittest.main()
