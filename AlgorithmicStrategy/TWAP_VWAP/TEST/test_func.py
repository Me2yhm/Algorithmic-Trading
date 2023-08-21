import unittest
import torch as t
from torch import einsum
from einops import rearrange


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

    def test_rearrange(self):
        tensor = t.randn(3, 4, 5)
        rearranged_tensor = rearrange(tensor, "a b c -> b a c")
        permuted_tensor = tensor.permute(1, 0, 2)
        self.assertEqual(permuted_tensor.equal(rearranged_tensor), True)

    def test_rearrange_2(self):
        images = t.randn((32, 30, 40, 3))
        self.assertEqual(
            t.tensor(rearrange(images, "b h w c -> (b h) w c").shape).equal(
                t.tensor([960, 40, 3])
            ),
            True,
        )

    def test_rearrange_3(self):
        images = t.randn((32, 60, 50))
        self.assertEqual(
            t.tensor(rearrange(images, "b n (h d) -> b h n d", h=2).shape).equal(
                t.tensor([32, 2, 60, 25])
            ),
            True,
        )


if __name__ == "__main__":
    unittest.main()
