"""Kiểm tra tự viết cho các phần dễ sai (RUBRIC mục H). Chạy: python -m unittest test_code -v (trong code/)."""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

import inference as inf  # noqa: E402
import losses  # noqa: E402
import model as mdl  # noqa: E402
import train  # noqa: E402
from benchmark import bench  # noqa: E402


class TestLosses(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.z = torch.randn(64, 9) * 3
        self.y = torch.randint(0, 9, (64,))

    def test_focal_gamma0_equals_ce(self):
        ce = F.cross_entropy(self.z, self.y)
        fl = losses.FocalLoss(gamma=0.0)(self.z, self.y)
        self.assertLess(abs(ce.item() - fl.item()), 1e-6)

    def test_focal_downweights_easy(self):
        self.assertLess(losses.FocalLoss(2.0)(self.z, self.y).item(), F.cross_entropy(self.z, self.y).item())

    def test_label_smoothing_matches_torch(self):
        for eps in (0.0, 0.1, 0.3):
            ours = losses.LabelSmoothingCE(eps)(self.z, self.y)
            ref = F.cross_entropy(self.z, self.y, label_smoothing=eps)
            self.assertLess(abs(ours.item() - ref.item()), 1e-6)

    def test_class_weights(self):
        counts = [675, 637, 618, 613, 637, 605, 644, 609, 5463]
        w = losses.class_weights(counts)
        self.assertAlmostEqual(w.mean().item(), 1.0, places=5)
        self.assertLess(w[8].item(), w[0].item())
        wcb = losses.class_weights(counts, beta=0.999)
        self.assertAlmostEqual(wcb.sum().item(), 9.0, places=4)

    def test_cutmix_lambda_is_true_area(self):
        x = torch.zeros(8, 3, 32, 32)
        x[4:] = 1.0  # nửa batch toàn 1 để đo diện tích dán vào
        y = torch.arange(8)
        rng = np.random.default_rng(0)
        for _ in range(50):
            xm, (ya, yb, lam) = losses.mix_batch(x, y, 1.0, "cutmix", rng=rng)
            self.assertTrue(torch.equal(ya, y))
            for i in range(8):
                src = int(yb[i])
                if (src >= 4) != (i >= 4):  # ảnh nhận hộp từ ảnh có màu khác -> đo được diện tích
                    frac = (xm[i, 0] != x[i, 0]).float().mean().item()
                    self.assertAlmostEqual(frac, 1 - lam, places=6)

    def test_mixup_mixes_labels(self):
        x, y = torch.randn(16, 3, 8, 8), torch.randint(0, 9, (16,))
        xm, t = losses.mix_batch(x, y, 0.4, "mixup", rng=np.random.default_rng(1))
        z = torch.randn(16, 9)
        ref = t[2] * F.cross_entropy(z, t[0]) + (1 - t[2]) * F.cross_entropy(z, t[1])
        self.assertAlmostEqual(losses.mixed_loss(nn.CrossEntropyLoss(), z, t).item(), ref.item(), places=6)


class TestModel(unittest.TestCase):
    def test_param_groups_no_wd_on_norm_bias_and_head_lr(self):
        m = mdl.build_model("resnet18", pretrained=False, init="finetune")
        groups = mdl.param_groups(m, 1e-4, 1e-3, 0.05)
        head = {id(p) for p in m.get_classifier().parameters()}
        n = 0
        for g in groups:
            for p in g["params"]:
                n += 1
                if p.ndim <= 1:
                    self.assertEqual(g["weight_decay"], 0.0)
                self.assertEqual(g["lr"], 1e-3 if id(p) in head else 1e-4)
        self.assertEqual(n, len(list(m.parameters())))

    def test_frozen_keeps_bn_eval(self):
        m = mdl.build_model("resnet18", pretrained=False, init="frozen")
        trainable = [p for p in m.parameters() if p.requires_grad]
        self.assertEqual({id(p) for p in trainable}, {id(p) for p in m.get_classifier().parameters()})
        mdl.set_train_mode(m)
        self.assertFalse(m.bn1.training)
        self.assertTrue(m.get_classifier().training)
        before = m.bn1.running_mean.clone()
        m(torch.randn(4, 3, 64, 64))
        self.assertTrue(torch.equal(before, m.bn1.running_mean))

    def test_initial_loss_close_to_ln9(self):
        torch.manual_seed(0)
        m = mdl.build_model("resnet18", pretrained=False).eval()
        with torch.no_grad():
            loss = F.cross_entropy(m(torch.randn(32, 3, 64, 64)), torch.randint(0, 9, (32,)))
        self.assertLess(abs(loss.item() - np.log(9)), 0.5)

    def test_count_params(self):
        m = mdl.build_model("resnet18", pretrained=False)
        self.assertAlmostEqual(mdl.count_params(m), 11.18, delta=0.05)


class TestInference(unittest.TestCase):
    def _bn_model(self, name):
        torch.manual_seed(0)
        m = mdl.build_model(name, pretrained=False).eval()
        for mod in m.modules():  # thống kê BN khác mặc định để phép gộp có ý nghĩa
            if isinstance(mod, nn.BatchNorm2d):
                mod.running_mean.uniform_(-0.5, 0.5)
                mod.running_var.uniform_(0.5, 2.0)
                mod.weight.data.uniform_(0.5, 1.5)
                mod.bias.data.uniform_(-0.2, 0.2)
        return m

    def test_fuse_bn_resnet(self):
        m = self._bn_model("resnet18")
        x = torch.randn(2, 3, 64, 64)
        f = inf.fuse_conv_bn(m, check_input=x, verbose=False)
        self.assertEqual(sum(isinstance(t, nn.BatchNorm2d) for t in f.modules()), 0)
        self.assertGreater(f.n_fused_bn, 0)
        self.assertLess(f.fuse_max_abs_diff, 1e-4)

    def test_fuse_bn_efficientnet(self):
        m = self._bn_model("efficientnet_b0")
        x = torch.randn(2, 3, 64, 64)
        f = inf.fuse_conv_bn(m, check_input=x, verbose=False)
        self.assertEqual(sum(isinstance(t, nn.BatchNorm2d) for t in f.modules()), 0)
        self.assertLess(f.fuse_max_abs_diff, 1e-4)

    def test_temperature_recovers_T(self):
        rng = np.random.default_rng(0)
        z = rng.normal(size=(5000, 9)) * 3
        p = inf._softmax(z / 2.5)
        y = np.array([rng.choice(9, p=pi) for pi in p])
        T = inf.fit_temperature(z, y)
        self.assertAlmostEqual(T, 2.5, delta=0.15)
        np.testing.assert_array_equal(inf.apply_temperature(z, T).argmax(1), z.argmax(1))

    def test_aggregate_and_views(self):
        x = torch.arange(2 * 3 * 6 * 6, dtype=torch.float32).view(2, 3, 6, 6)
        self.assertTrue(torch.equal(inf.view_hflip(inf.view_hflip(x)), x))
        crops = inf.views_multicrop(x, 4, flip=True)
        self.assertEqual(len(crops), 10)
        self.assertTrue(all(c.shape[-1] == 4 for c in crops))
        l1, l2 = np.random.randn(5, 9), np.random.randn(5, 9)
        for space in ("prob", "logit"):
            p = inf.aggregate_views([l1, l2], space)
            np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-9)
        np.testing.assert_allclose(inf.aggregate_views([l1, l1], "logit"), inf._softmax(l1))


class TestTrain(unittest.TestCase):
    def test_scheduler_warmup_cosine(self):
        m = nn.Linear(2, 2)
        opt = torch.optim.SGD(m.parameters(), lr=1.0)
        cfg = train.Config(epochs=4, warmup_epochs=1)
        sch = train.build_scheduler(opt, cfg, 10)
        lrs = []
        for _ in range(40):
            lrs.append(opt.param_groups[0]["lr"])
            opt.step()
            sch.step()
        self.assertAlmostEqual(lrs[0], 0.1)
        self.assertAlmostEqual(max(lrs), 1.0)
        self.assertAlmostEqual(lrs[9], 1.0)  # hết warmup
        self.assertAlmostEqual(lrs[10], 1.0)  # bắt đầu cosine
        self.assertLess(lrs[-1], 0.01)

    def test_ema(self):
        m = nn.Linear(3, 1)
        ema = train.EMA(m, 0.5)
        with torch.no_grad():
            m.weight.add_(1.0)
        ema.update(m)  # d = min(0.5, 2/11)
        d = 2 / 11
        w0 = ema.module.weight
        self.assertTrue(torch.allclose(w0, d * (m.weight - 1) + (1 - d) * m.weight))

    def test_parse_overrides(self):
        o = train.parse_overrides(["seed=1", "loss=focal", "ema_decay=none", "amp=false", "lr_head=3e-4",
                                   "mix=cutmix", "sampler=None"])
        self.assertEqual(o, {"seed": 1, "loss": "focal", "ema_decay": None, "amp": False, "lr_head": 3e-4,
                             "mix": "cutmix", "sampler": None})
        with self.assertRaises(KeyError):
            train.parse_overrides(["foo=1"])

    def test_bench(self):
        r = bench(lambda: sum(range(1000)), warmup=2, iters=50)
        self.assertEqual(r["n"], 50)
        self.assertLessEqual(r["p50"], r["p95"])
        self.assertLessEqual(r["p95"], r["p99"])


if __name__ == "__main__":
    unittest.main()
