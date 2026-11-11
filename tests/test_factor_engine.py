"""Pure-data factor-engine and original-VM compatibility checks."""
import random
import unittest
from datetime import date

import numpy as np
import pandas as pd
import torch

from ashare.daily import DAILY_COLUMNS, LABEL_COLUMN
from ashare.tensor import build_daily_tensor
from ashare.formula import AShareFormulaEngine
from ashare.evaluate import cross_sectional_ic, search_baseline
from ashare.operators import _cs_rank
from model_core.vm import StackVM


def example(rows=30):
    dates = pd.bdate_range("2026-07-01", periods=rows)
    data = []
    for s, code in enumerate(["000001", "000002", "600000"]):
        for t, d in enumerate(dates):
            row = {c: .01 * (s+1) + t/100 for c in DAILY_COLUMNS}
            row.update(stock_code=code, trade_date=d,
                       main_force_created_at=d + pd.Timedelta(hours=16),
                       **{LABEL_COLUMN: .002 * (s+1) + t/1000})
            data.append(row)
    return pd.DataFrame(data)


class TestTensor(unittest.TestCase):
    def test_shape_asof_missing_and_label_separation(self):
        df = example()
        df.loc[(df.stock_code == "000002") & (df.trade_date == df.trade_date.min()), "ret_1d"] = np.nan
        df.loc[df.index[-1], "main_force_created_at"] = df.loc[df.index[-1], "trade_date"] + pd.Timedelta(days=2)
        df = df[~((df.stock_code == "000001") & (df.trade_date == df.trade_date.iloc[4]))]
        data = build_daily_tensor(df)
        self.assertEqual(data.features.shape, (3, 12, 30))
        self.assertEqual(data.target.shape, (3, 30))
        self.assertFalse(data.feature_valid[0, :, 4].any())
        self.assertFalse(data.feature_valid[1, 0, 0])
        self.assertFalse(data.feature_valid[2, DAILY_COLUMNS.index("holding"), -1])
        self.assertTrue(data.feature_valid[2, DAILY_COLUMNS.index("ret_1d"), -1])
        self.assertTrue(data.target_valid[2, -1])

    def test_future_target_change_does_not_modify_features(self):
        df = example()
        a = build_daily_tensor(df)
        df.loc[df.trade_date == df.trade_date.max(), LABEL_COLUMN] = 1000
        b = build_daily_tensor(df)
        self.assertTrue(torch.equal(a.features, b.features))
        self.assertTrue(torch.equal(a.feature_valid, b.feature_valid))

    def test_duplicate_rejected(self):
        df = example()
        with self.assertRaises(ValueError):
            build_daily_tensor(pd.concat([df, df.iloc[[0]]]))


class TestFormula(unittest.TestCase):
    def test_random_rpn_always_valid(self):
        engine = AShareFormulaEngine()
        rng = random.Random(1)
        for _ in range(1000):
            seq = engine.sample(rng)
            self.assertTrue(engine.validate(seq))
            self.assertLessEqual(len(seq), 12)
        for expression in ("ADD ret_1d", "ret_1d ret_5d", "UNKNOWN"):
            with self.assertRaises(ValueError):
                engine.parse(expression)
        with self.assertRaises(ValueError):
            engine.validate((-1,))

    def test_masked_arithmetic_and_division(self):
        data = build_daily_tensor(example())
        engine = AShareFormulaEngine()
        x, m = engine.execute(engine.parse("ret_1d holding ADD"), data)
        self.assertTrue(m.all())
        self.assertAlmostEqual(float(x[0, 0]),
                               float(data.features[0, 0, 0] + data.features[0, 8, 0]))
        zeros, good = engine.execute(engine.parse("ret_1d ret_1d SUB ret_1d DIV"), data)
        self.assertTrue(good.all())
        self.assertTrue(torch.allclose(zeros, torch.zeros_like(zeros)))
        bad, mask = engine.execute(engine.parse("ret_1d ret_1d ret_1d SUB DIV"), data)
        self.assertFalse(mask.any())
        self.assertTrue(torch.isnan(bad).all())

    def test_rolling_missing_is_not_zero_filled(self):
        df = example()
        df.loc[(df.stock_code == "000001") &
               (df.trade_date == df.trade_date.iloc[3]), "ret_1d"] = np.nan
        data = build_daily_tensor(df)
        engine = AShareFormulaEngine()
        values, mask = engine.execute(engine.parse("ret_1d TS_MEAN5"), data)
        self.assertFalse(mask[0, 4])
        self.assertFalse(mask[0, 7])
        self.assertTrue(mask[0, 8])
        self.assertTrue(torch.isnan(values[0, 4]))
        _, mask = engine.execute(engine.parse("ret_1d DELAY1"), data)
        self.assertFalse(mask[0, 4])
        self.assertFalse(mask[0, 0])

    def test_cross_sectional_ranks_ties_and_missing(self):
        x = torch.tensor([[1., 7.], [1., 8.], [3., 9.]])
        m = torch.tensor([[True, True], [True, False], [True, False]])
        ranks, valid = _cs_rank((x, m))
        self.assertAlmostEqual(float(ranks[0, 0]), .25)
        self.assertAlmostEqual(float(ranks[1, 0]), .25)
        self.assertAlmostEqual(float(ranks[2, 0]), 1.0)
        self.assertFalse(valid[:, 1].any())

    def test_legacy_stack_vm_still_works(self):
        vm = StackVM()
        tensor = torch.ones((2, 6, 8))
        self.assertTrue(torch.equal(vm.execute([0], tensor), torch.ones((2, 8))))
        self.assertIsNone(vm.execute([0, 1], tensor))


class TestEvaluation(unittest.TestCase):
    def test_ic_and_baseline_split(self):
        data = build_daily_tensor(example(rows=40))
        engine = AShareFormulaEngine()
        values, mask = engine.execute(engine.parse("ret_1d"), data)
        ic = cross_sectional_ic(values, mask, data, min_stocks=3)
        self.assertEqual(ic.dates, 40)
        self.assertAlmostEqual(ic.mean_ic, 1.0)
        self.assertEqual(ic.observations, 120)
        result = search_baseline(
            data, start=date(2026, 7, 1), end=date(2026, 8, 25),
            samples=25, seed=4, min_stocks=3, min_train_days=5)
        self.assertIsNotNone(result)
        self.assertGreater(result["train"].dates, 5)
        self.assertGreater(result["test"].dates, 0)


if __name__ == "__main__":
    unittest.main()
