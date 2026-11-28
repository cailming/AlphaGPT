import unittest
from datetime import date, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from ashare.p4_action_pipeline import atomic_json
from ashare.p58_fund_snapshot import CHINA
from ashare.p66_fund_eval import label_path, metric, summarize
from ashare.p66_fund_research import FUNDS, cutoff, factor_table, latest_snapshot, score_summary, snapshot_visible

DAY = date(2026, 11, 27)
LIMIT = cutoff(DAY)


def record():
    return {"signal_date":str(DAY),"formal_stocks_at_observation":300,"read_only_repeatable_read":True,
            "phase":"next-open","availability":"EVENING_OBSERVATION_FOR_FOLLOWING_SESSION",
            "data_read_started_at":"2026-11-27T22:10:01+08:00","snapshot_at":"2026-11-27T22:10:02+08:00",
            "files_ready_at":"2026-11-27T22:10:03+08:00"}


class FundTests(unittest.TestCase):
    def test_next_open_boundary_and_weekend(self):
        self.assertEqual(str(LIMIT.date()), "2026-11-30")
        now = datetime(2026,11,28,7,tzinfo=CHINA)
        item = record()
        self.assertTrue(snapshot_visible(item,DAY,LIMIT,now))
        item["files_ready_at"] = "2026-11-30T09:16:00+08:00"
        self.assertFalse(snapshot_visible(item,DAY,LIMIT,LIMIT))

    def test_rejected_mtime_after_declared_availability(self):
        import os
        from ashare.p58_fund_snapshot import SOURCES
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/str(DAY)/"next-open/r00001"
            path.mkdir(parents=True)
            file = path/"manifest.json"
            atomic_json(file, record())
            for source in SOURCES:
                (path/(source+".parquet")).touch()
            os.utime(file, (LIMIT.timestamp()+5, LIMIT.timestamp()+5))
            self.assertIsNone(latest_snapshot(Path(tmp), DAY, LIMIT))

    def test_factor_directions_delta_and_signal_status(self):
        frame = pd.DataFrame({"stock_code":["11","12"],"main_force_score":[30,10],"activity_score":[2,1],
            "holding_score":[2,1],"distribution_risk":[.6,.2],"vol_ratio_5_20":[.5,.5],"source_version":["v1","v1"]})
        prior = frame.assign(main_force_score=[20,9],source_version=["v1","other"])
        signals = pd.DataFrame({"stock_code":["11","12"],"signal_code":["OPEN30_STRONG_PULLBACK","NECKLINE_BREAKOUT"],
                                "signal_status":["CONFIRMED","INVALID"]})
        universe = pd.DataFrame({"ma20":[1,0]},index=["11","12"])
        result = factor_table(FUNDS[0], universe, frame, prior, signals)
        self.assertEqual(result.delta.iloc[0],10)
        self.assertTrue(np.isnan(result.delta.iloc[1]))
        self.assertEqual(result.low_distribution_risk.tolist(), [-.6,-.2])
        self.assertEqual(result.breakout_pullback.tolist(), [.9,.4])

    def test_v2_categorical_state_never_becomes_ownership_evidence(self):
        frame = pd.DataFrame({"stock_code":["11"],"selection_score":[30],"activity_score":[2],"pullback_score":[2],
            "overheat_risk":[3],"position_state":["ABOVE_MA20"],"source_version":["v"],"algorithm_version":["a"]})
        result = factor_table(FUNDS[1],pd.DataFrame({"ma20":[0]},index=["11"]),frame,None,pd.DataFrame())
        stats = score_summary(result,FUNDS[1])
        self.assertTrue(stats["holding"]["unavailable_field"])
        self.assertTrue(stats["low_distribution_risk"]["unavailable_field"])
        self.assertEqual(stats["holding"]["scored"],0)
        self.assertTrue(np.isnan(result.breakout_pullback.iloc[0]))

    def test_planned_gap_never_shifts_exit(self):
        observed = ["2026-11-27","2026-11-30","2026-12-02","2026-12-03","2026-12-04","2026-12-07","2026-12-08"]
        state, days = label_path(DAY,"2026-11-30",observed,date(2026,12,8))
        self.assertEqual(state,"MISSING_PLANNED_FORMAL_SESSION")
        self.assertEqual(days[-1],"2026-12-07")
        self.assertEqual(label_path(DAY,"2026-11-30",observed,DAY)[0],"LABEL_PENDING")

    def test_topk_chosen_before_labels_and_paired_baseline(self):
        frame = pd.DataFrame({"stock_code":[str(i).zfill(3) for i in range(25)], "factor":range(25),"ma20":range(25),
                              "economic_return":np.arange(25)/100})
        result = metric(frame,"factor")
        self.assertAlmostEqual(result["rank_ic"],1)
        self.assertAlmostEqual(result["top"]["10"]["net_excess_by_per_side_bps"]["10"],.195-.12-.002)
        frame.loc[24,"economic_return"] = np.nan
        frame.loc[0,"ma20"] = np.nan
        compared = summarize(frame,"factor")
        self.assertIsNone(compared["factor"]["top"]["10"]["mean"])
        self.assertEqual(compared["paired_factor"]["eligible"],24)
        self.assertEqual(compared["paired_ma20"]["eligible"],24)


if __name__ == "__main__":
    unittest.main()
