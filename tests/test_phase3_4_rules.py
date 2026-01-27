import sys, json
sys.path.insert(0, "./src")

from strategy_creator.phase3_4 import build_phase3_4


def test_jedi_long_only_no_short_and_risk_clamped():
    payload = dict(
        instrument_type="etf",
        is_thematic=True,
        constraints={"long_only": True},
        liquidity="media",
        volatility="alta",
        directionality="trend",
        structure="impulsiva",
        news_reaction="moderata",
        natural_horizon="swing",
    )

    out = build_phase3_4(payload)

    assert out["primary_archetype"] == "trend_following"
    assert out["secondary_archetype"] in (None, "volatility_breakout")
    assert "strategies" in out and len(out["strategies"]) == 2

    txt = json.dumps(out, ensure_ascii=False).lower()
    assert "short" not in txt

    for s in out["strategies"]:
        assert s["operation"] in ("long", "long-only")
        assert 0.0 < float(s["risk_per_trade_pct"]) <= 0.5


def test_range_oscillante_prefers_mean_reversion():
    payload = dict(
        instrument_type="equity",
        is_thematic=False,
        constraints={"long_only": False},
        liquidity="alta",
        volatility="media",
        directionality="range",
        structure="oscillante",
        news_reaction="moderata",
        natural_horizon="swing",
    )
    out = build_phase3_4(payload)
    assert out["primary_archetype"] == "mean_reversion"
    assert out["secondary_archetype"] == "support_resistance"


def test_intraday_high_vol_prefers_breakout_and_secondary_none():
    payload = dict(
        instrument_type="crypto",
        is_thematic=False,
        constraints={"long_only": False},
        liquidity="alta",
        volatility="alta",
        directionality="mista",
        structure="impulsiva",
        news_reaction="forte",
        natural_horizon="intraday",
    )
    out = build_phase3_4(payload)
    assert out["primary_archetype"] == "volatility_breakout"
    assert out["secondary_archetype"] is None
