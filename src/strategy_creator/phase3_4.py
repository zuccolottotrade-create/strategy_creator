from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class Phase2Input:
    instrument_type: str
    is_thematic: bool
    constraints: Dict[str, bool]
    liquidity: str
    volatility: str
    directionality: str
    structure: str
    news_reaction: str
    natural_horizon: str


@dataclass(frozen=True)
class StrategySpec:
    name: str
    archetype: str
    operation: str
    timeframe: str
    setup: str
    entry_trigger: str
    stop_loss: str
    exit_management: str
    risk_per_trade_pct: float


def _first_present(payload: Dict[str, Any], keys: List[str]) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _adapt_phase2_input(payload: Dict[str, Any]) -> Phase2Input:
    instrument_type = _first_present(payload, ["instrument_type", "instrument", "instrumentType"])
    is_thematic = _first_present(payload, ["is_thematic", "isThematic"])
    constraints = _first_present(payload, ["constraints", "constraint", "constraints_dict", "constraintsDict"])
    liquidity = _first_present(payload, ["liquidity", "liquidita"])
    volatility = _first_present(payload, ["volatility", "volatilita"])
    directionality = _first_present(payload, ["directionality", "regime"])
    structure = _first_present(payload, ["structure", "struttura"])
    news_reaction = _first_present(payload, ["news_reaction", "newsReaction"])
    natural_horizon = _first_present(payload, ["natural_horizon", "time_horizon", "timeHorizon"])

    missing = [
        name
        for name, value in [
            ("instrument_type", instrument_type),
            ("is_thematic", is_thematic),
            ("constraints", constraints),
            ("liquidity", liquidity),
            ("volatility", volatility),
            ("directionality", directionality),
            ("structure", structure),
            ("news_reaction", news_reaction),
            ("natural_horizon", natural_horizon),
        ]
        if value is None
    ]
    if missing:
        raise ValueError(f"Missing phase2 fields: {', '.join(missing)}")

    constraints_dict = dict(constraints)
    if "long_only" not in constraints_dict:
        constraints_dict["long_only"] = bool(constraints_dict.get("longOnly", False))

    # normalizza stringhe attese (difensivo)
    dir_norm = str(directionality).lower()
    if dir_norm not in {"trend", "range", "mista"}:
        dir_norm = str(directionality)

    vol_norm = str(volatility).lower()
    if vol_norm not in {"alta", "media", "bassa"}:
        vol_norm = str(volatility)

    liq_norm = str(liquidity).lower()
    if liq_norm not in {"alta", "media", "bassa"}:
        liq_norm = str(liquidity)

    struct_norm = str(structure).lower()
    if struct_norm not in {"impulsiva", "oscillante"}:
        struct_norm = str(structure)

    news_norm = str(news_reaction).lower()
    if news_norm not in {"forte", "moderata", "debole"}:
        news_norm = str(news_reaction)

    horizon_norm = str(natural_horizon).lower()
    if horizon_norm not in {"intraday", "swing", "position"}:
        horizon_norm = str(natural_horizon)

    return Phase2Input(
        instrument_type=str(instrument_type),
        is_thematic=bool(is_thematic),
        constraints=constraints_dict,
        liquidity=liq_norm,
        volatility=vol_norm,
        directionality=dir_norm,
        structure=struct_norm,
        news_reaction=news_norm,
        natural_horizon=horizon_norm,
    )


def _score_archetypes(data: Phase2Input) -> Dict[str, int]:
    scores = {
        "trend_following": 0,
        "volatility_breakout": 0,
        "mean_reversion": 0,
        "support_resistance": 0,
        "event_news": 0,
    }

    # directionality
    if data.directionality == "trend":
        scores["trend_following"] += 4
        scores["volatility_breakout"] += 2
        scores["mean_reversion"] -= 2
    elif data.directionality == "range":
        scores["mean_reversion"] += 4
        scores["support_resistance"] += 2
        scores["trend_following"] -= 2
    else:  # mista
        scores["trend_following"] += 1
        scores["mean_reversion"] += 1
        scores["support_resistance"] += 1

    # structure
    if data.structure == "impulsiva":
        scores["trend_following"] += 2
        scores["volatility_breakout"] += 3
        scores["mean_reversion"] -= 1
    else:  # oscillante
        scores["mean_reversion"] += 2
        scores["support_resistance"] += 2
        scores["volatility_breakout"] -= 1

    # volatility
    if data.volatility == "alta":
        scores["volatility_breakout"] += 3
        scores["trend_following"] += 1
    elif data.volatility == "bassa":
        scores["mean_reversion"] += 2
        scores["support_resistance"] += 2
        scores["volatility_breakout"] -= 2
    else:  # media
        scores["trend_following"] += 1
        scores["support_resistance"] += 1

    # news reaction
    if data.news_reaction == "forte":
        scores["event_news"] += 3
        scores["volatility_breakout"] += 1
    elif data.news_reaction == "debole":
        scores["event_news"] -= 3

    # horizon
    if data.natural_horizon == "intraday":
        scores["volatility_breakout"] += 2
        scores["mean_reversion"] += 1
        scores["trend_following"] -= 1
    elif data.natural_horizon == "position":
        scores["trend_following"] += 2
        scores["mean_reversion"] -= 1
    else:  # swing
        scores["trend_following"] += 1
        scores["support_resistance"] += 1

    # liquidity penalties
    if data.liquidity == "bassa":
        scores["volatility_breakout"] -= 2
        scores["event_news"] -= 2

    # ETF diversificati tendono a penalizzare news-driven; su tematici meno penalità
    if data.instrument_type == "etf" and data.news_reaction != "forte" and not data.is_thematic:
        scores["event_news"] -= 2

    return scores


def _reason_for_exclusion(archetype: str, data: Phase2Input) -> str:
    if archetype == "trend_following" and data.directionality == "range":
        return "Prezzo in range: persistenza direzionale insufficiente per trend following."
    if archetype == "mean_reversion" and data.directionality == "trend":
        return "Mercato direzionale: mean reversion tende a essere contro-trend e meno robusta."
    if archetype == "volatility_breakout" and data.volatility == "bassa":
        return "Volatilità bassa: breakout meno frequenti e spesso falsi."
    if archetype == "support_resistance" and data.structure == "impulsiva":
        return "Struttura impulsiva: livelli statici meno affidabili, meglio regole su swing/struttura."
    if archetype == "event_news" and data.news_reaction in {"debole", "moderata"} and data.instrument_type == "etf":
        return "ETF: impatto news spesso diluito; news-driven non è archetipo primario robusto."
    if data.liquidity == "bassa" and archetype in {"event_news", "volatility_breakout"}:
        return "Liquidità bassa: slippage elevato in fasi veloci, peggiora l'edge."
    return "Archetipo secondario rispetto alle caratteristiche operative dominanti."


def _risk_pct(data: Phase2Input) -> float:
    """
    Rischio per trade (in %) coerente con strumento/volatilità.
    Target: vol alta (tematici) 0.25–0.5; media 0.5–0.8; bassa 0.8–1.0.
    """
    if data.volatility == "alta":
        base = 0.4
        if data.liquidity == "bassa":
            base = 0.25
        return min(base, 0.5)
    if data.volatility == "bassa":
        return 0.9
    return 0.7


def _timeframe(data: Phase2Input) -> str:
    return {
        "intraday": "M5-M30",
        "swing": "H4-D1",
        "position": "D1-W1",
    }.get(data.natural_horizon, "H4-D1")


def _operation(data: Phase2Input) -> str:
    long_only = bool(data.constraints.get("long_only"))
    return "long-only" if long_only else "long"


def _strategy_for_archetype(archetype: str, data: Phase2Input, label: str) -> StrategySpec:
    operation = _operation(data)
    timeframe = _timeframe(data)
    risk = _risk_pct(data)

    # Nota: i testi sono esplicitamente LONG-only/long: nessun riferimento a short.
    if archetype == "trend_following":
        return StrategySpec(
            name=f"{label} - Trend pullback (struttura)",
            archetype=archetype,
            operation=operation,
            timeframe=timeframe,
            setup=(
                "Trend attivo: massimi/minimi crescenti su H4/D1 e ultimo impulso con break of structure "
                "(chiusura sopra l'ultimo swing high)."
            ),
            entry_trigger=(
                "Dopo un impulso (swing low -> swing high), attendi un pullback nella zona 50–61.8% "
                "dell'impulso e entra LONG alla chiusura H4 sopra il massimo della barra precedente "
                "all'interno della zona, con higher-low (minimo corrente > minimo precedente)."
            ),
            stop_loss=(
                "Stop sotto il minimo del pullback (swing low del ritracciamento) con margine conservativo "
                "(es. 0.3% sotto il minimo) per evitare rumore."
            ),
            exit_management=(
                "TP1 sul precedente swing high; poi trailing stop sotto l'ultimo higher-low su H4 "
                "(aggiorna solo a nuovi swing). Uscita completa se si forma un lower-low su H4 (struttura rotta)."
            ),
            risk_per_trade_pct=risk,
        )

    if archetype == "volatility_breakout":
        return StrategySpec(
            name=f"{label} - Breakout di range (conferma)",
            archetype=archetype,
            operation=operation,
            timeframe=timeframe,
            setup=(
                "Identifica un range di consolidamento: massimi/minimi degli ultimi 20 barre H4 "
                "(range stretto e ben definito)."
            ),
            entry_trigger=(
                "Entra LONG alla chiusura H4 sopra il massimo del range (range_high) e conferma con una "
                "seconda chiusura H4 consecutiva ancora sopra range_high (double-close)."
            ),
            stop_loss=(
                "Stop all'interno del range: sotto range_high di metà ampiezza del range "
                "(stop = range_high - 0.5*(range_high-range_low))."
            ),
            exit_management=(
                "Target 1 = range_high + (range_high-range_low). Dopo TP1: stop a pareggio e trailing "
                "sotto higher-lows su H4 finché la struttura resta rialzista."
            ),
            risk_per_trade_pct=risk,
        )

    if archetype == "mean_reversion":
        return StrategySpec(
            name=f"{label} - Mean reversion su estremi (solo long)",
            archetype=archetype,
            operation=operation,
            timeframe=timeframe,
            setup=(
                "Mercato laterale: range definito da massimi/minimi recenti e struttura oscillante."
            ),
            entry_trigger=(
                "Entra LONG dopo falsa rottura verso il basso e rientro nel range: "
                "chiusura H4 di rientro sopra il minimo precedente del range e successiva chiusura H4 positiva."
            ),
            stop_loss="Stop sotto il minimo della falsa rottura (minimo di capitolazione intrarange).",
            exit_management=(
                "TP al centro range; lascia una parte verso la parte alta del range con trailing sotto higher-lows."
            ),
            risk_per_trade_pct=risk,
        )

    if archetype == "support_resistance":
        return StrategySpec(
            name=f"{label} - Reazione su supporto (struttura)",
            archetype=archetype,
            operation=operation,
            timeframe=timeframe,
            setup=(
                "Individua un supporto testato almeno 2 volte e coerente con struttura oscillante."
            ),
            entry_trigger=(
                "Entra LONG alla terza reazione: chiusura H4 sopra il massimo della barra di test del supporto "
                "con minimo crescente rispetto alla barra precedente."
            ),
            stop_loss="Stop sotto il minimo del test sul supporto con margine conservativo (es. 0.3%).",
            exit_management=(
                "Uscita parziale al primo swing opposto; trailing sotto higher-lows fino al prossimo livello."
            ),
            risk_per_trade_pct=risk,
        )

    # event_news
    return StrategySpec(
        name=f"{label} - Evento e follow-through (solo long)",
        archetype=archetype,
        operation=operation,
        timeframe=timeframe,
        setup="Definisci in anticipo livelli pre-evento e attendi conferma strutturale post-evento.",
        entry_trigger=(
            "Entra LONG solo dopo conferma post-evento: prima chiusura H4 sopra il massimo pre-evento "
            "e successivo pullback che tiene sopra quel livello (retest riuscito)."
        ),
        stop_loss="Stop sotto il minimo del pullback di retest post-evento.",
        exit_management="TP1 a 1R, poi trailing sotto higher-lows finché la volatilità resta elevata e la struttura regge.",
        risk_per_trade_pct=risk,
    )


def build_phase3_4(payload: Dict[str, Any] | Phase2Input) -> Dict[str, Any]:
    data = payload if isinstance(payload, Phase2Input) else _adapt_phase2_input(payload)

    scores = _score_archetypes(data)
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    primary = sorted_scores[0][0]

    # secondario solo se:
    # - abbastanza vicino al primario
    # - e non è un negativo/zero (evita secondari "a caso")
    secondary = None
    if len(sorted_scores) > 1:
        cand, cand_score = sorted_scores[1]
        prim_score = sorted_scores[0][1]
        if cand_score >= max(2, prim_score - 1):
            secondary = cand

    excluded = []
    for archetype, _score in sorted_scores:
        if archetype in {primary, secondary}:
            continue
        excluded.append({"archetype": archetype, "reason": _reason_for_exclusion(archetype, data)})

    strategies = [
        _strategy_for_archetype(primary, data, "Strategia 1"),
        _strategy_for_archetype(secondary or primary, data, "Strategia 2"),
    ]

    return {
        "primary_archetype": primary,
        "secondary_archetype": secondary,
        "excluded_archetypes": excluded,
        "strategies": [asdict(strategy) for strategy in strategies],
    }
