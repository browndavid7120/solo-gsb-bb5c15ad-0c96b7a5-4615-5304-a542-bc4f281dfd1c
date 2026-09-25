"""End-to-end tests for the read-only reversal-plan endpoint.

POST /datasets/{dataset_id}/reversal-plans
"""
from __future__ import annotations

import itertools
import os
import tempfile
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio

# Point the ledger at an isolated database before importing the app.
_tmp = tempfile.mkdtemp(prefix="dpledger-test-reversal-plan-")
os.environ["DATABASE_PATH"] = os.path.join(_tmp, "reversal-plan.db")

from httpx import ASGITransport, AsyncClient  # noqa: E402

from app import db, rdp  # noqa: E402
from app.main import app  # noqa: E402

_counter = itertools.count()
_rev_counter = itertools.count()


def _new_dataset_id(name: str) -> str:
    return f"dsrp-{name}-{next(_counter)}"


def _new_reversal_id() -> str:
    return f"rev-rp-{next(_rev_counter)}"


@pytest_asyncio.fixture()
async def client():
    await db.init_db()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


async def _register(client, dataset_id: str, eps: float = 1.0, delta: float = 1e-6):
    return await client.post(
        f"/datasets/{dataset_id}",
        json={"epsilon_cap": eps, "delta_cap": delta,
              "description": "test dataset"},
    )


async def _grant(client, ds: str, request_id: str, epsilon: float,
                 delta: float = 1e-8, query_type: str = "q"):
    return await client.post(
        f"/datasets/{ds}/requests",
        json={"request_id": request_id, "query_type": query_type,
              "epsilon": epsilon, "delta": delta},
    )


async def _plan(client, ds: str, pending: dict, request_ids: list[str]):
    return await client.post(
        f"/datasets/{ds}/reversal-plans",
        json={"pending_item": pending, "request_ids": request_ids},
    )


async def _backdate(dataset_id: str, reservation_id: str,
                    expires_at: datetime) -> None:
    async with db.connect() as conn:
        await db.backdate_reservation_expiry(
            conn,
            dataset_id=dataset_id,
            reservation_id=reservation_id,
            expires_at=expires_at.isoformat(),
        )
        await conn.commit()


def _independent_plan(
    baseline: list[tuple[float, float]],
    candidates: list[tuple[str, float, float]],
    pending: tuple[float, float],
    cap: float,
    target_delta: float,
    tol: float = 1e-12,
):
    """Brute-force the expected suggestion directly with rdp.evaluate_entries.

    Returns ``(chosen_ids, epsilon_before, after, with_pending)`` or ``None``
    when no subset is feasible.
    """
    n = len(candidates)
    epsilon_before = max(
        0.0,
        rdp.evaluate_entries(
            baseline + [(e, d) for _, e, d in candidates], target_delta
        )[1],
    )
    best = None
    for mask in range(1 << n):
        remaining = [
            (e, d) for i, (_, e, d) in enumerate(candidates)
            if not mask & (1 << i)
        ]
        with_pending = max(
            0.0,
            rdp.evaluate_entries(
                baseline + remaining + [pending], target_delta
            )[1],
        )
        if with_pending > cap + tol:
            continue
        after = max(
            0.0,
            rdp.evaluate_entries(baseline + remaining, target_delta)[1],
        )
        key_ids = sorted(
            cid for i, (cid, _, _) in enumerate(candidates) if mask & (1 << i)
        )
        count = mask.bit_count()
        if (
            best is None
            or count < best[1]
            or (count == best[1] and after > best[2] + tol)
            or (
                count == best[1]
                and abs(after - best[2]) <= tol
                and key_ids < best[0]
            )
        ):
            best = (key_ids, count, after, with_pending)
    if best is None:
        return None
    return best[0], epsilon_before, best[2], best[3]


# ----------------------------------------------------------------- behaviour --


@pytest.mark.asyncio
async def test_pending_fits_without_any_reversal(client):
    ds = _new_dataset_id("fits")
    await _register(client, ds, eps=1.0, delta=1e-6)
    await _grant(client, ds, "a", 0.1, 1e-8)
    await _grant(client, ds, "b", 0.2, 1e-8)

    r = await _plan(client, ds, {"epsilon": 0.1, "delta": 1e-8}, ["a", "b"])
    assert r.status_code == 200
    body = r.json()
    assert body["feasible"] is True
    assert body["reason"] is None
    assert body["reversal_ids"] == []
    assert body["reversal_count"] == 0
    assert body["suggestion_id"].startswith("rplan-")
    assert body["candidate_count"] == 2
    assert body["candidate_request_ids"] == ["a", "b"]
    assert body["pending_item"] == {"epsilon": 0.1, "delta": 1e-8}

    expected_before = max(
        0.0, rdp.evaluate_entries([(0.1, 1e-8), (0.2, 1e-8)], 1e-6)[1]
    )
    expected_with_pending = max(
        0.0,
        rdp.evaluate_entries(
            [(0.1, 1e-8), (0.2, 1e-8), (0.1, 1e-8)], 1e-6
        )[1],
    )
    assert body["epsilon_before"] == pytest.approx(expected_before, abs=1e-12)
    assert body["epsilon_after_reversal"] == pytest.approx(
        expected_before, abs=1e-12
    )
    assert body["epsilon_with_pending"] == pytest.approx(
        expected_with_pending, abs=1e-12
    )
    assert body["epsilon_with_pending"] <= 1.0 + 1e-12


@pytest.mark.asyncio
async def test_tiebreak_picks_largest_post_reversal_occupancy(client):
    ds = _new_dataset_id("maxocc")
    await _register(client, ds, eps=0.62, delta=1e-6)
    await _grant(client, ds, "big", 0.4, 1e-8)
    await _grant(client, ds, "small", 0.1, 1e-8)

    # The pending item fits after reversing either single request; the plan
    # must reverse the *smaller* one, keeping the ledger occupancy largest.
    body = (
        await _plan(client, ds, {"epsilon": 0.2, "delta": 1e-8},
                    ["big", "small"])
    ).json()
    assert body["feasible"] is True
    assert body["reversal_ids"] == ["small"]
    assert body["reversal_count"] == 1

    expected_after = max(
        0.0, rdp.evaluate_entries([(0.4, 1e-8)], 1e-6)[1]
    )
    expected_with_pending = max(
        0.0, rdp.evaluate_entries([(0.4, 1e-8), (0.2, 1e-8)], 1e-6)[1]
    )
    assert body["epsilon_after_reversal"] == pytest.approx(
        expected_after, abs=1e-12
    )
    assert body["epsilon_with_pending"] == pytest.approx(
        expected_with_pending, abs=1e-12
    )
    # Reversing "big" instead would also fit, but leaves less occupied.
    assert body["epsilon_after_reversal"] > max(
        0.0, rdp.evaluate_entries([(0.1, 1e-8)], 1e-6)[1]
    )


@pytest.mark.asyncio
async def test_tiebreak_falls_back_to_lexicographic_ids(client):
    ds = _new_dataset_id("lex")
    await _register(client, ds, eps=0.65, delta=1e-6)
    # Identical (epsilon, delta): both single reversals are feasible with
    # exactly the same post-reversal occupancy.
    await _grant(client, ds, "req-b", 0.3, 1e-8)
    await _grant(client, ds, "req-a", 0.3, 1e-8)

    body = (
        await _plan(client, ds, {"epsilon": 0.2, "delta": 1e-8},
                    ["req-b", "req-a"])
    ).json()
    assert body["feasible"] is True
    assert body["reversal_ids"] == ["req-a"]
    assert body["reversal_count"] == 1


@pytest.mark.asyncio
async def test_fewest_reversals_wins(client):
    ds = _new_dataset_id("fewest")
    await _register(client, ds, eps=1.0, delta=1e-6)
    for rid in ("req-a", "req-b", "req-c"):
        await _grant(client, ds, rid, 0.3, 1e-8)

    # No single reversal fits the pending item; any pair does.
    body = (
        await _plan(client, ds, {"epsilon": 0.5, "delta": 1e-8},
                    ["req-c", "req-a", "req-b"])
    ).json()
    assert body["feasible"] is True
    assert body["reversal_count"] == 2
    # All pairs score identically, so the lexicographically smallest wins.
    assert body["reversal_ids"] == ["req-a", "req-b"]
    expected_after = max(
        0.0, rdp.evaluate_entries([(0.3, 1e-8)], 1e-6)[1]
    )
    expected_with_pending = max(
        0.0, rdp.evaluate_entries([(0.3, 1e-8), (0.5, 1e-8)], 1e-6)[1]
    )
    assert body["epsilon_after_reversal"] == pytest.approx(
        expected_after, abs=1e-12
    )
    assert body["epsilon_with_pending"] == pytest.approx(
        expected_with_pending, abs=1e-12
    )


@pytest.mark.asyncio
async def test_infeasible_when_no_subset_fits(client):
    ds = _new_dataset_id("infeasible")
    await _register(client, ds, eps=0.5, delta=1e-6)
    await _grant(client, ds, "a", 0.2, 1e-8)

    # The pending item alone already exceeds the cap: even reversing every
    # candidate cannot fit it.
    r = await _plan(client, ds, {"epsilon": 0.6, "delta": 1e-8}, ["a"])
    assert r.status_code == 200
    body = r.json()
    assert body["feasible"] is False
    assert body["suggestion_id"] is None
    assert body["reason"] is not None
    assert body["reversal_ids"] == []
    assert body["reversal_count"] == 0
    assert body["epsilon_after_reversal"] is None
    assert body["epsilon_with_pending"] is None
    assert body["rdp_best_order_after_reversal"] is None
    assert body["rdp_best_order_with_pending"] is None
    expected_before = max(0.0, rdp.evaluate_entries([(0.2, 1e-8)], 1e-6)[1])
    assert body["epsilon_before"] == pytest.approx(expected_before, abs=1e-12)


@pytest.mark.asyncio
async def test_plan_matches_independent_brute_force(client):
    ds = _new_dataset_id("brute")
    target_delta = 1e-6
    cap = 1.5
    await _register(client, ds, eps=cap, delta=target_delta)
    # Baseline: another granted request plus an active reservation.
    await _grant(client, ds, "other", 0.08, 1e-9)
    await client.post(
        f"/datasets/{ds}/reservations",
        json={"reservation_id": "res1", "epsilon": 0.07, "delta": 1e-8,
              "ttl_seconds": 3600},
    )
    specs = [
        ("c1", 0.10, 1e-8),
        ("c2", 0.05, 0.0),
        ("c3", 0.20, 1e-7),
        ("c4", 0.03, 0.0),
        ("c5", 0.15, 1e-9),
    ]
    for rid, e, d in specs:
        await _grant(client, ds, rid, e, d)

    pending = (0.25, 1e-8)
    body = (
        await _plan(
            client, ds,
            {"epsilon": pending[0], "delta": pending[1]},
            [s[0] for s in specs],
        )
    ).json()

    baseline = [(0.08, 1e-9), (0.07, 1e-8)]
    expected = _independent_plan(
        baseline,
        [(rid, e, d) for rid, e, d in specs],
        pending,
        cap,
        target_delta,
    )
    assert expected is not None
    chosen_ids, before, after, with_pending = expected
    assert body["feasible"] is True
    assert body["reversal_ids"] == chosen_ids
    assert body["reversal_count"] == len(chosen_ids)
    assert body["epsilon_before"] == pytest.approx(before, abs=1e-12)
    assert body["epsilon_after_reversal"] == pytest.approx(after, abs=1e-12)
    assert body["epsilon_with_pending"] == pytest.approx(
        with_pending, abs=1e-12
    )
    assert body["epsilon_with_pending"] <= cap + 1e-12
    # The baseline really participates: epsilon_before matches the live
    # budget's committed occupancy (granted + active reservation).
    budget = (await client.get(f"/datasets/{ds}/budget")).json()
    committed = budget["epsilon_consumed"] + budget["epsilon_reserved"]
    assert body["epsilon_before"] == pytest.approx(committed, abs=1e-12)


@pytest.mark.asyncio
async def test_expired_reservation_excluded_without_being_swept(client):
    ds = _new_dataset_id("expsweep")
    await _register(client, ds, eps=1.0, delta=1e-6)
    await _grant(client, ds, "a", 0.1, 1e-8)
    await client.post(
        f"/datasets/{ds}/reservations",
        json={"reservation_id": "exp", "epsilon": 0.4, "delta": 1e-8,
              "ttl_seconds": 3600},
    )
    # Backdating directly does not sweep: no budget-touching endpoint runs
    # between the backdate and the read-only plan.
    await _backdate(ds, "exp", datetime.now(timezone.utc) - timedelta(seconds=5))

    body = (
        await _plan(client, ds, {"epsilon": 0.1, "delta": 1e-8}, ["a"])
    ).json()
    # The past-TTL reservation is treated as inactive, so the occupancy
    # before any reversal only contains request "a".
    expected_before = max(0.0, rdp.evaluate_entries([(0.1, 1e-8)], 1e-6)[1])
    assert body["epsilon_before"] == pytest.approx(expected_before, abs=1e-12)
    # ...but the read-only plan must not sweep it itself.
    async with db.connect() as conn:
        row = await db.get_reservation(conn, ds, "exp")
        assert row["status"] == "granted"
        assert row["expired_at"] is None
    again = (
        await _plan(client, ds, {"epsilon": 0.1, "delta": 1e-8}, ["a"])
    ).json()
    assert again["epsilon_before"] == pytest.approx(expected_before, abs=1e-12)
    async with db.connect() as conn:
        row = await db.get_reservation(conn, ds, "exp")
        assert row["status"] == "granted"


@pytest.mark.asyncio
async def test_reversed_requests_excluded_from_baseline(client):
    ds = _new_dataset_id("revbase")
    await _register(client, ds, eps=1.0, delta=1e-6)
    await _grant(client, ds, "gone", 0.2, 1e-8)
    await client.post(
        f"/datasets/{ds}/requests/gone/reversals",
        json={"reversal_id": _new_reversal_id(), "reason": "cleanup"},
    )
    await _grant(client, ds, "a", 0.1, 1e-8)

    body = (
        await _plan(client, ds, {"epsilon": 0.1, "delta": 1e-8}, ["a"])
    ).json()
    expected_before = max(0.0, rdp.evaluate_entries([(0.1, 1e-8)], 1e-6)[1])
    assert body["epsilon_before"] == pytest.approx(expected_before, abs=1e-12)
    assert body["feasible"] is True
    assert body["reversal_ids"] == []


@pytest.mark.asyncio
async def test_plan_is_strictly_read_only(client):
    ds = _new_dataset_id("readonly")
    await _register(client, ds, eps=0.62, delta=1e-6)
    await _grant(client, ds, "big", 0.4, 1e-8)
    await _grant(client, ds, "small", 0.1, 1e-8)

    payload = {"pending_item": {"epsilon": 0.2, "delta": 1e-8},
               "request_ids": ["big", "small"]}
    first = (await client.post(f"/datasets/{ds}/reversal-plans",
                               json=payload)).json()
    second = (await client.post(f"/datasets/{ds}/reversal-plans",
                                json=payload)).json()
    for key in (
        "suggestion_id", "feasible", "reason", "pending_item",
        "candidate_count", "candidate_request_ids", "reversal_ids",
        "reversal_count", "epsilon_cap", "epsilon_before",
        "epsilon_after_reversal", "epsilon_with_pending",
        "rdp_best_order_after_reversal", "rdp_best_order_with_pending",
    ):
        assert first[key] == second[key]

    # No records of any kind were created and nothing was reversed.
    requests = (await client.get(f"/datasets/{ds}/requests")).json()
    assert requests["count"] == 2
    assert all(not item["reversed"] for item in requests["requests"])
    assert (await client.get(f"/datasets/{ds}/reservations")).json()["count"] == 0
    assert (await client.get(f"/datasets/{ds}/reversals")).json()["count"] == 0
    budget = (await client.get(f"/datasets/{ds}/budget")).json()
    assert budget["granted_requests"] == 2
    assert budget["epsilon_consumed"] == pytest.approx(
        first["epsilon_before"], abs=1e-12
    )


@pytest.mark.asyncio
async def test_plan_recomputes_against_live_ledger(client):
    ds = _new_dataset_id("live")
    await _register(client, ds, eps=0.62, delta=1e-6)
    await _grant(client, ds, "big", 0.4, 1e-8)
    await _grant(client, ds, "small", 0.1, 1e-8)

    before = (
        await _plan(client, ds, {"epsilon": 0.2, "delta": 1e-8},
                    ["big", "small"])
    ).json()
    assert before["reversal_ids"] == ["small"]

    # Once "small" is actually reversed, the pending item fits as-is.
    await client.post(
        f"/datasets/{ds}/requests/small/reversals",
        json={"reversal_id": _new_reversal_id(), "reason": "cleanup"},
    )
    after = (
        await _plan(client, ds, {"epsilon": 0.2, "delta": 1e-8}, ["big"])
    ).json()
    assert after["feasible"] is True
    assert after["reversal_ids"] == []
    assert after["suggestion_id"] != before["suggestion_id"]


@pytest.mark.asyncio
async def test_suggestion_id_is_deterministic_and_canonical(client):
    ds = _new_dataset_id("det")
    await _register(client, ds, eps=0.65, delta=1e-6)
    await _grant(client, ds, "req-b", 0.3, 1e-8)
    await _grant(client, ds, "req-a", 0.3, 1e-8)

    pending = {"epsilon": 0.2, "delta": 1e-8}
    first = (await _plan(client, ds, pending, ["req-a", "req-b"])).json()
    second = (await _plan(client, ds, pending, ["req-a", "req-b"])).json()
    # Repeated calls agree on the suggestion id...
    assert first["suggestion_id"] == second["suggestion_id"]
    # ...and the candidate order in the payload does not matter.
    reordered = (await _plan(client, ds, pending, ["req-b", "req-a"])).json()
    assert reordered["suggestion_id"] == first["suggestion_id"]
    # A different pending item yields a different suggestion id.
    other = (await _plan(client, ds, {"epsilon": 0.1, "delta": 1e-8},
                         ["req-a", "req-b"])).json()
    assert other["suggestion_id"] != first["suggestion_id"]


@pytest.mark.asyncio
async def test_twelve_candidate_boundary(client):
    ds = _new_dataset_id("twelve")
    await _register(client, ds, eps=1.0, delta=1e-6)
    ids = [f"r{i:02d}" for i in range(12)]
    for rid in ids:
        await _grant(client, ds, rid, 0.01, delta=0.0)
    r = await _plan(client, ds, {"epsilon": 0.01, "delta": 0.0}, ids)
    assert r.status_code == 200
    body = r.json()
    assert body["candidate_count"] == 12
    assert body["feasible"] is True
    # Pure epsilon-DP composes exactly: 0.13 <= 1.0 without any reversal.
    assert body["reversal_ids"] == []
    assert body["epsilon_before"] == pytest.approx(0.12, abs=1e-12)
    assert body["epsilon_with_pending"] == pytest.approx(0.13, abs=1e-12)


# ---------------------------------------------------------------- validation --


@pytest.mark.asyncio
async def test_missing_dataset_404(client):
    r = await _plan(client, "no-such-dataset",
                    {"epsilon": 0.1, "delta": 1e-8}, ["a"])
    assert r.status_code == 404
    assert r.json()["error"] == "dataset_not_found"


@pytest.mark.asyncio
async def test_missing_request_404(client):
    ds = _new_dataset_id("missing")
    await _register(client, ds)
    r = await _plan(client, ds, {"epsilon": 0.1, "delta": 1e-8}, ["nope"])
    assert r.status_code == 404
    assert r.json()["error"] == "request_not_found"
    assert r.json()["request_id"] == "nope"


@pytest.mark.asyncio
async def test_rejected_request_409(client):
    ds = _new_dataset_id("rejected")
    await _register(client, ds, eps=0.01, delta=1e-6)
    await client.post(
        f"/datasets/{ds}/requests",
        json={"request_id": "rej", "query_type": "q",
              "epsilon": 5.0, "delta": 1e-8},
    )
    r = await _plan(client, ds, {"epsilon": 0.1, "delta": 1e-8}, ["rej"])
    assert r.status_code == 409
    assert r.json()["error"] == "request_not_granted"


@pytest.mark.asyncio
async def test_reversed_request_409(client):
    ds = _new_dataset_id("reversed")
    await _register(client, ds)
    await _grant(client, ds, "a", 0.1, 1e-8)
    await client.post(
        f"/datasets/{ds}/requests/a/reversals",
        json={"reversal_id": _new_reversal_id(), "reason": "x"},
    )
    r = await _plan(client, ds, {"epsilon": 0.1, "delta": 1e-8}, ["a"])
    assert r.status_code == 409
    assert r.json()["error"] == "request_already_reversed"


@pytest.mark.asyncio
async def test_duplicate_request_ids_422(client):
    ds = _new_dataset_id("dup")
    await _register(client, ds)
    await _grant(client, ds, "a", 0.1, 1e-8)
    r = await _plan(client, ds, {"epsilon": 0.1, "delta": 1e-8}, ["a", "a"])
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_empty_and_oversized_lists_422(client):
    ds = _new_dataset_id("sizes")
    await _register(client, ds)
    pending = {"epsilon": 0.1, "delta": 1e-8}
    assert (await _plan(client, ds, pending, [])).status_code == 422
    assert (
        await _plan(client, ds, pending, [f"x{i}" for i in range(13)])
    ).status_code == 422


@pytest.mark.asyncio
async def test_invalid_request_id_shape_422(client):
    ds = _new_dataset_id("badshape")
    await _register(client, ds)
    pending = {"epsilon": 0.1, "delta": 1e-8}
    assert (await _plan(client, ds, pending, [""])).status_code == 422
    assert (await _plan(client, ds, pending, ["x" * 201])).status_code == 422


@pytest.mark.asyncio
async def test_invalid_pending_item_422(client):
    ds = _new_dataset_id("badpending")
    await _register(client, ds)
    await _grant(client, ds, "a", 0.1, 1e-8)
    for pending in (
        {"epsilon": 0.0, "delta": 1e-8},
        {"epsilon": -0.1, "delta": 1e-8},
        {"epsilon": 1000.1, "delta": 1e-8},
        {"epsilon": 0.1, "delta": 1.0},
        {"epsilon": 0.1, "delta": -0.1},
        {"epsilon": 0.1},
        {},
    ):
        r = await _plan(client, ds, pending, ["a"])
        assert r.status_code == 422, pending
