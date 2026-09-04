"""
The audit and risk endpoints — the ones that let the UI say WHY.

The point of these tests is a single distinction: "nothing was attempted" and
"attempts were made and refused" must never render the same way. A dashboard
that shows an empty trade list for both is telling the operator nothing, and
that is precisely the failure this endpoint exists to fix.
"""

from __future__ import annotations

from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.api.routes.audit import _check_identifier, _headline, _plain_reason, _to_entry
from app.database.models import Base
from tests.test_e2e_paper_trade import (
    PRICE,
    WIDE_RISK,
    buy_signal,
    make_stack,
    sell_signal,
)

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def session_factory():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


def entries_of(stack: Any) -> list:
    return [_to_entry(r.to_dict()) for r in stack.audit.sinks[0].records]


# =========================================================================== #
#  A rejection must name its cause                                            #
# =========================================================================== #

class TestRejectionsExplainThemselves:

    async def test_a_risk_rejection_names_the_limit_not_just_failure(
        self, session_factory
    ):
        """'No trade' is not an answer. The gate has to be named."""
        stack = await make_stack(session_factory)
        await stack.service.submit_signal(
            buy_signal(), 10, reference_price=PRICE,
            **{**WIDE_RISK, "max_daily_risk": 1.0, "daily_risk_used": 1.0},
        )
        e = entries_of(stack)[-1]

        assert not e.submitted
        assert e.reason == "daily risk limit"
        assert e.headline == "RELIANCE BUY x10 rejected — daily risk limit"
        assert e.detail and "Rs 500" in e.detail, (
            "the numeric specifics were lost, so an operator cannot see by how "
            "much the limit was breached"
        )

    async def test_the_detail_keeps_its_original_casing(self, session_factory):
        """Lowercasing turned 'used Rs 1' into 'used rs 1'."""
        stack = await make_stack(session_factory)
        await stack.service.submit_signal(
            buy_signal(), 10, reference_price=PRICE,
            **{**WIDE_RISK, "max_daily_risk": 1.0, "daily_risk_used": 1.0},
        )
        e = entries_of(stack)[-1]
        assert "Rs" in (e.detail or ""), f"currency was mangled: {e.detail!r}"

    async def test_a_kill_switch_block_says_so(self, session_factory):
        stack = await make_stack(session_factory)
        stack.kill_switch_store.engage("operator halt")
        await stack.service.submit_signal(
            buy_signal(), 5, reference_price=PRICE, **WIDE_RISK
        )
        e = entries_of(stack)[-1]
        assert e.headline == "RELIANCE BUY x5 blocked — kill switch engaged"
        assert e.kill_switch_active is True

    async def test_a_BROKER_side_rejection_is_not_reported_as_submitted(
        self, session_factory
    ):
        """
        An order can pass every one of our gates and be refused at the venue.

        Reporting the execution outcome alone (SUBMITTED — it did reach the
        broker) would tell an operator the order went through.
        """
        stack = await make_stack(session_factory)
        await stack.service.submit_signal(
            sell_signal(), 40, reference_price=PRICE, **WIDE_RISK
        )
        e = entries_of(stack)[-1]

        assert "rejected by broker" in e.headline, e.headline
        assert "short selling" in e.headline.lower()
        assert not e.headline.endswith("submitted")

    async def test_the_headline_never_repeats_itself(self, session_factory):
        """It read 'blocked — kill switch engaged — kill switch engaged'."""
        stack = await make_stack(session_factory)
        stack.kill_switch_store.engage("halt")
        await stack.service.submit_signal(
            buy_signal(), 5, reference_price=PRICE, **WIDE_RISK
        )
        e = entries_of(stack)[-1]
        assert e.headline.count("—") <= 1, e.headline
        assert e.headline.count("kill switch") == 1, e.headline

    async def test_a_successful_order_reports_its_broker_id(self, session_factory):
        stack = await make_stack(session_factory)
        await stack.service.submit_signal(
            buy_signal(), 10, reference_price=PRICE, **WIDE_RISK
        )
        e = entries_of(stack)[-1]
        assert e.submitted
        assert "submitted" in e.headline
        assert e.broker_order_id


class TestReasonExtraction:
    """Unit-level: the parsing that turns an engine string into plain language."""

    def test_a_composite_check_yields_its_identifier(self):
        assert _check_identifier(
            "risk_limit: Daily risk limit would be breached: used Rs 1"
        ) == "RISK_LIMIT"

    def test_a_bare_identifier_still_works(self):
        assert _check_identifier("SECTOR_EXPOSURE") == "SECTOR_EXPOSURE"

    def test_a_known_check_becomes_plain_language(self):
        r = _plain_reason({"failed_risk_checks": ["sector_exposure: over cap"]})
        assert r == "sector exposure limit"

    def test_an_unknown_check_is_still_readable(self):
        r = _plain_reason({"failed_risk_checks": ["some_new_gate: whatever"]})
        assert r == "some new gate"

    def test_a_record_with_no_reason_falls_back_to_the_outcome(self):
        assert _plain_reason({"outcome": "BLOCKED_MODE"}) == "trading mode not authorised"

    def test_the_users_example_renders_as_requested(self):
        """RELIANCE BUY rejected / Reason: sector exposure limit."""
        record = {
            "symbol": "RELIANCE", "side": "BUY", "quantity": 12,
            "outcome": "BLOCKED_RISK",
            "failed_risk_checks": ["sector_exposure: ENERGY at 25% cap"],
        }
        reason = _plain_reason(record)
        assert reason == "sector exposure limit"
        assert _headline(record, reason) == (
            "RELIANCE BUY x12 rejected — sector exposure limit"
        )


# =========================================================================== #
#  The endpoint                                                               #
# =========================================================================== #

class TestAuditEndpoint:

    async def _client(self, session_factory, monkeypatch):
        import httpx

        import app.execution.runtime as runtime_mod
        from app.main import create_app
        from tests.test_e2e_paper_trade import MARKET_OPEN_IST, DeterministicFeed

        real = runtime_mod.build_production_stack

        async def build(**kw):
            kw["session_factory"] = session_factory
            kw["data_broker"] = DeterministicFeed()
            kw["paper_clock"] = lambda: MARKET_OPEN_IST
            kw["paper_state_path"] = None
            return await real(**kw)

        monkeypatch.setattr(runtime_mod, "build_production_stack", build)
        app = create_app()
        return app, httpx

    async def test_it_returns_rejections_with_reasons(
        self, session_factory, monkeypatch
    ):
        app, httpx = await self._client(session_factory, monkeypatch)
        async with app.router.lifespan_context(app):
            svc = app.state.execution_service
            await svc.submit_signal(
                buy_signal(), 10, reference_price=PRICE,
                **{**WIDE_RISK, "max_daily_risk": 1.0, "daily_risk_used": 1.0},
            )
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://t"
            ) as c:
                r = await c.get("/api/v1/audit?rejected_only=true")

        assert r.status_code == 200
        body = r.json()
        assert body["entries"], "a rejection was recorded but not returned"
        assert body["rejected"] >= 1
        top = body["entries"][0]
        assert top["reason"], "a rejection with no reason is what this replaced"
        assert "rejected" in top["headline"]

    async def test_an_empty_trail_says_nothing_was_attempted(
        self, session_factory, monkeypatch
    ):
        """
        Distinguishing "no attempts" from "attempts refused" is the whole point.
        """
        app, httpx = await self._client(session_factory, monkeypatch)
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://t"
            ) as c:
                body = (await c.get("/api/v1/audit")).json()

        if not body["entries"]:
            assert body["unavailable_reason"]
            assert "NOT that attempts were made" in body["unavailable_reason"]

    async def test_filters_work(self, session_factory, monkeypatch):
        app, httpx = await self._client(session_factory, monkeypatch)
        async with app.router.lifespan_context(app):
            svc = app.state.execution_service
            await svc.submit_signal(
                buy_signal(), 10, reference_price=PRICE, **WIDE_RISK
            )
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://t"
            ) as c:
                hit = (await c.get("/api/v1/audit?symbol=RELIANCE")).json()
                miss = (await c.get("/api/v1/audit?symbol=NOSUCH")).json()

        assert hit["entries"]
        assert not miss["entries"]


class TestRiskEndpoint:

    async def test_limits_are_served_without_authentication(
        self, session_factory, monkeypatch
    ):
        """The configured limits are not account data."""
        import httpx

        from app.main import create_app

        app = create_app()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            r = await c.get("/api/v1/risk/limits")

        assert r.status_code == 200
        limits = r.json()["limits"]
        assert limits
        names = {row["name"] for row in limits}
        assert "max_single_stock_pct" in names
        assert "max_sector_pct" in names

    async def test_an_unmeasurable_limit_is_flagged_not_zeroed(
        self, session_factory, monkeypatch
    ):
        """
        A limit with no current reading must not render as comfortable.

        Reporting 0% utilisation because the account could not be read looks
        exactly like safety, which is the most dangerous thing a risk page can
        do.
        """
        import httpx

        from app.main import create_app

        app = create_app()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            limits = (await c.get("/api/v1/risk/limits")).json()["limits"]

        unmeasured = [row for row in limits if row["current"] is None]
        assert unmeasured, "every limit claimed to be measurable with no book"
        for row in unmeasured:
            assert row["measurable"] is False
            assert row["utilisation"] is None
            assert row["breached"] is False
            assert row["detail"]

    async def test_state_requires_authentication(self, session_factory):
        import httpx

        from app.main import create_app

        app = create_app()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            r = await c.get("/api/v1/risk/state")
        assert r.status_code == 401
