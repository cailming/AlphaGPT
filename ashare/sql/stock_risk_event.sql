-- P3.3-C: optional historical risk event register. Run manually after review.
-- Existing research code reads a CSV; this DDL is not executed by Python.
-- public.stock_info.st remains a current snapshot and must not populate
-- historical event rows without dated announcement evidence.

CREATE TABLE IF NOT EXISTS public.stock_risk_event (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    stock_code varchar(6) NOT NULL,
    event_type varchar(24) NOT NULL,
    effective_date date NOT NULL,
    published_on date NOT NULL,
    source_url text NOT NULL,
    source_id text NOT NULL,
    ingested_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT stock_risk_event_code_check
        CHECK (stock_code ~ '^[0-9]{6}$'),
    CONSTRAINT stock_risk_event_type_check
        CHECK (event_type IN (
            'RISK_ST', 'RISK_STAR_ST', 'RISK_NORMAL',
            'HALT', 'RESUME', 'DELISTED'
        )),
    CONSTRAINT stock_risk_event_source_check
        CHECK (length(btrim(source_id)) > 0
               AND source_url ~ '^https?://'),
    CONSTRAINT stock_risk_event_source_unique
        UNIQUE (stock_code, event_type, effective_date, source_id)
);

CREATE INDEX IF NOT EXISTS idx_stock_risk_event_asof
    ON public.stock_risk_event (stock_code, effective_date, published_on);

CREATE UNIQUE INDEX IF NOT EXISTS uq_stock_risk_event_status_transition
    ON public.stock_risk_event (stock_code, effective_date)
    WHERE event_type IN ('RISK_ST', 'RISK_STAR_ST', 'RISK_NORMAL');

CREATE UNIQUE INDEX IF NOT EXISTS uq_stock_risk_event_halt_transition
    ON public.stock_risk_event (stock_code, effective_date)
    WHERE event_type IN ('HALT', 'RESUME');

COMMENT ON TABLE public.stock_risk_event IS
    '历史股票风险/停牌公告事件，保留公告已知日期和生效日期。禁止用当前stock_info.st倒灌历史。';
COMMENT ON COLUMN public.stock_risk_event.stock_code IS '证券代码，6位字符，保留前导0';
COMMENT ON COLUMN public.stock_risk_event.event_type IS
    '事件类型：RISK_ST/ST风险警示；RISK_STAR_ST/*ST；RISK_NORMAL/撤销风险警示；HALT/停牌；RESUME/复牌；DELISTED/终止上市';
COMMENT ON COLUMN public.stock_risk_event.effective_date IS '事件实际生效日期，禁止作为当时已公开的替代';
COMMENT ON COLUMN public.stock_risk_event.published_on IS '公告公开日期（无分秒时保守按次一交易日可知）';
COMMENT ON COLUMN public.stock_risk_event.source_url IS '公告原始链接，必须留存';
COMMENT ON COLUMN public.stock_risk_event.source_id IS '公告编号或可唯一追溯的来源标识';
COMMENT ON COLUMN public.stock_risk_event.ingested_at IS '本地记录时间，不代表历史公告可知时间';
