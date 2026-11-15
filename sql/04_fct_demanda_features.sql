-- Features for a 7-day-ahead forecast. Every window is strictly backward-looking AND at least
-- 7 days old: on the forecast day only data up to t-7 is known, so lag_1d would be leakage here.
CREATE OR REPLACE VIEW fct_demanda_features AS
SELECT
    f.din_instante,
    f.id_subsistema,
    CASE f.id_subsistema WHEN 'SE' THEN 0 WHEN 'S' THEN 1 WHEN 'NE' THEN 2 WHEN 'N' THEN 3 END AS cod_subsistema,
    f.demanda_maxima_mw,

    LAG(f.demanda_maxima_mw, 7)   OVER w AS lag_7d,     -- also the seasonal naive forecast
    LAG(f.demanda_maxima_mw, 14)  OVER w AS lag_14d,
    LAG(f.demanda_maxima_mw, 364) OVER w AS lag_52w,

    AVG(f.demanda_maxima_mw) OVER (
        PARTITION BY f.id_subsistema
        ORDER BY f.din_instante
        ROWS BETWEEN 13 PRECEDING AND 7 PRECEDING
    ) AS media_movel_7d,                                -- last fully known week

    c.dia_semana,
    c.mes,
    c.dia_ano,
    c.feriado
FROM fct_demanda_diaria f
JOIN dim_calendario c USING (din_instante)
WINDOW w AS (PARTITION BY f.id_subsistema ORDER BY f.din_instante);
