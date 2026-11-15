-- Calendar 2024-2027. Covers the forecast year, so 2027 rows get the same features as history.
-- `feriados` (date, name) is registered from Python (`holidays` package, national + optional).
CREATE OR REPLACE VIEW dim_calendario AS
SELECT
    d.din_instante,
    ISODOW(d.din_instante)                  AS dia_semana,   -- 1 = Monday
    MONTH(d.din_instante)                   AS mes,
    DAYOFYEAR(d.din_instante)               AS dia_ano,
    YEAR(d.din_instante)                    AS ano,
    CAST(f.data IS NOT NULL AS INTEGER)     AS feriado
FROM (
    SELECT CAST(range AS DATE) AS din_instante
    FROM range(DATE '2024-01-01', DATE '2028-01-01', INTERVAL 1 DAY)
) d
LEFT JOIN feriados f ON f.data = d.din_instante;
