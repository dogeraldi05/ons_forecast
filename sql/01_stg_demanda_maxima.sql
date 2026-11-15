-- Staging: typing, renaming, trimming padded codes, deduplication, analysis period.
CREATE OR REPLACE VIEW stg_demanda_maxima AS
SELECT DISTINCT
    CAST(dat_referencia AS DATE)                AS din_instante,
    TRIM(id_subsistema)                         AS id_subsistema,
    TRIM(nom_subsistema)                        AS nom_subsistema,
    CAST(val_demandainstantanea AS DOUBLE)      AS demanda_maxima_mw,
    CAST(val_demandaintegralizada AS DOUBLE)    AS demanda_integralizada_mwmed,
    CAST(din_demandainstantanea AS TIMESTAMP)   AS din_pico
FROM raw_demanda_maxima
WHERE CAST(dat_referencia AS DATE) BETWEEN DATE '2024-01-01' AND DATE '2026-12-31';
