-- One row per subsystem per day. A date spine makes gaps explicit (NULL) instead of silently dropped,
-- which also keeps the ROWS-based lags in fct_demanda_features aligned to calendar days.
CREATE OR REPLACE VIEW fct_demanda_diaria AS
WITH subs AS (SELECT DISTINCT id_subsistema, nom_subsistema FROM stg_demanda_maxima),
     dias AS (
         SELECT din_instante FROM dim_calendario
         WHERE din_instante BETWEEN (SELECT MIN(din_instante) FROM stg_demanda_maxima)
                                AND (SELECT MAX(din_instante) FROM stg_demanda_maxima)
     )
SELECT
    d.din_instante,
    s.id_subsistema,
    s.nom_subsistema,
    st.demanda_maxima_mw,
    st.demanda_integralizada_mwmed,
    st.din_pico
FROM dias d
CROSS JOIN subs s
LEFT JOIN stg_demanda_maxima st
       ON st.din_instante = d.din_instante AND st.id_subsistema = s.id_subsistema;
