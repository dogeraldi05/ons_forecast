-- Ingestion, CSV variant. Everything arrives as text; typing happens in staging.
CREATE OR REPLACE VIEW raw_demanda_maxima AS
SELECT *
FROM read_csv('data/raw/DEMANDA_MAXIMA_DI_*.csv', delim = ';', header = true, all_varchar = true);
