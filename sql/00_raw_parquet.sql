-- Ingestion, Parquet variant. Types come from the file schema.
CREATE OR REPLACE VIEW raw_demanda_maxima AS
SELECT *
FROM read_parquet('data/raw/DEMANDA_MAXIMA_DI_*.parquet');
