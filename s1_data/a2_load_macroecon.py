import duckdb
import os 

folder = "data"
database = "AmesHousePrice.duckdb"

database_path = os.path.join(folder, database)

conn = duckdb.connect(database=database_path, read_only=False)


# Freddie Mac HPI
conn.execute("""
    create or replace table ames_hpi as
        with cte as (
            select
                make_date(Year, Month, 1) as Dt,
                cast(Year as int) as Year,
                cast(Month as int) as Month,
                cast(Index_SA as double) as HPI
            from read_csv_auto('https://www.freddiemac.com/fmac-resources/research/docs/fmhpi_master_file.csv')       
            where GEO_Type = 'CBSA' and GEO_Name ilike '%Ames%' and GEO_Code = '11180'
        )
        select
             A.Dt,
             A.Year,
             A.Month,
             A.HPI,
             (A.HPI - B.HPI) / B.HPI as HPA
        from cte as A
        join cte as B
          on B.Dt = A.Dt - interval 12 month
       order by A.Dt;
             
             """)


print(conn.execute("SHOW TABLES").fetchall())
conn.close()

print(f"Saved DuckDB database to {database_path}")