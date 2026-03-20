import duckdb
import os 
from io import StringIO
import requests
import pandas as pd

base_folder = "data"
database = "AmesHousePrice.duckdb"
database_path = os.path.join(base_folder, database)

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

# 30-yr pmms
# Fetch csv from stlouis Fed site
#   - Check for successful response: code 200
url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"
response = requests.get(url)
if response.status_code == 200:
    data = StringIO(response.text)
    pmms = pd.read_csv(data)
else:
    print(f"Failed to fetch pmms 30yr data: {response.status_code}")


conn.execute("""
    create or replace table pmms as
        with cte as (
            select 
                date_trunc('month', cast(observation_date as DATE)) as Dt,
                avg(cast(MORTGAGE30US as double)) as pmms
            from pmms
            group by date_trunc('month', cast(observation_date as DATE))
        )
        select 
             A.Dt,
             '30yr' as Term,
             A.pmms,
             A.pmms - B.pmms as pmms_chg
        from cte as A
        join cte as B
          on B.Dt = A.Dt - interval 12 month
       order by A.Dt;
             """)


# Unemployment rate in Ames, IA
url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id=AMES119URN"
response = requests.get(url)
if response.status_code == 200:
    data = StringIO(response.text)
    ue = pd.read_csv(data)
else:
    print(f"Failed to fetch Ames Unemployment Rate data: {response.status_code}")

conn.execute("""
    create or replace table ue as
        with cte as (
            select 
                date_trunc('month', cast(observation_date as DATE)) as Dt,
                cast(AMES119URN as double) as ue
            from ue
        )
        select 
             A.Dt,
             A.ue,
             A.ue - B.ue as ue_chg
        from cte as A
        join cte as B
          on B.Dt = A.Dt - interval 12 month
       order by A.Dt;
             """)


# Property listing median days on market YoY in Ames, IA
url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id=MEDDAYONMARYY11180"
response = requests.get(url)
if response.status_code == 200:
    data = StringIO(response.text)
    dom = pd.read_csv(data)
else:
    print(f"Failed to fetch Median Days on Market data: {response.status_code}")


conn.execute("""
    create or replace table dom_yoy as
        select 
            date_trunc('month', cast(observation_date as DATE)) as Dt,
            cast(MEDDAYONMARYY11180 as double) as dom_yoy
        from dom;
            """)

print(conn.execute("SHOW TABLES").fetchall())
conn.close()

print(f"Saved DuckDB database to {database_path}")