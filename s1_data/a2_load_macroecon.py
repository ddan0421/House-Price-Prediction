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
    pmms_df = pd.read_csv(data)
else:
    print(f"Failed to fetch pmms 30yr data: {response.status_code}")


conn.execute("""
    create or replace table pmms as
        with cte as (
            select 
                date_trunc('month', cast(observation_date as DATE)) as Dt,
                avg(cast(MORTGAGE30US as double)) as pmms
            from pmms_df
            group by date_trunc('month', cast(observation_date as DATE))
        )
        select 
             A.Dt,
             '30yr' as Term,
             A.pmms / 100 as pmms,
             (A.pmms - B.pmms) / 100 as pmms_chg
        from cte as A
        join cte as B
          on B.Dt = A.Dt - interval 12 month
       order by A.Dt;
             """)


# Unemployment rate in Ames, IA
url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id=AMES119UR"
response = requests.get(url)
if response.status_code == 200:
    data = StringIO(response.text)
    ue_df = pd.read_csv(data)
else:
    print(f"Failed to fetch Ames Unemployment Rate data: {response.status_code}")

conn.execute("""
    create or replace table ue as
        with cte as (
            select 
                date_trunc('month', cast(observation_date as DATE)) as Dt,
                cast(AMES119UR as double) as ue
            from ue_df
        )
        select 
             A.Dt,
             A.ue / 100 as ue,
             (A.ue - B.ue) / 100 as ue_chg
        from cte as A
        join cte as B
          on B.Dt = A.Dt - interval 12 month
       order by A.Dt;
             """)


# All Employees Total NonFarm in Ames, IA (MSA)
url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id=AMES119NA"
response = requests.get(url)
if response.status_code == 200:
    data = StringIO(response.text)
    totalnonfarm_df = pd.read_csv(data)
else:
    print(f"Failed to fetch Total NonFarm data: {response.status_code}")


conn.execute("""
    create or replace table totalnonfarm as
        with cte as (
            select 
                date_trunc('month', cast(observation_date as DATE)) as Dt,
                cast(AMES119NA as double) as nonfarm
            from totalnonfarm_df
        )
        select 
             A.Dt,
             A.nonfarm,
             (A.nonfarm - B.nonfarm) / B.nonfarm as nonfarm_yoy
        from cte as A
        join cte as B
          on B.Dt = A.Dt - interval 12 month
       order by A.Dt;
            """)


print(f"Saved DuckDB database to {database_path}")

def join_macroecon(df):
    query = f"""
    create or replace table {df} as
        select 
            A.*,
            B.HPI,
            B.HPA,
            C.pmms,
            C.pmms_chg,
            D.ue,
            D.ue_chg,
            E.nonfarm,
            E.nonfarm_yoy
        from {df} as A
        left join ames_hpi as B
        on datediff('month', B.Dt, make_date(A.YrSold, A.MoSold, 1)) = 1
        left join pmms AS C
        on datediff('month', C.Dt, make_date(A.YrSold, A.MoSold, 1)) = 1
        left join ue AS D
        on datediff('month', D.Dt, make_date(A.YrSold, A.MoSold, 1)) = 1
        left join totalnonfarm AS E
        on datediff('month', E.Dt, make_date(A.YrSold, A.MoSold, 1)) = 1;

    """
    conn.execute(query)

join_macroecon("train")
join_macroecon("test")

print(conn.execute("SHOW TABLES").fetchall())
conn.close()