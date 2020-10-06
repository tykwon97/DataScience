import pymysql
import pandas as pd


def load_db_score_1():
    conn = pymysql.connect(host='localhost', user='datascience', password='datascience', db='university')
    curs = conn.cursor(pymysql.cursors.DictCursor)
        
    drop_sql = """drop table if exists db_score """
    curs.execute(drop_sql)
    conn.commit()
    
    
    create_sql = """
        create table db_score (
                sno int primary key,
                attendance float,
                homework float,
                discussion int,
                midterm float,
                final float,
                score float,
                grade char(1)
                )
        """
    curs.execute(create_sql)
    conn.commit()
    
    xl_file = 'db_score.xlsx'
    db_score = pd.read_excel(xl_file)
    
    rows = []
    
    for t in db_score.values:
        rows.append(tuple(t))
    
    insert_sql = """insert into db_score(sno, attendance, homework, discussion, midterm, final, score, grade)
                    values(%s, %s, %s, %s, %s, %s, %s, %s)"""
    curs.executemany(insert_sql, rows)                
    conn.commit()


def load_db_score_2():
    xl_file = 'db_score.xlsx'
    db_score = pd.read_excel(xl_file)

    conn = pymysql.connect(host='localhost', user='datascience', password='datascience', db='university')
    curs = conn.cursor(pymysql.cursors.DictCursor)
        
    drop_sql = """drop table if exists db_score """
    curs.execute(drop_sql)
    conn.commit()
    
    import sqlalchemy
    database_username = 'datascience'
    database_password = 'datascience'
    database_ip       = 'localhost'
    database_name     = 'university'
    database_connection = sqlalchemy.create_engine('mysql+pymysql://{0}:{1}@{2}/{3}'.
                                                   format(database_username, database_password, 
                                                          database_ip, database_name))
    db_score.to_sql(con=database_connection, name='db_score', if_exists='replace')    




def select_db_score():
    conn = pymysql.connect(host='localhost', user='datascience', password='datascience', db='university')
    curs = conn.cursor(pymysql.cursors.DictCursor)
    
    select_sql = """select sno, midterm, final 
                    from db_score
                    where midterm >=20 and final >= 20
                    order by sno"""
    
    curs.execute(select_sql)
    r = curs.fetchone()
    
    while r:
        print(r)
        r = curs.fetchone()
        
    curs.close()
    conn.close()
    
    
load_db_score_2()
select_db_score()    