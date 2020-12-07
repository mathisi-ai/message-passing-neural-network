import os
import psycopg2


class PostgresConnector:
    def __init__(self):
        self.penalty_table = os.environ['PENALTY_TABLE']
        self.dataset_table = os.environ['DATASET_TABLE']
        self._connection = None
        self._cursor = None

    def open_connection(self):
        self._connection = psycopg2.connect(database=os.environ["DATABASE"],
                                            user=os.environ["POSTGRES_USERNAME"],
                                            password=os.environ["POSTGRES_PASSWORD"],
                                            host=os.environ["POSTGRES_HOST"],
                                            port=os.environ["POSTGRES_PORT"])
        self._cursor = self._connection.cursor()

    def execute_query(self,
                      fields: list,
                      where: str = None,
                      order_by: list = None,
                      limit: int = None,
                      use_case: str = None):
        if use_case == 'penalty':
            table = self.penalty_table
        else:
            table = self.dataset_table
        sql = """select {} from {} """.format(",".join(fields), table)
        if where:
            sql += """where {} """.format(where)
        if order_by:
            sql += """order by {}""".format(",".join(order_by))
        if limit:
            sql += """limit {}""".format(str(limit))
        sql += """;"""
        self._cursor.execute(sql)
        return self._cursor.fetchall()

    def execute_insert_dataset(self, pdb_code, features, neighbors, labels):
        sql = """insert into {} (pdb_code, features, neighbors, labels) """.format(self.dataset_table)
        sql += """values ('{}', '{}', '{}', '{}') """.format(pdb_code, features, neighbors, labels)
        sql += """on conflict do nothing;"""
        self._cursor.execute(sql)
        self._connection.commit()

    def execute_insert_penalty(self, residue, penalty):
        sql = """insert into {} (residue, penalty) """.format(self.penalty_table)
        sql += """values ('{}', '{}') """.format(residue, penalty)
        sql += """on conflict do nothing;"""
        self._cursor.execute(sql)
        self._connection.commit()

    def create_table(self, table_name: str, fields: str):
        sql = """create table if not exists {} {};""".format(table_name, fields)
        self._cursor.execute(sql)
        self._connection.commit()

    def truncate_table(self, table_name: str):
        sql = """truncate table {};""".format(table_name)
        self._cursor.execute(sql)
        self._connection.commit()

    def close_connection(self):
        self._cursor.close()
        self._connection.close()
