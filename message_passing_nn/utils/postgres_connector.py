import os
import psycopg2


class PostgresConnector:
    def __init__(self):
        self._connection = None
        self._cursor = None

    def open_connection(self):
        self._connection = psycopg2.connect(database=os.environ["DATABASE"],
                                            user=os.environ["POSTGRES_USERNAME"],
                                            password=os.environ["POSTGRES_PASSWORD"],
                                            host=os.environ["POSTGRES_HOST"],
                                            port=os.environ["POSTGRES_PORT"])
        self._cursor = self._connection.cursor()

    def execute_query(self, fields: list, table: str, where: str = None, order_by: list = None, limit: int = None):
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

    def close_connection(self):
        self._cursor.close()
        self._connection.close()
