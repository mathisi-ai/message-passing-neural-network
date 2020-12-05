import os

from tests.fixtures.postgres_variables import *
from message_passing_nn.utils.postgres_connector import PostgresConnector

os.environ['DATASET_TABLE'] = TEST_DATASET
os.environ['PENALTY_TABLE'] = TEST_PENALTY
os.environ["DATABASE"] = PDB_TEST
os.environ["POSTGRES_USERNAME"] = POSTGRES
os.environ["POSTGRES_PASSWORD"] = POSTGRES
os.environ["POSTGRES_HOST"] = LOCALHOST
os.environ["POSTGRES_PORT"] = PORT

postgres_connector = PostgresConnector()
postgres_connector.open_connection()
postgres_connector.create_table("message_passing_tests",
                                "(pdb_code varchar primary key, "
                                "features float[][], "
                                "neighbors float[][], "
                                "labels float[])")
