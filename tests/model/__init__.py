import os

from tests.fixtures.environment_variables import *

os.environ['TABLE'] = TEST_DATASET
os.environ["DATABASE"] = PDB_TEST
os.environ["POSTGRES_USERNAME"] = POSTGRES
os.environ["POSTGRES_PASSWORD"] = POSTGRES
os.environ["POSTGRES_HOST"] = LOCALHOST
os.environ["POSTGRES_PORT"] = PORT
