import os
import dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
dotenv.load_dotenv(dotenv_path)


DB = os.environ.get("DB")
USR = os.environ.get("USR")
PASS = os.environ.get("PASS")
HOST = os.environ.get("HOST")
PORT = os.environ.get("PORT")
API_KEY = os.environ.get("API_KEY")


postgres_dev_str = f'postgresql://{USR}:{PASS}@{HOST}:{PORT}/{DB}'

