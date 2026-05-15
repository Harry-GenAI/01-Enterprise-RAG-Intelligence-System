import os
import psycopg2
from dotenv import load_dotenv
from logger import logger

load_dotenv()

def get_conn():
    logger.info("Creating DB connection")
    return psycopg2.connect(os.getenv("DB_url"))

#Create table
def create_table():
    logger.info("ensuring chat table exists")
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS chats(
    id SERIAL PRIMARY KEY,
    session_id TEXT,
    user_message TEXT,
    ai_response TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    conn.commit()
    cur.close()
    conn.close()

    logger.info("Table created")

#save chat
def save_chat(session_id, user_message, ai_response):
    try:
        conn= get_conn()
        cur = conn.cursor()
        
        cur.execute("""
        INSERT INTO chats(session_id, user_message, ai_response) VALUES(%s,%s,%s)
        """,(session_id,user_message,ai_response),
        )
        conn.commit()
        cur.close()
        conn.close()

        logger.info(f"chat saved| session={session_id}")
    
    except Exception:
        logger.exception("DB save failed")

#get_chat_history
def get_chat_history(sessionid, limit=5):
    logger.info(f"Loading memory | session={sessionid}")
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    SELECT user_message, ai_response
    FROM chats
    WHERE session_id=%s
    ORDER BY created_at DESC
    LIMIT %s

    """, (sessionid, limit),
    )

    rows = cur.fetchall()

    cur.close()
    conn.close()

    history=""
    for u,a in reversed(rows):
        history+=f"\nuser:{u}\nAssistant:{a}\n"
    
    return history



