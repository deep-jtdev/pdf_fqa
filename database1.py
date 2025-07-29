import sqlite3
import json

class create_db:
    def __init__(self, token, chunk_json1, filename):
        conn = sqlite3.connect('my_database.db')
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS token_data (
                token_id TEXT PRIMARY KEY,
                chunk_data TEXT,
                filename TEXT 
            )
        """)

        chunk_json = json.dumps(chunk_json1)

        try:
            cursor.execute(
                "INSERT INTO token_data (token_id, chunk_data, filename) VALUES (?, ?, ?)",
                (token, chunk_json, filename)
            )
            conn.commit()
            print({"message": f"✅ {filename} uploaded and stored successfully"})
        except sqlite3.IntegrityError:
            print({"error": f"❌ Token already exists for: {filename}"})

        conn.close()

    @staticmethod
    def get_all_filenames():
        conn = sqlite3.connect('my_database.db')
        cursor = conn.cursor()
        cursor.execute("SELECT filename FROM token_data")
        rows = cursor.fetchall()
        conn.close()
        return {"pdfs": [{"filename": row[0]} for row in rows]}

    # ✅ NEW: Fetch record by filename and token
    @staticmethod
    def fetch_by_token_or_filename(identifier):
        conn = sqlite3.connect('my_database.db')
        cursor = conn.cursor()
        cursor.execute("""
            SELECT token_id, chunk_data, filename, full_content
            FROM token_data
            WHERE token_id = ? OR filename = ?
        """, (identifier, identifier))
        result = cursor.fetchone()
        conn.close()
    
        if result:
            return {
                "token": result[0],
                "chunk_data": result[1],
                "filename": result[2],
            }
        else:
            return {"error": "No matching record found for token or filename."}