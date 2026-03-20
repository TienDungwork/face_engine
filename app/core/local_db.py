import sqlite3
from typing import List, Dict, Any
import asyncio
from datetime import datetime
import aiohttp
from app.core.config import app_config


class LocalDatabaseManager:
    def __init__(self):
        self.db_path = "local_persons.db"
        self._lock = asyncio.Lock()
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create persons table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS persons (
            personId TEXT PRIMARY KEY,
            compId INTEGER,
            personCode TEXT,
            fullname TEXT,
            deptName TEXT,
            position TEXT,
            jobDuties TEXT,
            passport TEXT,
            phoneNumber TEXT,
            personType INTEGER,
            avatarPath TEXT,
            faceFeature TEXT,
            faceStatus INTEGER,
            status INTEGER,
            is_deleted INTEGER DEFAULT 0,
            last_sync TIMESTAMP
        )
        ''')

        conn.commit()
        conn.close()

    async def save_persons(self, persons: List[Dict[str, Any]]):
        """Save or update persons in local database"""
        async with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            current_time = datetime.utcnow().isoformat()

            for person in persons:
                # Normalize keys from different API formats (camelCase, snake_case)
                normalized_person: Dict[str, Any] = {}

                # ID fields
                normalized_person['personId'] = person.get(
                    'personId') or person.get('id')
                normalized_person['compId'] = person.get(
                    'compId') or person.get('company_id')
                normalized_person['personCode'] = person.get(
                    'personCode') or person.get('code')
                normalized_person['fullname'] = person.get(
                    'fullname') or person.get('full_name')

                # Optional descriptive fields
                normalized_person['deptName'] = person.get(
                    'deptName') or person.get('department_name') or person.get('department_id')
                normalized_person['position'] = person.get(
                    'position')
                normalized_person['jobDuties'] = person.get(
                    'jobDuties') or person.get('job_duties')
                normalized_person['passport'] = person.get(
                    'passport') or person.get('citizen_id')
                normalized_person['phoneNumber'] = person.get(
                    'phoneNumber') or person.get('phone_number')
                normalized_person['personType'] = person.get(
                    'personType') or person.get('person_type')
                normalized_person['avatarPath'] = person.get(
                    'avatarPath') or person.get('avatar_path')

                # Face feature / status
                face_feature = person.get('faceFeature') or person.get(
                    'face_feature')
                if isinstance(face_feature, (bytes, bytearray)):
                    face_feature = face_feature.decode(
                        'utf-8', errors='replace')
                normalized_person['faceFeature'] = face_feature

                normalized_person['faceStatus'] = person.get(
                    'faceStatus') or person.get('face_match')

                # Status & delete flag
                normalized_person['status'] = person.get('status')
                delete_flag = person.get('delete')
                if delete_flag is None:
                    delete_flag = person.get('is_delete', False)
                normalized_person['delete'] = delete_flag

                # Prepare values for insertion/update
                values = (
                    normalized_person.get('personId'),
                    normalized_person.get('compId'),
                    normalized_person.get('personCode'),
                    normalized_person.get('fullname'),
                    normalized_person.get('deptName'),
                    normalized_person.get('position'),
                    normalized_person.get('jobDuties'),
                    normalized_person.get('passport'),
                    normalized_person.get('phoneNumber'),
                    normalized_person.get('personType'),
                    normalized_person.get('avatarPath'),
                    normalized_person.get('faceFeature'),
                    normalized_person.get('faceStatus'),
                    normalized_person.get('status'),
                    int(bool(normalized_person.get('delete', False))),
                    current_time
                )

                # Insert or update record
                cursor.execute('''
                INSERT OR REPLACE INTO persons 
                (personId, compId, personCode, fullname, deptName, position, jobDuties, 
                passport, phoneNumber, personType, avatarPath, faceFeature, faceStatus, 
                status, is_deleted, last_sync)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', values)

            conn.commit()
            conn.close()

    async def get_all_persons(self) -> List[Dict[str, Any]]:
        """Retrieve all persons from local database"""
        async with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('SELECT * FROM persons WHERE is_deleted = 0')
            columns = [description[0] for description in cursor.description]
            persons = []

            for row in cursor.fetchall():
                person = dict(zip(columns, row))
                persons.append(person)

            conn.close()
            return persons

    async def sync_persons(self):
        """Sync persons from remote API to local database.
        Nếu bất kỳ 1 page nào lỗi ko sync được, ngay lập tức dừng việc sync và không delete db.
        """
        try:
            async with aiohttp.ClientSession() as session:
                # Get total pages
                async with session.get(app_config.PAGING_URL) as response:
                    if response.status != 200:
                        print(f"Failed to get total pages: {response.status}")
                        return
                    paging_data = await response.json()
                    total_pages = paging_data.get('data', 1)
                    print(f"Total pages to sync: {total_pages}")

                # Get persons data from each page
                all_persons = []
                for page in range(total_pages):
                    page_url = f"{app_config.GET_PERSON_URL}/{page}"
                    print(f"Syncing page {page + 1}/{total_pages}")

                    async with session.get(page_url) as response:
                        if response.status != 200:
                            print(
                                f"Failed to get persons data for page {page}: {response.status}")
                            # Dừng việc sync và không xóa db
                            print("Sync aborted. Database is not modified.")
                            return

                        persons_data = await response.json()
                        if persons_data.get('status') == 200 and 'data' in persons_data:
                            all_persons.extend(persons_data['data'])
                            print(
                                f"Successfully synced {len(persons_data['data'])} persons from page {page + 1}")
                        else:
                            print(
                                f"Failed to get valid persons data for page {page}")
                            # Dừng việc sync và không xóa db
                            print("Sync aborted. Database is not modified.")
                            return

                # Save all persons to local database
                if all_persons:
                    # Clear the persons table before saving new data
                    async with self._lock:
                        conn = sqlite3.connect(self.db_path)
                        cursor = conn.cursor()
                        cursor.execute('DELETE FROM persons')
                        conn.commit()
                        conn.close()
                    await self.save_persons(all_persons)
                    print(
                        f"Successfully saved total {len(all_persons)} persons to local database (rewritten)")
                    # Reload persons in database manager
                    from app.core.database_manager import database_manager
                    await database_manager.load_persons()
                else:
                    print("No persons data to save")

        except Exception as e:
            print(f"Error during sync: {str(e)}")


# Create a singleton instance
local_db_manager = LocalDatabaseManager()
