import pymssql
import os
from dotenv import load_dotenv
import json
from datetime import datetime

# Load environment variables from .env file
load_dotenv("resources/.env")

# Database configuration from environment variables
DATABASE_CONFIG = {
    "server": os.getenv("DATABASE_URL", "192.168.1.19"),
    "port": int(os.getenv("DATABASE_PORT", 1433)),
    "user": os.getenv("DATABASE_USERNAME", "dev"),
    "password": os.getenv("DATABASE_PASSWORD", "Ab@123456"),
    "database": os.getenv("DATABASE_NAME", "LP_Bank_PROD"),
}
EVENT_TABLE_NAME = os.getenv("EVENT_TABLE_NAME", "View_Event")


def test_database_connection():
    """Test database connection and display connection info"""
    print("=== DATABASE CONNECTION INFORMATION ===")
    print(f"Host: {DATABASE_CONFIG['server']}")
    print(f"Port: {DATABASE_CONFIG['port']}")
    print(f"Database: {DATABASE_CONFIG['database']}")
    print(f"Username: {DATABASE_CONFIG['user']}")
    print(f"Password: {'*' * len(DATABASE_CONFIG['password'])}")
    print(f"Connection Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    


def get_first_5_rows():
    """Get first 5 rows from event table (EVENT_TABLE_NAME)"""
    try:
        # Connect to database
        conn = pymssql.connect(**DATABASE_CONFIG)
        cursor = conn.cursor(as_dict=True)

        print("=== CONNECTING TO DATABASE ===")
        print("✅ Successfully connected to MSSQL database")
        print()

        # Query first 5 rows from event table where UserCode is not null
        query = f"""
        SELECT TOP 5 
            [Id],
            [EventId],
            [UserCode],
            [UserName],
            [DepName],
            [AreaName],
            [DeviceName],
            [AccessTime],
            [EventTypeId],
            [Status],
            [PersonType],
            [PersonId],
            [AreaId],
            [CompanyId],
            [DeviceId],
            [ServerUrl],
            [Gender],
            [Age],
            [ScoreMatch]
        FROM [{EVENT_TABLE_NAME}] 
        WHERE [UserCode] IS NOT NULL
        ORDER BY [Id]
        """

        print("=== EXECUTING QUERY ===")
        print(
            f"Query: SELECT * FROM {EVENT_TABLE_NAME} WHERE UserCode IS NOT NULL LIMIT 5"
        )
        print()

        cursor.execute(query)
        rows = cursor.fetchall()

        print(f"=== FIRST 5 ROWS FROM {EVENT_TABLE_NAME} (UserCode IS NOT NULL) ===")
        if rows:
            for i, row in enumerate(rows, 1):
                print(f"\n--- Row {i} ---")
                for key, value in row.items():
                    # Truncate long values for better display
                    if isinstance(value, str) and len(value) > 50:
                        display_value = value[:47] + "..."
                    else:
                        display_value = value
                    print(f"{key}: {display_value}")
        else:
            print(f"No data found in {EVENT_TABLE_NAME} table")

        # Get table statistics
        print("\n=== TABLE STATISTICS ===")
        cursor.execute(f"SELECT COUNT(*) as total_rows FROM [{EVENT_TABLE_NAME}]")
        total_count = cursor.fetchone()["total_rows"]
        print(f"Total rows in {EVENT_TABLE_NAME}: {total_count}")

        # Get column information
        cursor.execute(
            """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns 
            WHERE table_name = %s
            ORDER BY ordinal_position
        """,
            (EVENT_TABLE_NAME,),
        )
        columns = cursor.fetchall()
        print(f"Total columns: {len(columns)}")
        print("\nColumn information:")
        for col in columns:
            print(
                f"  - {col['column_name']}: {col['data_type']} ({'NULL' if col['is_nullable'] == 'YES' else 'NOT NULL'})"
            )

        cursor.close()
        conn.close()
        print("\n✅ Database connection closed successfully")

    except pymssql.Error as e:
        print(f"❌ Database error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Main function"""
    print(f"🔍 Testing Database Connection and {EVENT_TABLE_NAME} Table")
    print("=" * 60)

    # Display database connection information
    test_database_connection()

    # Get first 5 rows from event table
    get_first_5_rows()

    print("\n" + "=" * 60)
    print("✅ Test completed successfully!")


if __name__ == "__main__":
    main()
