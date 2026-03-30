#!/usr/bin/env python3
"""
Database setup script for ML predictions
"""

import psycopg2
import os
import time
from datetime import datetime

def wait_for_db(host, port, database, user, password, max_retries=30):
    """Wait for database to be ready"""
    for i in range(max_retries):
        try:
            conn = psycopg2.connect(
                host=host,
                port=port,
                database=database,
                user=user,
                password=password
            )
            conn.close()
            print("Database is ready!")
            return True
        except psycopg2.OperationalError:
            print(f"Waiting for database... ({i+1}/{max_retries})")
            time.sleep(2)
    
    return False

def create_tables():
    """Create necessary tables for ML application"""
    
    # Database connection parameters
    db_params = {
        'host': os.getenv('DB_HOST', 'postgres'),
        'port': os.getenv('DB_PORT', '5432'),
        'database': os.getenv('DB_NAME', 'mldb'),
        'user': os.getenv('DB_USER', 'mluser'),
        'password': os.getenv('DB_PASSWORD', 'mlpassword')
    }
    
    # Wait for database to be ready
    if not wait_for_db(**db_params):
        print("Database not ready, exiting...")
        return False
    
    try:
        # Connect to database
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()
        
        # Create predictions table
        cursor.execute
