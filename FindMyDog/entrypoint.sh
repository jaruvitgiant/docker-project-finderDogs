#!/bin/bash

# Exit on error
set -e

echo "Waiting for PostgreSQL to be ready..."
if [ -n "$DB_HOST" ]; then
    until pg_isready -h db -p 5432 -U $POSTGRES_USER; do
      echo "Waiting for database..."
      sleep 2
    done
    echo "PostgreSQL is up - executing command"
fi
# ลบ cron เก่าและแอด cron ใหม่จาก settings.py
python manage.py crontab add

# สั่งเริ่ม service cron ใน background
service cron start

# Run migrations
echo "Running database migrations..."
python manage.py migrate --noinput

# Collect static files
echo "Collecting static files..."
python manage.py collectstatic --noinput --clear

echo "Starting application..."
exec "$@"
