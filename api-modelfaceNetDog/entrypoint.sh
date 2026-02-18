#!/bin/bash

# Exit on error
set -e

echo "Waiting for PostgreSQL to be ready..."
until pg_isready -h "${DB_HOST:-postgres}" -p "${DB_PORT:-5432}" -U "${DB_USER:-admin}" > /dev/null 2>&1; do
  echo "PostgreSQL is unavailable - sleeping"
  sleep 2
done

echo "PostgreSQL is ready!"

echo "Starting FastAPI application..."
exec "$@"
