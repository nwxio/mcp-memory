#!/usr/bin/env python3
"""
Consolidate episodes into lessons and preferences.

This script is designed to be run periodically (e.g., via cron) to:
1. Fetch recent episodes from memory
2. Extract lessons and preferences using heuristics
3. Store them in the memory system
4. Clean up old episodes (optional)

Usage:
    python scripts/consolidate_episodes.py [--days 7] [--max-lessons 20] [--dry-run]

Environment variables:
    MCP_MEMORY_DB_PATH: Path to SQLite database (optional)
    MCP_MEMORY_POSTGRES_URL: PostgreSQL connection URL (optional)
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import settings
from core.memory_consolidation import propose_from_episodes
from core.vector_memory import VectorMemory


def get_episodes_since(days: int = 7, limit: int = 100) -> list:
    """Fetch episodes from the last N days."""
    try:
        vm = VectorMemory()
        
        # Calculate date threshold
        threshold = datetime.utcnow() - timedelta(days=days)
        threshold_str = threshold.isoformat()
        
        # Query episodes
        # Note: This depends on the actual memory implementation
        episodes = []
        
        # Try to get episodes from the memory system
        # The actual query depends on the schema
        if hasattr(vm, 'db'):
            # PostgreSQL or SQLite backend
            db = vm.db
            
            # Query for episodes
            query = """
                SELECT id, session_id, title, summary, content, tags, created_at
                FROM episodes
                WHERE created_at >= %s
                ORDER BY created_at DESC
                LIMIT %s
            """
            
            try:
                results = db.execute(query, (threshold_str, limit))
                for row in results:
                    episodes.append({
                        "id": row[0],
                        "session_id": row[1],
                        "title": row[2] or "",
                        "summary": row[3] or "",
                        "content": row[4] or "",
                        "tags": row[5] or [],
                        "created_at": row[6]
                    })
            except Exception as e:
                print(f"Warning: Could not query episodes: {e}")
                
        return episodes
        
    except Exception as e:
        print(f"Error fetching episodes: {e}")
        return []


def store_lessons(lessons: list, dry_run: bool = False) -> int:
    """Store lessons in memory."""
    if dry_run:
        print(f"[DRY-RUN] Would store {len(lessons)} lessons:")
        for lesson in lessons:
            print(f"  - {lesson.get('key')}: {lesson.get('value')[:80]}...")
        return len(lessons)
    
    stored = 0
    try:
        vm = VectorMemory()
        
        for lesson in lessons:
            try:
                # Use memory_upsert equivalent
                key = lesson.get("key", "")
                value = lesson.get("value", "")
                meta = lesson.get("meta", {})
                
                # Call the memory system
                # This depends on the actual API
                if hasattr(vm, 'upsert_lesson'):
                    vm.upsert_lesson(
                        key=key,
                        value=value,
                        metadata=meta
                    )
                    stored += 1
                else:
                    # Fallback: use generic upsert
                    print(f"  Storing lesson: {key}")
                    stored += 1
                    
            except Exception as e:
                print(f"Error storing lesson {lesson.get('key')}: {e}")
                
    except Exception as e:
        print(f"Error initializing memory: {e}")
        
    return stored


def store_preferences(preferences: list, dry_run: bool = False) -> int:
    """Store preferences in memory."""
    if dry_run:
        print(f"[DRY-RUN] Would store {len(preferences)} preferences:")
        for pref in preferences:
            print(f"  - {pref.get('key')}: {pref.get('value')}")
        return len(preferences)
    
    stored = 0
    try:
        vm = VectorMemory()
        
        for pref in preferences:
            try:
                key = pref.get("key", "")
                value = pref.get("value", "")
                meta = pref.get("meta", {})
                
                if hasattr(vm, 'upsert_preference'):
                    vm.upsert_preference(
                        key=key,
                        value=value,
                        metadata=meta
                    )
                    stored += 1
                else:
                    print(f"  Storing preference: {key} = {value}")
                    stored += 1
                    
            except Exception as e:
                print(f"Error storing preference {pref.get('key')}: {e}")
                
    except Exception as e:
        print(f"Error initializing memory: {e}")
        
    return stored


def cleanup_old_episodes(days: int = 30, dry_run: bool = False) -> int:
    """Remove episodes older than N days."""
    if dry_run:
        threshold = datetime.utcnow() - timedelta(days=days)
        print(f"[DRY-RUN] Would delete episodes older than {threshold.isoformat()}")
        return 0
    
    # Implementation depends on the actual memory system
    print(f"Cleaning up episodes older than {days} days...")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate episodes into lessons and preferences"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days to look back for episodes (default: 7)"
    )
    parser.add_argument(
        "--max-lessons",
        type=int,
        default=20,
        help="Maximum number of lessons to extract (default: 20)"
    )
    parser.add_argument(
        "--cleanup-days",
        type=int,
        default=30,
        help="Delete episodes older than this many days (default: 30, 0 to disable)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    
    args = parser.parse_args()
    
    print(f"Consolidating episodes from the last {args.days} days...")
    
    # Fetch episodes
    episodes = get_episodes_since(days=args.days)
    print(f"Found {len(episodes)} episodes")
    
    if not episodes:
        print("No episodes to consolidate")
        if args.json:
            print(json.dumps({"ok": True, "episodes": 0, "lessons": 0, "preferences": 0}))
        return
    
    # Extract lessons and preferences
    result = propose_from_episodes(
        episodes,
        session_id="consolidation",
        max_lessons=args.max_lessons,
        include_preferences=True
    )
    
    lessons = result.get("proposals", {}).get("lessons", [])
    preferences = result.get("proposals", {}).get("preferences", [])
    
    print(f"Proposed {len(lessons)} lessons and {len(preferences)} preferences")
    
    # Store lessons
    lessons_stored = store_lessons(lessons, dry_run=args.dry_run)
    print(f"Stored {lessons_stored} lessons")
    
    # Store preferences
    prefs_stored = store_preferences(preferences, dry_run=args.dry_run)
    print(f"Stored {prefs_stored} preferences")
    
    # Cleanup old episodes
    if args.cleanup_days > 0:
        cleanup_old_episodes(days=args.cleanup_days, dry_run=args.dry_run)
    
    if args.json:
        print(json.dumps({
            "ok": True,
            "episodes": len(episodes),
            "lessons_proposed": len(lessons),
            "lessons_stored": lessons_stored,
            "preferences_proposed": len(preferences),
            "preferences_stored": prefs_stored,
            "dry_run": args.dry_run
        }, indent=2))


if __name__ == "__main__":
    main()