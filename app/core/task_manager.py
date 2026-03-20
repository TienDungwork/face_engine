import asyncio
from datetime import datetime, timedelta
from app.core.config import app_config
from app.core.centroid_manager import centroid_manager
from app.core.local_db import local_db_manager


class TaskManager:
    """Manager for handling scheduled tasks."""

    def __init__(self):
        self.tasks = []

    async def start_scheduled_tasks(self):
        """Start all scheduled tasks."""
        print("Starting scheduled tasks...")

        # Start sync task
        if app_config.RELOAD_DB:
            self.tasks.append(asyncio.create_task(self._hourly_sync_task()))

        # Start centroid reload task
        if app_config.RELOAD_CENTROID:
            self.tasks.append(asyncio.create_task(
                self._centroid_reload_task()))

        print(f"Started {len(self.tasks)} scheduled tasks")

    async def _hourly_sync_task(self):
        """Hourly sync task at configured interval."""
        while True:
            try:
                # Perform sync
                print(
                    f"Starting hourly sync at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                await local_db_manager.sync_persons()
                print(
                    f"Hourly sync completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                # Wait for the configured interval (in hours)
                sync_interval_seconds = app_config.SYNC_INTERVAL_HOURS * 3600
                print(f"Next sync in {app_config.SYNC_INTERVAL_HOURS} hour(s)")
                await asyncio.sleep(sync_interval_seconds)

            except Exception as e:
                print(f"Error in hourly sync task: {str(e)}")
                # Wait 1 minute before retrying if there's an error
                await asyncio.sleep(60)

    async def _centroid_reload_task(self):
        """Daily centroid reload task at configured time."""
        while True:
            try:
                # Calculate time until next reload time
                now = datetime.now()
                target_time = now.replace(
                    hour=app_config.CENTROID_RELOAD_HOUR,
                    minute=app_config.CENTROID_RELOAD_MINUTE,
                    second=0,
                    microsecond=0
                )

                # If it's already past reload time today, schedule for tomorrow
                if now >= target_time:
                    target_time += timedelta(days=1)

                # Calculate seconds to wait
                seconds_to_wait = (target_time - now).total_seconds()
                print(
                    f"Next centroid reload scheduled at {target_time.strftime('%Y-%m-%d %H:%M:%S')} (in {seconds_to_wait:.0f} seconds)")

                # Wait until reload time
                await asyncio.sleep(seconds_to_wait)

                # Perform centroid reload
                print(
                    f"Starting centroid reload at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                centroid_manager.reload_centroid_embeddings()
                print(
                    f"Centroid reload completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            except Exception as e:
                print(f"Error in centroid reload task: {str(e)}")
                # Wait 1 minute before retrying if there's an error
                await asyncio.sleep(60)

    async def stop_all_tasks(self):
        """Stop all scheduled tasks."""
        print("Stopping scheduled tasks...")
        for task in self.tasks:
            task.cancel()

        # Wait for all tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)

        print("All scheduled tasks stopped")


# Create a singleton instance
task_manager = TaskManager()
