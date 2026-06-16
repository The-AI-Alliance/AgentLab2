# Relax Reddit (Postmill) rate limits so the agent doesn't get throttled mid-task.
# Shared by reddit_launch.sh and all_sites_launch.sh (injected once from
# resources.py — see _REDDIT_RELAX). All steps are best-effort: a missing file
# or a failed exec must not fail the launch.
# See https://github.com/gasse/webarena-setup/commit/b4426309

docker exec webarena_reddit sh -lc '
if [ -f /srv/forum/app/DataSource/SubmissionData.php ]; then
  sed -i \
    -e "s/1 hour/2 minutes/g" \
    -e "s/5 minutes/2 minutes/g" \
    -e "s/max=3/max=50/g" \
    -e "s/max=15/max=50/g" \
    /srv/forum/app/DataSource/SubmissionData.php
else
  echo "[launch][warn] Missing /srv/forum/app/DataSource/SubmissionData.php; skipping reddit rate-limit patch"
fi
' || true

docker exec webarena_reddit sh -lc '
if [ -f /srv/forum/app/DataSource/CommentData.php ]; then
  sed -i \
    -e "s/5 minutes/2 minutes/g" \
    -e "s/max=10/max=50/g" \
    /srv/forum/app/DataSource/CommentData.php
else
  echo "[launch][warn] Missing /srv/forum/app/DataSource/CommentData.php; skipping reddit rate-limit patch"
fi
' || true

docker exec webarena_reddit sh -lc '
if [ -f /srv/forum/app/DataSource/UserData.php ]; then
  sed -i \
    -e '"'"'s/max="3"/max="50"/g'"'"' \
    -e "s/1 hour/2 minutes/g" \
    /srv/forum/app/DataSource/UserData.php
else
  echo "[launch][warn] Missing /srv/forum/app/DataSource/UserData.php; skipping reddit rate-limit patch"
fi
' || true

docker exec webarena_reddit sh -lc '
if [ -f /srv/forum/bin/console ]; then
  php /srv/forum/bin/console cache:clear
else
  echo "[launch][warn] Missing /srv/forum/bin/console; skipping cache:clear"
fi
' || true
docker exec webarena_reddit php -r "opcache_reset();" 2>/dev/null || true
echo "[launch] Reddit rate limits relaxed"
