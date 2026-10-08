#!/bin/bash
# Sets up a freshly rented box FROM THE PC (git-bash):  scripts/box/setup_box.sh <port> [host]   (host defaults to n1.de.clorecloud.net, key ~/.ssh/id_ed25519_clore)
# 1. checks the hardware (RAM >= 60 GB, free disk, GPU) and refuses to go on otherwise,  2. puts the helpers and the updated model catalogue on the box,
# 3. starts the downloads in the background (they run while the tests start).  Nothing secret is copied.
set -u
PORT="$1"; HOST="${2:-n1.de.clorecloud.net}"; KEY="${CLORE_KEY:-$HOME/.ssh/id_ed25519_clore}"
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
SSH="ssh -p $PORT -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new -i $KEY root@$HOST"
SCP="scp -q -P $PORT -o BatchMode=yes -i $KEY"
$SSH 'echo up' >/dev/null || { echo "the box does not answer on port $PORT"; exit 1; }
read -r RAM_GB DISK_GB < <($SSH 'echo $(free -g | awk "/Mem:/{print \$2}") $(df -BG --output=avail /workspace | tail -1 | tr -dc 0-9)')
echo "RAM ${RAM_GB} GB, free disk ${DISK_GB} GB, GPU: $($SSH 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader')"
[ "${RAM_GB:-0}" -ge 60 ] || { echo "REFUSED: less than 64 GB of RAM"; exit 2; }
[ "${DISK_GB:-0}" -ge 250 ] || { echo "REFUSED: less than 250 GB of free disk"; exit 3; }
$SSH 'mkdir -p /root/logs /workspace/models; cp -n /app/config/base_models.json /app/config/base_models.json.image 2>/dev/null; true'
$SCP "$HERE/scripts/box/start_app.sh" "$HERE/scripts/box/stop_app.sh" "$HERE/scripts/box/download_all.sh" "$HERE/scripts/box/dl_hf_repo.py" root@$HOST:/root/
$SCP "$HERE/config/base_models.json" root@$HOST:/app/config/base_models.json
$SCP "$HERE/config/loras.json" root@$HOST:/app/config/loras.json
$SSH 'chmod +x /root/*.sh /root/dl_hf_repo.py; cd /root && setsid nohup /root/download_all.sh > /root/logs/download_all.out 2>&1 < /dev/null & echo "downloads started"'
