#!/usr/bin/env bash
#
# Install DrishtiTrust as a system service on homeserver, behind the existing
# Cloudflare tunnel. Same shape as voice-lab/deploy/install.sh.
#
#   sudo bash /home/abhijith/dr-screening/deploy/install.sh
#
# Prerequisites, done as abhijith (no root needed):
#   python3 -m venv .venv
#   .venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
#   .venv/bin/pip install -r requirements.txt     (opencv-python-headless is fine)
#   .venv/bin/python scripts/bootstrap.py --backend python --skip-install
#
# Idempotent: safe to re-run. It touches exactly three things --
#
#   /etc/systemd/system/drishtitrust.service
#   /etc/cloudflared/config.yml   ONE ingress rule added, backed up first
#   /srv/nas/drishtitrust         the database directory
#
# DNS is separate: `cloudflared tunnel route dns 6d4003bd-... drishtitrust.abhijith-sriram.in`
# as abhijith (his ~/.cloudflared/cert.pem authorises it).

set -euo pipefail

APP_USER=abhijith
APP_DIR=/home/abhijith/dr-screening
DATA_DIR=/srv/nas/drishtitrust
HOSTNAME_FQDN=drishtitrust.abhijith-sriram.in
PORT=8026
CF_CONFIG=/etc/cloudflared/config.yml

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mFAILED: %s\033[0m\n' "$*" >&2; exit 1; }

[ "$(id -u)" -eq 0 ]               || die "run this with sudo"
[ -x "$APP_DIR/.venv/bin/uvicorn" ] || die "no venv at $APP_DIR/.venv -- see the prerequisites at the top"
[ -f "$APP_DIR/runs/ens_d3/args.json" ] || die "models missing -- run scripts/bootstrap.py --backend python --skip-install"

install -d -o "$APP_USER" -g "$APP_USER" "$DATA_DIR"

say "Installing the service"
install -m 0644 "$APP_DIR/deploy/drishtitrust.service" /etc/systemd/system/drishtitrust.service
systemctl daemon-reload
systemctl enable drishtitrust >/dev/null
systemctl restart drishtitrust

for _ in $(seq 1 30); do
    curl -fs -o /dev/null "http://127.0.0.1:$PORT/" && break
    sleep 1
done
systemctl is-active --quiet drishtitrust || {
    journalctl -u drishtitrust -n 40 --no-pager
    die "drishtitrust did not stay up (log above)"
}
curl -fsS -o /dev/null "http://127.0.0.1:$PORT/" || die "nothing answering on 127.0.0.1:$PORT"
say "Service is up on 127.0.0.1:$PORT"

if grep -q "$HOSTNAME_FQDN" "$CF_CONFIG"; then
    say "Tunnel already routes $HOSTNAME_FQDN -- config left untouched"
else
    say "Adding one ingress rule to the existing tunnel"
    BACKUP="$CF_CONFIG.bak.$(date +%Y%m%d-%H%M%S)"
    cp -a "$CF_CONFIG" "$BACKUP"
    echo "    backup: $BACKUP"

    # Immediately before the catch-all: rules after http_status:404 never fire.
    awk -v host="$HOSTNAME_FQDN" -v port="$PORT" '
        !done && /^[[:space:]]*-[[:space:]]*service:[[:space:]]*http_status:404/ {
            print "  - hostname: " host
            print "    service: http://localhost:" port
            print "    originRequest:"
            print "      # a CPU screen takes ~12 s; the first one also loads the models"
            print "      connectTimeout: 30s"
            done = 1
        }
        { print }
        END { if (!done) exit 3 }
    ' "$BACKUP" > "$CF_CONFIG" || { cp -a "$BACKUP" "$CF_CONFIG"; die "no catch-all rule found -- config restored"; }

    # --config belongs to `tunnel`, not `ingress validate` (else it exits 0 unvalidated).
    if ! cloudflared tunnel --config "$CF_CONFIG" ingress validate; then
        cp -a "$BACKUP" "$CF_CONFIG"
        die "ingress validation failed -- config rolled back, tunnel untouched"
    fi

    say "Reloading cloudflared"
    systemctl restart cloudflared
    sleep 3
    systemctl is-active --quiet cloudflared || {
        cp -a "$BACKUP" "$CF_CONFIG"
        systemctl restart cloudflared
        die "cloudflared would not start -- config rolled back and restarted"
    }
fi

say "Done"
echo
echo "  https://$HOSTNAME_FQDN"
echo
echo "  Logs:    journalctl -u drishtitrust -f"
echo "  Restart: sudo systemctl restart drishtitrust"
echo
