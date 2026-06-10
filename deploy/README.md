# Cloud deployment — run the monthly rebalance on a free-tier VM

This runs IB Gateway + the rebalance script on a small always-on cloud VM, so
the strategy trades every month **regardless of whether your personal computer
is on**.

## Why a VM (and which free tier)

The strategy is monthly, so the VM is idle 99.9% of the time — a tiny instance
is plenty. Best free options:

| Provider | Free offer | Good for |
|---|---|---|
| **Oracle Cloud "Always Free"** | 4 ARM cores / 24 GB RAM, **free forever** | ✅ Best — never expires |
| AWS EC2 free tier | t2.micro 750 h/mo, **12 months** | OK for a year |
| GCP free tier | e2-micro, **$300 credit / 90 days + small always-free** | OK to start |

A 1 GB RAM instance runs IB Gateway fine. Oracle's Always Free is the
recommended choice because it doesn't expire and the ARM instance is generous.

## Architecture

```
  ┌─────────────────── cloud VM (always on) ───────────────────┐
  │                                                            │
  │   IB Gateway (Docker)          cron (last trading day)     │
  │   ├─ IBC auto-login            └─ run_monthly.sh           │
  │   ├─ daily auto-restart           └─ run_live_rebalance.py │
  │   └─ API on 127.0.0.1:4002 ◄───────────┘ (connects here)   │
  │                                                            │
  └────────────────────────────────────────────────────────────┘
            │
            └──► orders transmitted to IBKR servers ──► fill at 4 PM close
```

Once orders reach IBKR, the VM's job is done — fills happen on IBKR's side.

## One-time setup

### 1. Create the VM
- Oracle Cloud → create an **Ampere (ARM) Always Free** instance, Ubuntu 22.04.
- Or AWS → t2.micro, Ubuntu. Or GCP → e2-micro, Ubuntu.
- SSH in. **Do not open ports 4001/4002 in the firewall** — they stay localhost-only.

### 2. Install Docker + clone the repo
```bash
sudo apt update && sudo apt install -y docker.io docker-compose-plugin git python3-pip
sudo usermod -aG docker $USER && newgrp docker
git clone <your-repo-url> ~/algo_trading
cd ~/algo_trading
pip3 install -r requirements.txt
```

### 3. Configure credentials
```bash
cd ~/algo_trading/deploy
cp .env.example .env
nano .env          # fill in TWS_USERID / TWS_PASSWORD, keep TRADING_MODE=paper
```
`.env` is gitignored — it never leaves the VM.

### 4. Start IB Gateway
```bash
docker compose up -d
docker compose logs -f ib-gateway     # watch it log in; Ctrl-C when settled
```

### 5. Verify connectivity (still safe — READ_ONLY_API=yes blocks trading)
```bash
cd ~/algo_trading
python3 scripts/run_live_rebalance.py --port 4002   # dry-run, paper Gateway
```
You should see the order list printed. No trades (dry-run + read-only).

### 6. Schedule it
Edit `deploy/run_monthly.sh` — set `REPO_DIR` to `/home/<youruser>/algo_trading`.
Then install the cron job:
```bash
crontab deploy/crontab.example
crontab -l        # confirm
```

## Going from paper to live (do this slowly)

1. **Paper, read-only, dry-run** (steps above) — confirm the order list looks sane.
2. **Paper, trading enabled**: set `READ_ONLY_API=no` in `.env`, `docker compose up -d`
   to restart. Let cron place orders on the **paper** account for **2–3 months**.
   Compare paper fills to what the backtest expected.
3. **Live**: only after paper looks right —
   - set `TRADING_MODE=live` in `.env`, restart Gateway
   - change `PORT=4001` in `run_monthly.sh`
   - keep position sizes small at first

## ⚠ Important caveats

- **2FA on live accounts.** IBKR requires two-factor auth. Paper accounts log in
  headless with just user/password. For **live**, you must use IBKR's
  "second factor device sharing" / IBKey settings so the Gateway can stay logged
  in — see the ib-gateway-docker docs. Plan for this before going live.
- **Daily restart.** IBKR force-restarts the Gateway once a day; IBC handles the
  re-login automatically. Your monthly script just needs the Gateway up at 3:30 PM ET.
- **Never expose the API port.** The compose file binds to `127.0.0.1` only.
  Anyone who can reach 4001/4002 can trade your account — keep it localhost.
- **Security.** Keep the VM patched, use SSH keys (not passwords), and consider a
  dedicated IBKR username with only the permissions this strategy needs.
- **Monitor it.** Check `data/cron.log` and `data/orders_*.csv` after each
  month-end so a silent failure (expired login, lost data subscription) doesn't go
  unnoticed.

## Files in this directory

| File | Purpose |
|---|---|
| `docker-compose.yml` | Headless IB Gateway (localhost-bound API) |
| `.env.example` | Credentials template → copy to `.env` (gitignored) |
| `run_monthly.sh` | Cron entrypoint with last-trading-day guards |
| `crontab.example` | The monthly schedule |
