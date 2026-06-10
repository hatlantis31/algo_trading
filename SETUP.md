# Complete Setup Guide — Cloud VM + IBKR Live Trading

This guide walks from zero to a fully automated monthly rebalancer running on a
free cloud VM. Total time: ~2 hours on first setup, then fully hands-off.

---

## Prerequisites (before you start)

- An **IBKR account** (paper account is free — sign up at interactivebrokers.com)
- An **Oracle Cloud account** (free forever — cloud.oracle.com)
- An **SSH key pair** on your local machine (`~/.ssh/id_rsa` or `~/.ssh/id_ed25519`)
- Basic terminal comfort (copy-paste from this guide is enough)

---

## Part 1 — IBKR Account Setup

### Step 1 — Enable paper trading

1. Log in to **Client Portal** (interactivebrokers.com → Log In → Client Portal).
2. Go to **Settings → Account Settings → Paper Trading Account**.
3. Click **Create Paper Trading Account** if you don't have one.
   - Paper accounts are completely separate from real money — safe to experiment.

### Step 2 — Enable the API on your paper account

1. In Client Portal, switch to your **Paper Account** (top-right account selector).
2. Go to **Settings → Account Settings → API**.
3. Enable **"ActiveX and Socket Clients"**.
4. Set **Trusted IP: 127.0.0.1** (localhost only — this is important for security).
5. You do NOT need to disable 2FA for paper accounts when using IB Gateway via
   IBC — IBC handles headless login automatically.

> **Note:** For a real/live account, the API setup is identical but you will also
> need to configure IBKey or "Second Factor Device Sharing" so Gateway can log in
> unattended. See the "Going Live" section at the bottom.

---

## Part 2 — Create the Cloud VM

### Step 3 — Create an Oracle Cloud "Always Free" instance

This is free forever with no credit card expiry risk.

1. Sign up at **cloud.oracle.com** → Create Account (free tier, requires credit card
   for identity verification, not charged).

2. In the Oracle Cloud console, go to **Compute → Instances → Create Instance**.

3. Configure:
   - **Name**: `algo-trading` (or anything)
   - **Image**: Ubuntu 22.04 (Canonical)
   - **Shape**: Click "Change Shape" → select **Ampere (VM.Standard.A1.Flex)**
     - 4 OCPUs, 24 GB RAM — all free forever under Always Free
     - (If unavailable in your region, use 1 OCPU / 6 GB — still plenty)
   - **Networking**: default VCN is fine; ensure "Assign public IPv4" is checked
   - **SSH Keys**: upload your **public key** (`~/.ssh/id_rsa.pub` or `id_ed25519.pub`)
     - If you don't have one: `ssh-keygen -t ed25519` on your local machine

4. Click **Create**. Note the **Public IP address** once it's running (~1 minute).

5. **Open port 22 in the firewall** (SSH):
   - Oracle Cloud → Networking → Virtual Cloud Networks → your VCN
   → Security Lists → Default Security List → Add Ingress Rules
   → Source: 0.0.0.0/0, Destination Port: 22, Protocol: TCP
   - **Do NOT open ports 4001 or 4002** — they stay localhost-only.

### Step 4 — SSH into the VM

```bash
# From your local machine:
ssh ubuntu@<YOUR_VM_PUBLIC_IP>
```

You're now inside the VM. All following commands run here.

---

## Part 3 — Install Software on the VM

### Step 5 — Install Docker, Git, and Python

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y docker.io docker-compose-plugin git python3-pip python3-venv
sudo usermod -aG docker $USER
newgrp docker          # apply group change without logging out
docker --version       # verify: Docker version 24.x or later
```

### Step 6 — Clone the repository

```bash
git clone https://github.com/<your-username>/algo_trading.git ~/algo_trading
cd ~/algo_trading
```

### Step 7 — Set up Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> **Tip:** Add `source ~/algo_trading/.venv/bin/activate` to your `~/.bashrc`
> so it auto-activates on login.

---

## Part 4 — Configure IB Gateway

### Step 8 — Set up your credentials

```bash
cd ~/algo_trading/deploy
cp .env.example .env
nano .env
```

Fill in your details:

```bash
TWS_USERID=your_ibkr_username_here     # your IBKR paper account login
TWS_PASSWORD=your_ibkr_password_here   # your IBKR paper account password
TRADING_MODE=paper                     # KEEP as "paper" until you're confident
READ_ONLY_API=yes                      # KEEP as "yes" — blocks any order from being placed
TZ=America/New_York                    # VM timezone (must match NYSE hours)
```

Save and exit nano: `Ctrl+O`, `Enter`, `Ctrl+X`.

> **Security:** `.env` is gitignored and stays only on this VM. Never commit
> real credentials. `READ_ONLY_API=yes` is a hard block from IB Gateway — even
> if the script tries to trade, Gateway will reject it.

### Step 9 — Set the VM timezone

```bash
sudo timedatectl set-timezone America/New_York
timedatectl       # confirm: Time zone: America/New_York
```

This ensures cron runs at the right time relative to NYSE close (4 PM ET).

### Step 10 — Start IB Gateway

```bash
cd ~/algo_trading/deploy
docker compose up -d
```

Watch the login process:

```bash
docker compose logs -f ib-gateway
```

You should see lines like:
```
Starting IB Gateway...
Logging in...
Login successful
API server started on port 4004
```

Press `Ctrl+C` to stop following logs (Gateway keeps running in background).

**Wait 30–60 seconds** for Gateway to fully initialize before testing.

> **Troubleshooting:** If login fails, check your username/password in `.env`.
> The paper account login is at `paper.interactivebrokers.com` — make sure the
> paper account is activated in Client Portal (Step 1).

---

## Part 5 — First Test (Dry-Run)

### Step 11 — Run the strategy in dry-run mode

```bash
cd ~/algo_trading
source .venv/bin/activate
python scripts/run_live_rebalance.py --port 4002
```

This connects to your paper Gateway and computes what orders would be placed —
but places nothing (dry-run + READ_ONLY_API blocks trading anyway).

Expected output:
```
08:30:01 INFO    Connecting to IBKR at port 4002…
08:30:02 INFO    Connected. Client ID 1, server version 176
08:30:02 INFO    Universe: 100 tickers from committed parquet
08:30:03 INFO    Fetching historical data for 100 symbols…
08:31:32 INFO    NAV: $1,000,000.00
08:31:32 INFO    Current positions: 0 holdings

  ORDERS (dry-run)
  ┌────────┬────────┬──────┬──────────────┬──────────────┬───────────┬──────────────┐
  │ Ticker │ Action │  Qty │ Current $    │ Target $     │ Price     │ Order Value  │
  ├────────┼────────┼──────┼──────────────┼──────────────┼───────────┼──────────────┤
  │ NVDA   │ BUY    │   12 │        $0.00 │    $9,989.88 │   $832.49 │    $9,989.88 │
  │ MSFT   │ BUY    │   25 │        $0.00 │   $10,009.25 │   $400.37 │   $10,009.25 │
  ...

DRY-RUN complete. Pass --execute to send these orders.
```

If you see this, everything is working correctly.

---

## Part 6 — Paper Trading with Cron (Automated Monthly Orders)

### Step 12 — Enable paper order placement (remove read-only)

Once the dry-run looks right, allow the Gateway to actually place orders on the
**paper** account:

```bash
nano ~/algo_trading/deploy/.env
```

Change:
```
READ_ONLY_API=no
```

Restart the Gateway to apply:
```bash
cd ~/algo_trading/deploy
docker compose down && docker compose up -d
```

### Step 13 — Configure the cron script

```bash
nano ~/algo_trading/deploy/run_monthly.sh
```

Change `REPO_DIR` to your actual path (it defaults to `/home/ubuntu/algo_trading`):
```bash
REPO_DIR="/home/ubuntu/algo_trading"   # confirm this matches your actual path
PORT=4002                               # 4002 = paper Gateway (correct for now)
EXECUTE="--execute"                     # leave this — it places orders on paper account
```

Save and exit.

Make it executable:
```bash
chmod +x ~/algo_trading/deploy/run_monthly.sh
```

Test that the script runs (safe — paper account only):
```bash
~/algo_trading/deploy/run_monthly.sh
```

### Step 14 — Install the cron job

```bash
crontab ~/algo_trading/deploy/crontab.example
crontab -l    # confirm it was installed
```

Expected output:
```
30 15 28-31 * * /home/ubuntu/algo_trading/deploy/run_monthly.sh >> /home/ubuntu/algo_trading/data/cron.log 2>&1
```

This runs at 3:30 PM Eastern Time on days 28–31 of each month. The script
inside has its own guards: it skips weekends and skips non-final weekdays, so
it only actually executes on the last trading day of the month.

---

## Part 7 — Monitoring

### Step 15 — Check logs after each month-end

After the cron fires:

```bash
# Check the cron log
tail -50 ~/algo_trading/data/cron.log

# Check what orders were placed
ls ~/algo_trading/data/orders_*.csv
cat ~/algo_trading/data/orders_2025-01-31.csv   # example
```

### Step 16 — Check Gateway is still running

IB Gateway restarts itself once daily (IBKR requirement). IBC handles the
re-login automatically. To verify it's healthy:

```bash
docker compose -f ~/algo_trading/deploy/docker-compose.yml ps
docker compose -f ~/algo_trading/deploy/docker-compose.yml logs --tail 20 ib-gateway
```

If Gateway is down:
```bash
cd ~/algo_trading/deploy && docker compose up -d
```

### Optional: email alerts on failure

Add this to `run_monthly.sh` near the top to get an email if the script fails:

```bash
trap 'echo "Rebalance FAILED at $(date)" | mail -s "algo-trading error" your@email.com' ERR
```

Requires `sudo apt install -y mailutils` and mail server configuration.

---

## Part 8 — Going Live (Do This Slowly)

Only proceed after **2–3 months of paper trading** that look correct
(order list matches expectations, fills happen at close, no errors in cron.log).

### Checklist before going live

- [ ] Paper trading ran for at least 2 months without errors
- [ ] Paper fills roughly matched backtest expectations (not dollar amounts, logic)
- [ ] You understand the strategy's expected behaviour in different markets
- [ ] You've read all the caveats in `deploy/README.md`
- [ ] Your IBKR live account has funds and the API is enabled
- [ ] You've configured IBKey / Second Factor Device Sharing for unattended login
  (Settings → Security → Two Factor Authentication → IBKey Sharing)

### Step 17 — Switch to live

```bash
nano ~/algo_trading/deploy/.env
```

Change:
```
TWS_USERID=your_LIVE_ibkr_username    # may differ from paper
TWS_PASSWORD=your_LIVE_ibkr_password
TRADING_MODE=live                     # switches Gateway to live session
READ_ONLY_API=no
```

```bash
nano ~/algo_trading/deploy/run_monthly.sh
```

Change:
```bash
PORT=4001    # 4001 = live Gateway (was 4002 for paper)
```

Restart Gateway:
```bash
cd ~/algo_trading/deploy
docker compose down && docker compose up -d
docker compose logs -f ib-gateway    # confirm "Login successful"
```

Do a final dry-run to confirm the live connection:
```bash
cd ~/algo_trading
python scripts/run_live_rebalance.py --port 4001   # dry-run (no --execute)
```

When ready to let cron trade live, no further changes needed — `run_monthly.sh`
already has `EXECUTE="--execute"` which will now send orders to the live account.

> **Start small.** Consider passing `--capital 10000` in `run_monthly.sh` for the
> first live month to limit exposure while you verify everything works end-to-end.

---

## Quick Reference

| Task | Command |
|------|---------|
| Start Gateway | `cd deploy && docker compose up -d` |
| Stop Gateway | `cd deploy && docker compose down` |
| Gateway logs | `docker compose logs -f ib-gateway` |
| Dry-run paper | `python scripts/run_live_rebalance.py --port 4002` |
| Dry-run live | `python scripts/run_live_rebalance.py --port 4001` |
| Place paper orders | `python scripts/run_live_rebalance.py --port 4002 --execute` |
| Check cron log | `tail -f data/cron.log` |
| View last orders | `ls data/orders_*.csv` |
| Re-install cron | `crontab deploy/crontab.example` |

---

## Security Reminders

- **Never expose ports 4001/4002** to the internet — they're bound to `127.0.0.1` only
- **`.env` is gitignored** — your password never leaves the VM
- **Keep the VM patched**: `sudo apt update && sudo apt upgrade -y` monthly
- **SSH keys only** — disable password SSH: `sudo nano /etc/ssh/sshd_config` →
  set `PasswordAuthentication no`, then `sudo systemctl restart sshd`
- **Monitor monthly** — a silent failure (expired login, data subscription lapse)
  won't warn you unless you check `cron.log`
