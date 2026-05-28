# Possible System Failures in the ADAPTS-HCT Study

Brainstormed failure modes for the ADAPTS-HCT RL API and study integration.

---

## 1. Server Fails to Call API

| Failure | Description |
|---------|-------------|
| **Server forgets to call API** | Server misses scheduled calls (e.g., Sunday 3AM update, 6AM game action, 9AM message actions). |
| **Scheduler/cron job fails** | Cron or system scheduler doesn't run or crashes, so no API calls are made. |
| **Server crashes before making calls** | Server process dies before or during the batch of API calls. |
| **Wrong endpoint called** | Server calls wrong URL (e.g., `/action` instead of `/add_group` or vice versa). |

---

## 2. Data Inconsistency Between Server and RL Database

| Failure | Description |
|---------|-------------|
| **Server data contaminated** | Server's local DB becomes corrupted or inconsistent with RL API state. |
| **Server and RL DB out of sync** | Server thinks a dyad is active but RL DB has different groups; or vice versa. |
| **Duplicate group registration** | Server registers same dyad twice; RL API rejects with "Group already exists." |
| **Server loses track of decision indices** | Server reuses or skips decision_idx, causing conflicts or gaps. |
| **Server forgets which groups are active** | Server stops calling for dyads that were recruited but still active. |

---

## 3. Decision Index and Request Validation Failures

| Failure | Description |
|---------|-------------|
| **Decision index already exists** | Server sends same `(group_id, decision_idx)` twice; RL API rejects. |
| **Wrong decision_type for time** | Server requests `dyad_game` at 9AM instead of 6AM, or wrong `decision_type` for the slot. |
| **Missing or invalid context** | Server sends `context` without `cur_var` or `past3_vars`; RL API returns mark these as missing and try to recover these information from later calls. |
| **Group not found** | Server requests action for a group_id that was never registered. |

---

## 4. RL API Resource and Performance Failures

| Failure | Description |
|---------|-------------|
| **Memory ran out for RL API** | API process OOM (e.g., Thompson Sampling params grow large; DB connections accumulate). |
| **CPU overload** | Model update blocks too long; API becomes unresponsive. |
| **Database connection pool exhausted** | Too many concurrent requests; API hangs or fails. |
| **Disk full** | RL API can't write logs, backups, or DB; fails silently or crashes. |
| **RL API process killed** | OOM killer, system crash, or manual restart. |

---

## 5. Model Update Failures

| Failure | Description |
|---------|-------------|
| **Update runs too long** | Update blocks; API can't serve new action requests during update. |
| **Update fails mid-run** | Exception in update logic; DB partially updated; model inconsistent. |
| **Callback URL unreachable** | Server's callback URL is down or wrong; update "succeeds" but server never sees completion. |
| **Backup fails before update** | BACKUP_DATABASE=True but disk full or backup path invalid; update aborted or inconsistent. |
| **No data for update** | Update runs with empty StudyData; algorithm may behave unexpectedly. |

---

## 6. Network and Connectivity Failures

| Failure | Description |
|---------|-------------|
| **Network timeout** | Server's request to RL API times out; server retries or gives up. |
| **RL API unreachable** | Server can't reach RL API (DNS, firewall, wrong port). |
| **Intermittent connectivity** | Some requests succeed, others fail; partial state. |
| **SSL/TLS errors** | Certificate issues if HTTPS is used. |

---

## 7. Database Failures

| Failure | Description |
|---------|-------------|
| **PostgreSQL down** | RL API can't connect; all requests fail. |
| **Database corruption** | Data corruption; inconsistent reads or writes. |
| **Migration not applied** | New schema (e.g., ThompsonSamplingParams) not migrated; runtime errors. |
| **Transaction rollback** | Partial commit; Action saved but StudyData not, or vice versa. |

---

## 8. Timing and Clock Failures

| Failure | Description |
|---------|-------------|
| **Server clock drift** | Server timestamps wrong; decision ordering or time logic affected. |
| **Timezone mismatch** | Server uses UTC but RL expects EST; scheduled calls at wrong times. |
| **Daylight saving change** | 3AM/6AM/9AM EST shifts when DST changes; missed or duplicated calls. |
| **Update and action race** | Server calls action before update completes; stale model used. |

---

## 9. Data Quality and Algorithm Failures

| Failure | Description |
|---------|-------------|
| **Context values out of range** | `cur_var` or `past3_vars` invalid (NaN, extreme); algorithm or DB errors. |
| **Reward never observed** | Outcome/upload_data never called; StudyData has placeholder outcome only. |
| **Model parameters not initialized** | ModelParameters table empty; action returns 404. |

---

## 10. Deployment and Configuration Failures

| Failure | Description |
|---------|-------------|
| **Wrong config** | RL_ALGORITHM, DB URI, or port misconfigured; wrong behavior. |
| **Environment variable missing** | DATABASE_URL or similar not set; connection fails. |
| **Port conflict** | Port 5001 in use (e.g., AirPlay on 5000); API can't bind. |
| **Multiple API instances** | Two instances share same DB; race conditions on updates. |

---

## 11. Human and Operational Failures

| Failure | Description |
|---------|-------------|
| **Manual DB reset** | Someone runs `flask reset-db`; all data lost. |
| **Wrong study prefix** | Simulation uses wrong prefix; duplicate groups or conflicts. |
| **Server redeployed without RL sync** | Server restored from backup; RL DB has newer data; inconsistency. |

---
