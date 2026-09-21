# Azure deployment — every command, in order

Deploying the fake-review forensics backend to Azure with a student account.

**Nothing here has been run for you.** Each command needs you signed into your own Azure
account, so these are yours to execute. Codespaces is the cheaper route and is written up in
`deploy/README.md`; this file is Azure only.

---

## 0. Before you start

| | |
|---|---|
| credit | Azure for Students: **$100**, 12 months, no credit card |
| VM size | **Standard_B2s** (2 vCPU, 4 GiB) — the realistic minimum |
| **do NOT use** | **Standard_B1s** (1 GiB). The server peaks at **1.1 GB resident / 2.3 GB committed**, so the OOM killer takes it on the first photo. The "free" tier is a trap here. |
| region | `centralindia` is closest; `eastus` is usually cheapest |

**Set a token first.** Without it, `/analyze` is open to anyone who finds the URL:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Keep that string — it goes on the server (`RPF_TOKEN`) and in the extension popup.

---

## 1. Install the CLI and sign in

```bash
winget install Microsoft.AzureCLI
```

Close and reopen the terminal, then:

```bash
az login
```

That opens a browser. Check the right subscription is active:

```bash
az account show --output table
```

---

## 2. Option A — Container Apps (recommended on Azure)

Scales to zero, so you pay only while a request is being served, and **HTTPS is included** —
which the extension needs, because Chrome will not let it call plain `http://` from an
`https://` page.

```bash
az extension add --name containerapp --upgrade
```

```bash
az provider register --namespace Microsoft.App
```

```bash
az group create --name rpf --location centralindia
```

Build the 50 MB data bundle, then deploy from source:

```bash
python deploy/make_bundle.py
```

```bash
az containerapp up --name rpf-backend --resource-group rpf --location centralindia --source . --target-port 8000 --ingress external --env-vars RPF_TOKEN=PASTE_YOUR_TOKEN_HERE
```

The command prints your HTTPS URL, e.g. `https://rpf-backend.<hash>.centralindia.azurecontainerapps.io`.

Check it:

```bash
curl https://YOUR-APP-URL/health
```

You should see `"auth_required": true` and `"features": 41`.

> **Memory note.** Container Apps replicas are disposable, so the learned image memory in
> `out/ext_memory/` is lost when a replica recycles. For a demo that is fine. To keep it, mount
> Azure Files at `/app/fake_review_dataset_v6/out/ext_memory` — or use the VM below, which
> keeps it in a Docker volume.

---

## 3. Option B — a virtual machine

More control, keeps the image memory on disk, but you must add HTTPS yourself.

### Create it

```bash
az group create --name rpf --location centralindia
```

```bash
az vm create --resource-group rpf --name rpf-vm --image Ubuntu2204 --size Standard_B2s --admin-username azureuser --generate-ssh-keys
```

```bash
az vm open-port --resource-group rpf --name rpf-vm --port 8000
```

### Connect

```bash
ssh azureuser@$(az vm show -d -g rpf -n rpf-vm --query publicIps -o tsv)
```

### Install and run (inside the VM)

```bash
sudo apt update && sudo apt install -y docker.io git python3
```

```bash
sudo usermod -aG docker $USER && newgrp docker
```

```bash
git clone https://github.com/Venkatasai200628/fake_review_detection.git && cd fake_review_detection
```

```bash
python3 deploy/make_bundle.py && docker build -f deploy/Dockerfile -t rpf .
```

```bash
docker run -d --name rpf --restart unless-stopped -p 8000:8000 -e RPF_TOKEN=PASTE_YOUR_TOKEN_HERE -v rpf-memory:/app/fake_review_dataset_v6/out/ext_memory rpf
```

The `-v rpf-memory:...` is what keeps the learned image memory across restarts. Without it, every
restart forgets every photo the server has seen, and reuse evidence stops accumulating.

### Check it is alive

```bash
curl http://localhost:8000/health
```

```bash
docker logs -f rpf
```

---

## 4. Seeing your VM, your usage and your credit

List VMs with power state and public IP:

```bash
az vm list -d --resource-group rpf --output table
```

Just the IP:

```bash
az vm show -d -g rpf -n rpf-vm --query publicIps -o tsv
```

Live CPU/disk/network metrics:

```bash
az monitor metrics list --resource $(az vm show -g rpf -n rpf-vm --query id -o tsv) --metric "Percentage CPU" --output table
```

What you have spent:

```bash
az consumption usage list --output table
```

In the portal (<https://portal.azure.com>):

| what you want to see | where |
|---|---|
| the VM, start/stop buttons, its IP | **Virtual machines → rpf-vm** |
| CPU, memory, disk graphs | **rpf-vm → Monitoring → Metrics** |
| **credit remaining** | **Cost Management + Billing → Credits** |
| spending so far, by resource | **Cost Management + Billing → Cost analysis** |
| the Container Apps URL and logs | **Container Apps → rpf-backend** |

---

## 5. Cost control — this is the important part

`Standard_B2s` is about **$0.042/hour**.

| how you run it | per month | how long $100 lasts |
|---|---|---|
| left running 24/7 | ~$30 + ~$6 disk & IP | **~2.8 months** |
| **deallocated when unused** (~20 h/month) | ~$0.85 + ~$6 | **~14 months** |

Stop it when you are not demoing:

```bash
az vm deallocate --resource-group rpf --name rpf-vm
```

Start it again:

```bash
az vm start --resource-group rpf --name rpf-vm
```

> **Shutting down from inside the VM (`sudo poweroff`) still bills you for compute.** Only
> `az vm deallocate` releases it. The managed disk and a static IP are billed even while
> deallocated, which is the ~$6/month floor.

Set a budget alert so you cannot be surprised:

```bash
az consumption budget create --budget-name rpf-budget --amount 20 --time-grain Monthly --category Cost
```

---

## 6. Point the extension at it

1. Chrome → the extension icon → **Backend location**
2. URL: your `https://...azurecontainerapps.io` (or `http://<vm-ip>:8000` — see the HTTPS note)
3. Token: the one you generated in step 0
4. **Save and re-check** → the dot turns green and the footer says a hosted backend is in use

The popup refuses a remote `http://` address on purpose: Chrome blocks it from an `https://`
page, so it would fail on every real product page and look like the server was down.

**HTTPS on a VM.** The VM gives you plain `http://`. To use it from Amazon or Flipkart you need
a domain and a certificate — easiest with Caddy, which gets one automatically:

```bash
sudo apt install -y caddy && sudo caddy reverse-proxy --from yourdomain.com --to localhost:8000
```

Container Apps avoids all of this, which is why it is option A.

---

## 7. Tear it down

Deletes the VM, disk, IP and network in one go:

```bash
az group delete --name rpf --yes --no-wait
```

Confirm nothing is left billing:

```bash
az resource list --output table
```

---

## 8. Honest recommendation

For a viva on your own laptop, hosting buys you nothing: the local backend is faster (4 cores),
needs no token, and cannot be reached by anyone else. Use **Codespaces** (free with your Student
Pack, HTTPS included) if you want it off your machine, and keep the $100 Azure credit for
something that genuinely has to be always-on.

If you do use Azure: **Container Apps**, not a VM, and **deallocate** whenever you are not
demoing.
