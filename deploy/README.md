# Hosting the backend

The backend normally runs on your laptop at `127.0.0.1:8000` and nothing else can reach it.
Hosting it means anyone with the address can reach it, so **read section 1 before you make a
port public**. Everything here works the same on Codespaces, Azure Container Apps, Azure App
Service or a plain VM.

Measured requirements (not estimates — `Get-Process` on the running server):

| | |
|---|---|
| RAM | **1.1 GB resident, 2.3 GB committed** → give it 2 GB minimum, 4 GB comfortable |
| disk | ~50 MB data + ~24 MB model + ~1.5 GB CPU-only torch + ~800 MB pretrained encoders |
| CPU | 1–3 s per photo on 4 cores; expect 2–5 s on 1–2 cores |

**Azure's free B1s (1 vCPU, 1 GiB) will not work** — the OOM killer takes it on the first photo.

---

## 1. Two things to set before going public

**A token.** Without `RPF_TOKEN`, `/analyze` is open to anyone who finds the URL. With it, every
request needs `Authorization: Bearer <token>`; `/health` stays open so the extension can show
whether the server is up.

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"   # generate one
```

**The image allowlist, which is already on.** `/analyze` takes image URLs and fetches them. On a
laptop that is harmless; on a public host it is a server-side request forgery hole, and the usual
target is the cloud metadata endpoint (`169.254.169.254` on both Azure and AWS) which can hand
out credentials. `Pipeline.check_image_url` refuses anything that is not on
`config.ALLOWED_IMAGE_HOSTS`, **and** refuses any host that resolves to a private, loopback or
link-local address, because a DNS name can point anywhere. Tested:

```
  ALLOWED  https://m.media-amazon.com/...          allowed marketplace
  refused  http://169.254.169.254/metadata/...     image host not allowed
  refused  http://127.0.0.1:8000/health            image host not allowed
  refused  http://localhost/x.jpg                  resolves to a non-public address: ::1
  refused  file:///etc/passwd                      image URL must be http(s)
```

Add a marketplace to `ALLOWED_IMAGE_HOSTS` in `backend/config.py` when you add it to `sites.js`.

---

## 2. GitHub Codespaces — free with the Student Pack

The Student Pack includes GitHub Pro: **180 core-hours/month**, so a 2-core machine (8 GB RAM,
six times what we need) runs about **90 hours a month at no cost**.

1. On GitHub: **Code → Codespaces → Create codespace on main**. `.devcontainer/devcontainer.json`
   asks for 2 cores and installs the CPU-only wheels.
2. In the Codespace terminal:

```bash
export RPF_TOKEN=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
echo "token: $RPF_TOKEN"        # paste this into the extension popup
cd backend && python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

3. **Ports** tab → port 8000 → right-click → **Port Visibility → Public**. You get an HTTPS URL
   like `https://<name>-8000.app.github.dev`.
4. Extension popup → **Backend location** → paste that URL and the token → **Save and re-check**.

**It stops after 30 minutes idle.** Before a demo, open the Codespace and load
`https://<name>-8000.app.github.dev/health` once to wake it: the first request also loads the
models, which takes 30–60 s.

---

## 3. Azure

Build the bundle and image first:

```bash
python deploy/make_bundle.py          # ~50 MB into deploy/bundle/
docker build -f deploy/Dockerfile -t rpf-backend .
```

### Container Apps — cheapest, scales to zero

You pay only while a request is running, which suits a demo used a few times a day.

```bash
az containerapp up \
  --name rpf-backend --resource-group rpf --location centralindia \
  --source . --target-port 8000 --ingress external \
  --env-vars RPF_TOKEN=<your-token>
```

### A VM

`B2s` (2 vCPU, 4 GiB) is the realistic minimum at roughly $0.042/hour.

| how you run it | monthly | how long $100 lasts |
|---|---|---|
| left on 24/7 | ~$30 + ~$6 disk & IP | **~2.8 months** |
| **deallocated when unused** (~20 h/month) | ~$0.85 + ~$6 | **~14 months** |

Disk and a static IP are billed even while the VM is stopped, which is the ~$6 floor. Stop it
with `az vm deallocate` — merely shutting down from inside the VM still bills compute.

A VM also needs HTTPS in front of it (Caddy or nginx with Let's Encrypt), because Chrome will not
let the extension call plain `http://` from an `https://` page. Container Apps and App Service
give you HTTPS for free, which is why they are listed first.

---

## 4. Which to choose

For a student project: **Codespaces**. It is free with your Student Pack, gives HTTPS, has 8 GB
of RAM, and the repository is already there. Keep the $100 Azure credit for something that has to
be always-on — and if you do use a VM, deallocate it between sessions and the credit lasts a year.

Honestly, though: for a viva on your own laptop, hosting buys you nothing. The local backend is
faster (4 cores), needs no token, and cannot be reached by anyone else.
