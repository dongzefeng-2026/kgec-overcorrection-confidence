"""模型分段并行下载，下载后按源站公布的 sha256 校验。

--src hf：HuggingFace（走 AutoDL 学术加速）；--src ms：ModelScope（国内直连，不走代理）。
原 14 个配置的权重来自 ModelScope（run_all.sh 默认 SRC=ms），新增运行默认同源。

用法: python pdl.py <repo_id> <out_dir> [--src ms|hf] [--files f1 ...] [--seg 16] [--jobs 48]
- 小文件直接下；LFS 大文件按字节区间切成 --seg 段，所有段共用一个 --jobs 线程池；
- 每段失败自动重试（断点续传）；拼接后核对 sha256，不一致则删除并报错；
- 已存在且 sha256 一致的文件跳过。
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request

PROXY_ENV = "/etc/network_turbo"


def proxy_env():
    env = dict(os.environ)
    if os.path.exists(PROXY_ENV):
        out = subprocess.run(["bash", "-c", f"source {PROXY_ENV} >/dev/null 2>&1; env"],
                             capture_output=True, text=True).stdout
        for line in out.splitlines():
            k, _, v = line.partition("=")
            if k.lower() in ("http_proxy", "https_proxy", "no_proxy"):
                env[k] = v
    return env


ENV = proxy_env()


# 只取推理需要的文件；同时存在 safetensors 时跳过 .bin/.pt/.gguf 等冗余权重
KEEP_EXT = (".json", ".safetensors", ".py", ".txt", ".model", ".tiktoken", ".jinja")


def tree(repo, src):
    """统一成 [{path, size, sha256(None=小文件不校验)}]。"""
    if src == "hf":
        url = f"https://huggingface.co/api/models/{repo}/tree/main?recursive=1"
        out = subprocess.run(["curl", "-s", "--max-time", "60", url], capture_output=True,
                             text=True, env=ENV).stdout
        items = [{"path": x["path"], "size": x["size"],
                  "sha256": (x.get("lfs") or {}).get("oid")}
                 for x in json.loads(out) if x["type"] == "file"]
    else:
        url = (f"https://www.modelscope.cn/api/v1/models/{repo}/repo/files"
               "?Revision=master&Recursive=true")
        out = subprocess.run(["curl", "-s", "--max-time", "60", url], capture_output=True,
                             text=True).stdout
        items = [{"path": x["Path"], "size": x["Size"],
                  "sha256": x.get("Sha256") if x["Size"] > (64 << 20) else None}
                 for x in json.loads(out)["Data"]["Files"] if x["Type"] == "blob"]
    items = [x for x in items if x["path"].endswith(KEEP_EXT) and "/" not in x["path"]]
    return items


def file_url(repo, src, path):
    if src == "hf":
        return f"https://huggingface.co/{repo}/resolve/main/{path}"
    return f"https://www.modelscope.cn/models/{repo}/resolve/master/{path}"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def fetch_range(url, part, lo, hi, env):
    want = hi - lo + 1
    for attempt in range(30):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if have == want:
            return
        if have > want:
            os.remove(part); have = 0
        r = subprocess.run(["curl", "-sSL", "--max-time", "900", "-r", f"{lo + have}-{hi}",
                            "-o", "-", url], capture_output=True, env=env)
        with open(part, "ab") as f:
            f.write(r.stdout[: want - have])
        time.sleep(min(2 * attempt, 20) if r.returncode else 0)
    raise RuntimeError(f"段下载失败 {part}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("out")
    ap.add_argument("--src", choices=("ms", "hf"), default="ms")
    ap.add_argument("--files", nargs="*")
    ap.add_argument("--seg", type=int, default=16)
    ap.add_argument("--jobs", type=int, default=48)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    env = ENV if a.src == "hf" else {k: v for k, v in os.environ.items()
                                     if "proxy" not in k.lower()}
    items = tree(a.repo, a.src)
    if a.files:
        items = [x for x in items if x["path"] in a.files]
    # --src hf 时，若 ModelScope 上同名文件的 sha256 与 HF 相同，则分段交替从两个源下载，
    # 叠加两边的带宽（两边分别限速）；不一致或取不到时只用 HF。
    mirror = {}
    if a.src == "hf":
        try:
            ms_sha = {x["path"]: x["sha256"] for x in tree(a.repo, "ms") if x["sha256"]}
            mirror = {x["path"]: file_url(a.repo, "ms", x["path"]) for x in items
                      if x["sha256"] and ms_sha.get(x["path"]) == x["sha256"]}
        except Exception as ex:          # noqa: BLE001
            print("[pdl] ModelScope 镜像不可用，只用 HF:", ex)
    env_ms = {k: v for k, v in os.environ.items() if "proxy" not in k.lower()}
    tasks, big = [], []
    for x in items:
        dst = os.path.join(a.out, x["path"])
        url = file_url(a.repo, a.src, x["path"])
        if x["sha256"] and os.path.exists(dst) and os.path.getsize(dst) == x["size"] \
                and sha256(dst) == x["sha256"]:
            print("ok(skip)", x["path"]); continue
        if not x["sha256"]:
            if not os.path.exists(dst) or os.path.getsize(dst) != x["size"]:
                subprocess.run(["curl", "-sSL", "--retry", "10", "-o", dst, url], env=env)
            continue
        if os.path.exists(dst):
            os.remove(dst)
        n = max(1, min(a.seg, x["size"] // (64 << 20)))
        step = -(-x["size"] // n)
        parts = []
        for k in range(n):
            lo, hi = k * step, min(x["size"], (k + 1) * step) - 1
            part = f"{dst}.part{k:03d}"
            parts.append(part)
            if x["path"] in mirror and k % 2 == 1:
                tasks.append((mirror[x["path"]], part, lo, hi, env_ms))
            else:
                tasks.append((url, part, lo, hi, env))
        big.append((x, dst, parts))
    t0 = time.time()
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(lambda t: fetch_range(*t), tasks))
    bad = 0
    for x, dst, parts in big:
        with open(dst, "wb") as out:
            for p in parts:
                with open(p, "rb") as f:
                    while True:
                        b = f.read(1 << 24)
                        if not b:
                            break
                        out.write(b)
                os.remove(p)
        ok = sha256(dst) == x["sha256"]
        print(("ok  " if ok else "BAD ") + x["path"], x["size"], flush=True)
        if not ok:
            os.remove(dst); bad += 1
    gb = sum(x["size"] for x, _, _ in big) / 1e9
    print(f"[pdl] {a.repo}: {len(big)} 个大文件 {gb:.1f}GB, {time.time() - t0:.0f}s, 校验失败 {bad}")
    if not bad:
        # 完成标记：只有全部大文件 sha256 通过才写，下游据此判断目录可用
        open(os.path.join(a.out, ".pdl_complete"), "w").write(a.repo + "\n")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
