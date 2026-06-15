"""Procedural ReAct transcripts for COMPLETE TOOL + COMPUTER-USE MASTERY.
Teaches the model to drive a full dev environment like a human SWE: bash, git, docker, package
managers (pip/npm/cargo/go/uv/apt), build/test (make/pytest/etc), file editing (sed/heredoc),
processes/env (ps/kill/env/curl/chmod), debugging, AND multi-step end-to-end workflows
(write -> run -> read error -> fix -> test -> commit). Harness `$ ` protocol with realistic
tool output. Output: /tmp/devtools/devtools.txt
"""
import os, random

OUT = "/tmp/devtools/devtools.txt"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
SYS = ("You are a coding agent with full computer access. Run shell commands by writing a line "
       "starting with '$ '. Master bash, git, docker, package managers, build/test tools, "
       "editors, and processes to accomplish software engineering tasks.")
FILES = ["app.py", "main.py", "server.js", "utils.py", "index.ts", "api.go", "lib.rs"]
BR = ["feature", "fix-bug", "refactor", "dev", "hotfix", "add-tests"]
IMG = ["app", "web", "api", "worker", "ml-job"]
MSG = ["initial commit", "add feature", "fix bug", "update deps", "add tests"]
PKG = ["requests", "numpy", "flask", "pytest", "express", "axios", "serde", "tokio"]


def hx(rng, n=7): return "".join(rng.choice("0123456789abcdef") for _ in range(n))
def t(task, *steps):
    b = f"{SYS}\nTask: {task}\n"
    for c, o in steps:
        b += f"$ {c}\n" + (o + "\n" if o else "")
    return b + "\n"


def git_tx(rng):
    f, br, m, h = rng.choice(FILES), rng.choice(BR), rng.choice(MSG), hx(rng)
    return rng.choice([
        lambda: t(f"init a git repo and commit {f}", ("git init", "Initialized empty Git repository in /work/.git/"), ("git add "+f, ""), (f'git commit -m "{m}"', f"[main (root-commit) {h}] {m}\n 1 file changed, 12 insertions(+)")),
        lambda: t(f"create and switch to branch {br}", (f"git checkout -b {br}", f"Switched to a new branch '{br}'")),
        lambda: t("show working tree status", ("git status", f"On branch {br}\n  modified:   {f}\nno changes added to commit")),
        lambda: t(f"merge {br} into main", ("git checkout main", "Switched to branch 'main'"), (f"git merge {br}", f"Fast-forward\n {f} | 8 +++++---\n 1 file changed")),
        lambda: t(f"push branch {br}", (f"git push -u origin {br}", f"To github.com:u/repo.git\n * [new branch]  {br} -> {br}")),
    ])()


def docker_tx(rng):
    img, h = rng.choice(IMG), hx(rng, 12)
    return rng.choice([
        lambda: t(f"write a Dockerfile and build image {img}", ("cat > Dockerfile << 'EOF'\nFROM python:3.12-slim\nWORKDIR /app\nCOPY . .\nRUN pip install -r requirements.txt\nCMD [\"python\",\"app.py\"]\nEOF", ""), (f"docker build -t {img} .", f"Successfully built {h}\nSuccessfully tagged {img}:latest")),
        lambda: t(f"run {img} in background on 8080", (f"docker run -d -p 8080:8080 {img}", h)),
        lambda: t("list running containers", ("docker ps", f"CONTAINER ID   IMAGE   STATUS\n{h}   {img}   Up 2 minutes")),
        lambda: t(f"view logs of {h}", (f"docker logs {h}", "Server started on :8080")),
    ])()


def pkg_tx(rng):
    p = rng.choice(PKG)
    return rng.choice([
        lambda: t(f"install the {p} python package", (f"pip install {p}", f"Collecting {p}\nSuccessfully installed {p}-2.31.0")),
        lambda: t(f"add {p} to a node project", (f"npm install {p}", f"added 1 package in 1s")),
        lambda: t("install project dependencies", ("pip install -r requirements.txt", "Successfully installed flask-3.0 requests-2.31")),
        lambda: t(f"add the {p} crate", (f"cargo add {p}", f"      Adding {p} v1.0 to dependencies")),
        lambda: t(f"get the {p} go module", (f"go get {p}", f"go: added {p} v1.2.0")),
        lambda: t("create a venv and install deps", ("python -m venv .venv && . .venv/bin/activate", ""), ("pip install -r requirements.txt", "Successfully installed numpy-1.26")),
    ])()


def buildtest_tx(rng):
    n = rng.randint(3, 12)
    return rng.choice([
        lambda: t("run the python test suite", ("python -m pytest -q", f"{'.'*n}\n{n} passed in 0.{rng.randint(10,90)}s")),
        lambda: t("build the project with make", ("make build", "gcc -O2 -c main.c\ngcc -o app main.o\nBuild complete")),
        lambda: t("run node tests", ("npm test", f"Test Suites: 1 passed\nTests: {n} passed")),
        lambda: t("build and test the rust crate", ("cargo build", "   Compiling app v0.1.0\n    Finished dev [unoptimized] target(s)"), ("cargo test", f"test result: ok. {n} passed; 0 failed")),
        lambda: t("run go tests", ("go test ./...", f"ok  \tmodule/pkg\t0.0{rng.randint(1,9)}s")),
    ])()


def edit_tx(rng):
    f = rng.choice(FILES)
    return rng.choice([
        lambda: t(f"append a logging line to {f}", (f"echo 'import logging' >> {f}", ""), (f"tail -1 {f}", "import logging")),
        lambda: t(f"replace 'localhost' with '0.0.0.0' in {f}", (f"sed -i 's/localhost/0.0.0.0/g' {f}", ""), (f'grep 0.0.0.0 {f}', 'host = "0.0.0.0"')),
        lambda: t(f"create {f} with a main function", (f"cat > {f} << 'EOF'\ndef main():\n    print('hello')\nif __name__ == '__main__':\n    main()\nEOF", ""), (f"python {f}", "hello")),
    ])()


def sys_tx(rng):
    return rng.choice([
        lambda: t("find the process on port 8080 and the python version", ("lsof -i :8080", "python  4821 user   3u  IPv4  TCP *:8080 (LISTEN)"), ("python --version", "Python 3.12.3")),
        lambda: t("check an API endpoint with curl", ("curl -s localhost:8080/health", '{"status":"ok"}')),
        lambda: t("make a script executable and run it", ("chmod +x run.sh && ./run.sh", "Starting...\nDone.")),
        lambda: t("set an env var and run the app", ("export PORT=9000 && python app.py", "listening on :9000")),
        lambda: t("show disk usage of the build dir", ("du -sh build/", "128M\tbuild/")),
    ])()


def workflow_tx(rng):
    """Multi-step end-to-end SWE workflow: the heart of computer-use mastery."""
    f, br, h = rng.choice(FILES), rng.choice(BR), hx(rng)
    n = rng.randint(2, 6)
    return rng.choice([
        lambda: t(f"fix the failing test in {f} and commit",
                  ("python -m pytest -q", "F\n1 failed in 0.05s\nE   AssertionError: assert add(2,2)==4"),
                  (f"sed -i 's/return a - b/return a + b/' {f}", ""),
                  ("python -m pytest -q", ".\n1 passed in 0.04s"),
                  ("git add "+f, ""), (f'git commit -m "fix add bug"', f"[{br} {h}] fix add bug\n 1 file changed")),
        lambda: t(f"create a feature branch, add {f}, test, and push",
                  (f"git checkout -b {br}", f"Switched to a new branch '{br}'"),
                  (f"cat > {f} << 'EOF'\ndef f(n):\n    return n*n\nEOF", ""),
                  ("python -c 'import "+f.split('.')[0]+"' 2>/dev/null || python "+f, ""),
                  ("git add "+f+f' && git commit -m "add {f}"', f"[{br} {h}] add {f}\n 1 file changed"),
                  (f"git push -u origin {br}", f" * [new branch]  {br} -> {br}")),
        lambda: t("set up a python project, install deps, run tests in docker",
                  ("python -m venv .venv && . .venv/bin/activate", ""),
                  ("pip install -r requirements.txt", "Successfully installed pytest-8.0"),
                  ("python -m pytest -q", f"{'.'*n}\n{n} passed"),
                  ("docker build -t app .", "Successfully tagged app:latest")),
        lambda: t("debug a crashing script: read the traceback and fix it",
                  ("python app.py", 'Traceback (most recent call last):\n  File "app.py", line 3\n    print(x)\nNameError: name \'x\' is not defined'),
                  ("sed -i '2i x = 42' app.py", ""),
                  ("python app.py", "42")),
    ])()


def main():
    rng = random.Random(0)
    gens = [git_tx, docker_tx, pkg_tx, buildtest_tx, edit_tx, sys_tx,
            workflow_tx, workflow_tx]  # weight multi-step workflows
    with open(OUT, "w") as o:
        for _ in range(140000):
            o.write(rng.choice(gens)(rng))
    print(f"wrote 140000 tool/computer-use transcripts -> {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB)")
    print("--- sample (multi-step) ---\n" + open(OUT).read(700))


if __name__ == "__main__":
    main()
