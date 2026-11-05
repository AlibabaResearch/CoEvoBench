"""Unprivileged GPU worker with persisted deadlines and descendant cleanup."""

WORKER = r"""
import fcntl,json,os,signal,subprocess,sys,time
directory=sys.argv[1]
config=json.load(open(directory+"/config.json"))
env={"PATH":config["bin"]+":/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin","HOME":directory+"/workspace",
     "CUDA_VISIBLE_DEVICES":config["gpus"],"PYTHONUNBUFFERED":"1",
     "VIRTUAL_ENV":config["bin"].rsplit("/",1)[0]}
env.update({k:v for k,v in config.get("teacher_env",{}).items()
            if k in ("TEACHER_BASE_URL","TEACHER_API_KEY","TEACHER_MODEL")})
guard=open(directory+"/lock","a")
fcntl.flock(guard,fcntl.LOCK_EX)
if os.path.exists(directory+"/cancelled"):
    open(directory+"/status.json","w").write(json.dumps({"status":"completed","exit_code":137}))
    sys.exit(0)
if time.time() >= config["deadline"]:
    open(directory+"/status.json","w").write(
        json.dumps({"status":"completed","exit_code":124}))
    sys.exit(0)
with open(directory+"/job.log","w") as log:
    p=subprocess.Popen(["runuser","-u","cebtrain","--preserve-environment","--","bash","-c",config["command"]],
                       cwd=directory+"/workspace",env=env,stdin=subprocess.DEVNULL,stdout=log,
                       stderr=subprocess.STDOUT,start_new_session=True)
    state={"pid":p.pid,"status":"running"}
    temp=directory+"/status.tmp"
    open(temp,"w").write(json.dumps(state))
    os.replace(temp,directory+"/status.json")
    fcntl.flock(guard,fcntl.LOCK_UN)
    try: code=p.wait(timeout=max(1,config["deadline"]-time.time()))
    except subprocess.TimeoutExpired:
        subprocess.run(["pkill","-KILL","-u","cebtrain"],check=False)
        p.wait(timeout=10)
        code=124
    # Daemonized children must not retain GPUs after the submitted command exits.
    subprocess.run(["pkill","-KILL","-u","cebtrain"],check=False)
    state.update(status="completed",exit_code=code)
    temp=directory+"/status.tmp"
    open(temp,"w").write(json.dumps(state))
    os.replace(temp,directory+"/status.json")
"""
