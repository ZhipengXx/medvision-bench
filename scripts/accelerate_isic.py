"""Resume a large official S3 archive using bounded concurrent HTTP ranges."""
import concurrent.futures
import os
import signal
import time
from pathlib import Path
import psutil
import requests
ROOT=Path(__file__).resolve().parents[1]
URL='https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task3_Training_Input.zip'
out=ROOT/'data/isic/ISIC2018_Task3_Training_Input.zip'
partial=out.with_suffix('.zip.partial')
if out.exists():raise SystemExit('Archive already complete')
head=requests.head(URL,timeout=30);head.raise_for_status()
size=int(head.headers['Content-Length'])
test=requests.get(URL,headers={'Range':'bytes=0-7'},timeout=30)
if test.status_code!=206:raise SystemExit('Server does not support ranges')
# Stop only this project's downloader, preserving all other users/processes.
for proc in psutil.process_iter(['pid','cmdline','uids']):
    cmd=proc.info['cmdline'] or []
    if proc.info['uids'].real==os.getuid() and len(cmd)>1 and cmd[1]=='-u' and 'scripts/download_data.py' in cmd:
        if str(ROOT) in str(proc.cwd()):
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=20)
offset=partial.stat().st_size if partial.exists() else 0
print('Resuming from',offset,'of',size,flush=True)
directory=ROOT/'data/isic/range_parts';directory.mkdir(exist_ok=True)
def fetch(item):
    i,a,b=item
    dest=directory/f'{i}.part'
    if dest.exists() and dest.stat().st_size==b-a+1:return dest
    for retry in range(3):
        try:
            with requests.get(URL,headers={'Range':f'bytes={a}-{b}'},stream=True,timeout=(30,120)) as r:
                if r.status_code!=206:raise RuntimeError('Expected HTTP 206')
                with dest.open('wb') as f:
                    for chunk in r.iter_content(2**20):f.write(chunk)
            if dest.stat().st_size!=b-a+1:raise RuntimeError('Range length mismatch')
            print('Range complete',i,dest.stat().st_size,flush=True)
            return dest
        except Exception:
            if retry==2:raise
            time.sleep(2)
chunk=(size-offset+7)//8
items=[(i,a,min(a+chunk-1,size-1)) for i,a in enumerate(range(offset,size,chunk))]
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:parts=list(pool.map(fetch,items))
with partial.open('ab') as f:
    for p in parts:
        with p.open('rb') as g:
            for data in iter(lambda:g.read(2**20),b''):f.write(data)
if partial.stat().st_size!=size:raise RuntimeError('Total size mismatch')
partial.replace(out)
for p in parts:p.unlink()
print('Complete',out,size,flush=True)
