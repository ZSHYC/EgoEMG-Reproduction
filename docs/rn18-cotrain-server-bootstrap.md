# SSH受限时的RN18初始化部署包

服务器此前的新SSH连接在认证前返回`Exceeded MaxStartups`，当时现有控制连接也无法复用。因此准备了[固定版本启动包](../releases/rn18-cotrain-server-ab3e26f-20261101.tar.gz)，模型、配方、数据角色和完整训练预算沿用[RN18初始化方案](rn18-cotrain-initializer.md)。北京时间2026-11-01 15:50连接恢复后，已通过新的可复用SSH连接实际部署并启动，无需用户再次手动执行。

包中268个普通文件只有Python源码和JSON清单，不含数据、模型权重、MANO资产或认证文件。生产源码固定于`ab3e26f4efea63c0ca295199f6529b5037ac2948`，对应98项实际通过检查及一次审查修正；[打包核验记录](../results/rn18-cotrain-server-bootstrap-package-20261101.json)保留打包时尚未部署的历史状态。本机已验证所有文件摘要与源码编译，并确认在缺少服务器环境时会在创建运行状态前拒绝执行。后续[实际服务器核验记录](../results/rn18-cotrain-fixed-source-server-queue-20261101.json)证明修正版两次完整数据重验和独立CPU恢复均已结束，GPU后续队列已部署；GPU资格及150轮仍未完成。

部署后先并行执行修正版的两遍完整输入核验和独立CPU恢复/导出流程，再用新进程检查其原始结果、文件及拥有进程终态。只有这两项成功，才部署GPU0后续队列。该队列等待现有RN18原生视觉完整流程的精确注册进程成功退出、全部拥有者关闭和GPU空闲，然后执行原batch260/12workers/BF16资格及新鲜完整150轮。保留现有任务、数据、输出和原失败记录；已有目标目录会使启动拒绝，而不会自动重试。

下面保留首次部署命令用于复现。当前目标目录及任务已经存在，**不要重复启动**；命令也会拒绝覆盖现有目录。下载后先核对固定SHA256，校验通过才解压和执行；无需提供密码或密钥。

```bash
python3 - <<'PY'
from pathlib import Path
import hashlib, io, subprocess, sys, tarfile, urllib.request

expected = 'ae08a906676af976ba727112353ff3b755f35682ff495a426f20a64b0146f005'
url = 'https://raw.githubusercontent.com/ZSHYC/EgoEMG-Reproduction/refs/heads/codex/full-paper-reproduction/releases/rn18-cotrain-server-ab3e26f-20261101.tar.gz'
destination = Path('/data/shihao/rn18-cotrain-bootstrap-ab3e26f-20261101')
assert not destination.exists(), '请保留现有目录，让我检查已注册任务，避免重复启动'
with urllib.request.urlopen(url, timeout=60) as response:
    blob = response.read()
assert hashlib.sha256(blob).hexdigest() == expected, '下载内容校验失败'
with tarfile.open(fileobj=io.BytesIO(blob), mode='r:gz') as archive:
    members = archive.getmembers()
    assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
    destination.mkdir(mode=0o700)
    archive.extractall(destination)
entry = destination / 'rn18-cotrain-server-ab3e26f' / 'launch.py'
subprocess.run([sys.executable, '-B', str(entry), '--verify'], check=True)
subprocess.run([sys.executable, '-B', str(entry)], check=True)
PY
```

成功输出`bootstrap_started: true`后，启动器和各阶段均使用独立服务器后台会话。状态及日志在解压目录的`bootstrap-state.json`、`bootstrap.log`；三条实际流程仍写各自的`/data/shihao/egoemg-rn18-cotrain-*`目录。`gpu_workflow_queued`只表示后续队列已部署，完整150轮仍须依据实际训练、验证BEST、导出和独立核验结果验收。
