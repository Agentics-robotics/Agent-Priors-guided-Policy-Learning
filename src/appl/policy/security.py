"""Credential-free Torch/CUDA workers: fixed capabilities plus kernel isolation."""
import ast
import os
from pathlib import Path
import resource
import sys
import time
from appl.io import atomic,ROOT
from appl.kernel import _landlock,_seccomp


def audit(candidate):
    candidate=Path(candidate)
    local={p.stem for p in candidate.glob('*.py')}
    forbidden={'eval','exec','compile','open','input','globals','locals','vars','getattr','setattr','delattr','__import__','breakpoint'}
    for path in candidate.rglob('*.py'):
        if path.is_symlink():raise ValueError('Candidate symlinks are forbidden')
        tree=ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                names=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module or '']
                if isinstance(node,ast.ImportFrom) and node.level:raise ValueError('Use explicit own-module imports')
                for name in names:
                    if not (name.split('.')[0] in {'torch','numpy','math'}|local or name in {'appl.public', 'appl.policy.public', 'experiments.exp2.sol_flexible.public', 'experiments.exp2.six_tasks.flexible.public'}):
                        raise ValueError('Import outside numerical/public capabilities: '+name)
            if isinstance(node,ast.Name) and node.id in forbidden:raise ValueError('Dynamic IO/reflection is not a policy capability')
            if isinstance(node,ast.Attribute) and node.attr.startswith('_') and node.attr not in {'__init__'}:
                raise ValueError('Private attribute reflection is forbidden')
            if isinstance(node,(ast.Global,ast.Nonlocal)):raise ValueError('Global mutation is forbidden')


def lockdown(candidate,output,readonly=(),gpu=True):
    import torch
    from appl.gpu import verify_cuda
    candidate=Path(candidate).resolve();output=Path(output).resolve()
    audit(candidate)
    identity=verify_cuda() if gpu else None
    # Initialize CUDA math/optimizer libraries before installing thread-synced
    # restrictions; no generated source is loaded until after both restrictions.
    warmup_started=time.monotonic()
    if gpu:
        layer=torch.nn.Conv1d(8,16,3).cuda();opt=torch.optim.AdamW(layer.parameters())
        value=layer(torch.ones(2,8,16,device='cuda')).square().mean();value.backward();opt.step()
        torch.cuda.synchronize();del layer,opt,value
    else:
        # CPU kernels discover topology on first use. Initialize trusted Torch
        # before restricting filesystem access, just as the CUDA branch does.
        with torch.no_grad():
            layer=torch.nn.Conv1d(8,16,3)
            layer(torch.ones(2,8,16))
            del layer
    device_paths=[] if not gpu else [Path('/dev')/n for n in (
        f"nvidia{identity['minor']}",'nvidiactl','nvidia-uvm','nvidia-uvm-tools') if (Path('/dev')/n).exists()]
    runtime=[Path(sys.prefix),Path('/usr/lib'),Path('/usr/lib64'),Path('/etc/ld.so.cache'),
        Path('/dev/null'),Path('/dev/zero'),Path('/dev/urandom'),Path('/dev/random'),
        ROOT/'src/appl/public.py',ROOT/'src/appl/vendor',ROOT/'src/appl/policy/public.py',candidate,*map(Path,readonly)]
    environment={k:os.environ[k] for k in ('APPL_GPU_UUID','APPL_GPU_MINOR','APPL_PHYSICAL_GPU','CUDA_VISIBLE_DEVICES') if k in os.environ}
    os.environ.clear();os.environ.update(environment,PATH='/usr/bin:/bin',LANG='C.UTF-8',
        PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',
        OPENBLAS_NUM_THREADS='1',CUDA_CACHE_DISABLE='1',TMPDIR=str(output),XDG_CACHE_HOME=str(output/'cache'))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_CPU,(7200,7200))
    abi=_landlock([p.resolve() for p in runtime if p.exists()],[output],device_paths)
    _seccomp()
    receipt=dict(landlock_abi=abi,seccomp=True,network=False,credentials=False,
        source_imports_after_lockdown=True,device=identity,write_dirs=[str(output)],
        readonly_grants=[str(p.resolve()) for p in runtime if p.exists()],
        trusted_warmup_optimizer_steps=1 if gpu else 0,trusted_warmup_is_candidate_training=False,
        initialization_and_lockdown_seconds=time.monotonic()-warmup_started)
    atomic(output/'enforcement.json',receipt)
    sys.path.insert(0,str(candidate))
    return receipt
