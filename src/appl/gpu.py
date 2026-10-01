"""The sole GPU launcher: UUID/minor verification and a private device tree."""
import argparse
import os
from pathlib import Path
import re
import subprocess
import sys

from .io import PIXI, MANIFEST, ROOT


def identity(index):
    if not isinstance(index, int) or index < 0:
        raise ValueError('GPU index must be a nonnegative integer')
    value = subprocess.check_output(['nvidia-smi', '-i', str(index),
        '--query-gpu=uuid,pci.bus_id,memory.free', '--format=csv,noheader,nounits'], text=True)
    uuid, bus, free = [v.strip() for v in value.split(',')]
    pci = bus.lower().replace('00000000:', '0000:')
    information = (Path('/proc/driver/nvidia/gpus')/pci/'information').read_text()
    assert uuid in information
    minor = int(re.search(r'Device Minor:\s+(\d+)', information).group(1))
    return dict(physical_gpu=index, uuid=uuid, pci=pci, minor=minor, free_mib=int(free))


def launch(index, args, module='appl.cli'):
    ident = identity(index)
    if ident['free_mib'] < 2048:
        raise RuntimeError('Selected GPU has less than 2 GiB free; no alternate device chosen')
    devices = [Path('/dev')/n for n in ('nvidiactl','nvidia-uvm','nvidia-uvm-tools','nvidia-modeset',f"nvidia{ident['minor']}")]
    devices += [(Path('/dev/dri/by-path')/f"pci-{ident['pci']}-{suffix}").resolve() for suffix in ('card','render')]
    cmd = ['bwrap','--die-with-parent','--bind','/','/','--dev','/dev','--proc','/proc',
        '--unsetenv','DISPLAY','--unsetenv','WAYLAND_DISPLAY']
    for key,value in dict(APPL_PHYSICAL_GPU=index, APPL_GPU_UUID=ident['uuid'],
                          APPL_GPU_MINOR=ident['minor'], CUDA_VISIBLE_DEVICES=ident['uuid'],
                          CUDA_DEVICE_ORDER='PCI_BUS_ID', MUJOCO_EGL_DEVICE_ID=0,
                          CUBLAS_WORKSPACE_CONFIG=':4096:8',
                          LD_LIBRARY_PATH=str(Path(sys.prefix)/'lib')).items():
        cmd += ['--setenv',key,str(value)]
    for device in devices:
        assert device.is_char_device(), device
        cmd += ['--dev-bind',str(device),str(device)]
    # Pixi precedes the namespace: no later activation overwrites device binding.
    os.execvp(cmd[0],cmd+['--',sys.executable,'-m',module,*args,'--device-isolated'])


def verify_cuda():
    import torch
    expected = os.environ['APPL_GPU_UUID']
    assert int(os.environ['APPL_PHYSICAL_GPU']) >= 0
    torch.cuda.init()
    torch.zeros(1,device='cuda:0')
    actual = str(torch.cuda.get_device_properties(0).uuid)
    actual = actual if actual.startswith('GPU-') else 'GPU-'+actual
    assert actual == expected, (actual, expected)
    torch.cuda.synchronize()
    return dict(physical_gpu=int(os.environ['APPL_PHYSICAL_GPU']),uuid=actual,
                logical_gpu=0,minor=int(os.environ['APPL_GPU_MINOR']))


def command(args):
    return [PIXI,'run','--manifest-path',str(MANIFEST),'--locked','python','-m','appl.cli',*args]
