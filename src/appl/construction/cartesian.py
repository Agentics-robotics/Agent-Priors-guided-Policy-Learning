"""Verified Cartesian capability shared by every newly authored policy."""
from pathlib import Path
import ast
import shutil
from appl.io import digest, read
import importlib.util
NAME, POSTURE = 'panda_kinematics.py', 'panda_posture.py'

def require_cartesian(work, expected):
    work = Path(work)
    if digest(work/NAME) != expected:
        raise ValueError('panda_kinematics.py is read-only and must match the verified capability')
    contract = read(work/'policy_contract.json')
    if contract['conversion_check']['mode'] != 'task_space':
        raise ValueError('This study uses the Cartesian action interface: conversion_check.mode must be task_space')
    for path in sorted(work.glob('*.py')):
        if path.name == NAME:
            continue
        tree = ast.parse(path.read_text())
        imported = any((isinstance(n, ast.Import) and any(a.name == 'panda_kinematics' for a in n.names))
                       or (isinstance(n, ast.ImportFrom) and n.module == 'panda_kinematics') for n in ast.walk(tree))
        solved = any((isinstance(n, ast.Attribute) and n.attr == 'solve_ik') or (isinstance(n, ast.Name) and n.id == 'solve_ik')
                     or (isinstance(n, ast.alias) and n.name == 'solve_ik') for n in ast.walk(tree))
        if imported and solved:
            return
    raise ValueError('decode_action must obtain native joint targets through panda_kinematics.solve_ik')

def with_tcp(value, cfg):
    capability = cfg['kinematics']
    if digest(capability['path']) != capability['sha256']:
        raise ValueError('Verified kinematics changed')
    spec = importlib.util.spec_from_file_location('_appl_verified_kinematics', capability['path'])
    kinematics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kinematics)
    robot = read(cfg['public_context'])['robot']
    for row in value['rows']:
        if row.get('action') is not None:
            row['commanded_tcp_pose'] = kinematics.pose_vector(kinematics.forward_kinematics(row['action'][:7], robot)).tolist()
    return value

def extend_read_public(schemas, names):
    for item in schemas:
        if item['name'] == 'read_public':
            enum = item['parameters']['properties']['name']['enum']
            item['parameters']['properties']['name']['enum'] = enum + [n for n in names if n not in enum]
    return schemas


def install(work, cfg):
    for name, record in ((NAME, cfg['kinematics']), (POSTURE, cfg['kinematics']['posture'])):
        target = Path(work)/name
        if not target.exists():
            shutil.copyfile(record['path'], target)
        if digest(target) != record['sha256']:
            raise ValueError(name+' differs from the verified capability')


def require(work, cfg):
    require_cartesian(work, cfg['kinematics']['sha256'])
    if digest(Path(work)/POSTURE) != cfg['kinematics']['posture']['sha256']:
        raise ValueError('panda_posture.py differs from the verified capability')
