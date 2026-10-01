"""Explicit construction stages; imports and dry-runs do not call the API or GPU."""
import argparse
import json
from pathlib import Path
from appl.io import ROOT, read
from appl.benchmark.registry import TASKS
from .configuration import create, load, skill_item, policy_folder


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='stage', required=True)
    init = sub.add_parser('init', help='Create an isolated run from the 12 published training demonstrations')
    init.add_argument('--task', choices=TASKS, required=True)
    init.add_argument('--output', required=True)
    init.add_argument('--asset-root', default=str(ROOT))
    init.add_argument('--gpu', type=int, default=0)
    init.add_argument('--model', help='Default follows the paper; OPENAI_MODEL can override transport')
    for name in ('inspect', 'cut', 'design', 'train', 'verify', 'refine', 'finalize'):
        stage = sub.add_parser(name)
        stage.add_argument('--run', required=True)
        if name in ('design', 'refine', 'finalize'):
            stage.add_argument('--skill', required=True)
        if name in ('train', 'verify'):
            stage.add_argument('--policy', required=True)
        if name != 'inspect':
            stage.add_argument('--dry-run', action='store_true', help='Validate prerequisites without API or GPU execution')
    args = parser.parse_args(argv)
    if args.stage == 'init':
        result = create(args.task, args.output, args.asset_root, args.gpu, args.model)
        print(json.dumps(dict(run=result['output'], task=result['task_id'], stage='initialized',
                              api_calls=0, gpu_jobs=0), indent=2))
        return
    cfg = load(args.run)
    result = dict(stage=args.stage, task=cfg['task_id'], run=cfg['output'],
                  training_demonstrations=sorted(cfg['sources']), reasoning_effort=cfg['api']['reasoning_effort'],
                  new_run=True, published_result=False, test_feedback_used=False)
    if args.stage == 'inspect':
        result['api_calls'] = result['gpu_jobs'] = 0
        print(json.dumps(result, indent=2))
        return
    from . import pipeline
    if args.stage == 'cut':
        from .cut import CutTools, context
        from .demonstrations import Limits
        # This preflight checks synchronized media and tools without opening an API client.
        tool = CutTools([v['path'] for v in cfg['sources'].values()], cfg['dataset'],
            context=context(cfg),
            limits=Limits(**cfg['segmentation']), provenance=dict(scientific_owner='Runtime API', new_run=True,
            published_result=False), schema_path=cfg['cut_schema'])
        result['tools'] = [s['name'] for s in tool.schemas()]
        tool.j.db.close()
    elif args.stage == 'design':
        item = skill_item(cfg, args.skill)
        from .design import SkillTools
        tool = SkillTools(cfg, item)
        result['tools'] = [s['name'] for s in tool.schemas()]
        tool.close()
    elif args.stage == 'train':
        folder = policy_folder(cfg, args.policy)
        skill = args.policy.rsplit('__h', 1)[0]
        phase = 'refinement' if args.policy.endswith('__h04') else ''
        read(Path(cfg['output'])/phase/'skills'/skill/'submission.json')
        result.update(policy_id=args.policy, updates=cfg['training']['updates'], source=str(folder/'source'))
    elif args.stage == 'verify':
        result['verification'] = pipeline.verification_identity(cfg, args.policy)
    else:
        skill_item(cfg, args.skill)
        count = 3 if args.stage == 'refine' else 4
        identifiers = [f'{args.skill}__h{i:02d}' for i in range(1, count+1)]
        result['id_verification'] = pipeline.validation_view(cfg, identifiers)
    if args.dry_run:
        result.update(api_calls=0, gpu_jobs=0, dry_run=True)
    elif args.stage == 'cut':
        from .cut import run
        result = run(cfg)
    elif args.stage == 'design':
        from .design import run
        result = run(cfg, item)
    elif args.stage == 'train':
        result = pipeline.train(cfg, args.policy)
    elif args.stage == 'verify':
        result = pipeline.verify(cfg, args.policy)
    elif args.stage == 'refine':
        result = pipeline.refine(cfg, args.skill)
    else:
        result = pipeline.finalize(cfg, args.skill)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
