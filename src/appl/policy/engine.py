"""Trusted DDPM training/sampling; scientific model and objective come from API."""
from collections import deque
from pathlib import Path
import copy
import importlib.util
import math
import random
import time
import numpy as np
import torch
from diffusers import DDPMScheduler
from appl.io import atomic,read,digest,object_hash,event
from .numerics import save,tensor_hash,deterministic_numerics
from . import data
from . import public
from . import kinematics  # Trusted conversion checks must be loaded before sandboxing.
from .contract import validate_package, files_hash, tree_map, tree_stack, inputs_checked, finite_json


def module_at(source, name='policy'):
    description=importlib.util.spec_from_file_location('api_flexible_'+name,Path(source)/(name+'.py'))
    module=importlib.util.module_from_spec(description);description.loader.exec_module(module)
    for hook in (('build_model','compute_loss') if name=='policy' else ('build_inputs','encode_targets','decode_action')):
        if not callable(getattr(module,hook,None)):raise ValueError('Missing API hook: '+hook)
    return module


def scheduler(config):
    return DDPMScheduler(num_train_timesteps=config['denoising_train_steps'],beta_schedule='squaredcos_cap_v2',
        prediction_type='epsilon',clip_sample=config['clip_sample'])


@torch.no_grad()
def sample(model,inputs,spec,generator,batch_size):
    c=spec['training'];s=scheduler(c)
    s.set_timesteps(c['denoising_inference_steps'],device=generator.device)
    value=torch.randn((batch_size,c['horizon'],spec['contract']['action_dimension']),device=generator.device,generator=generator)
    for t in s.timesteps:
        prediction=model(value,t,inputs)
        if prediction.shape!=value.shape or not torch.isfinite(prediction).all():raise ValueError('Invalid diffusion prediction')
        value=s.step(prediction,t,value,generator=generator).prev_sample
    represented=public.denormalize_representation(value,spec)
    if not torch.isfinite(represented).all():raise ValueError('Invalid diffusion sample')
    return represented


def fit(cfg,folder,source,output,updates,check=False):
    from appl.gpu import verify_cuda
    from .security import lockdown
    folder=Path(folder);source=Path(source);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    entry=read(folder/'assignment.json');metadata=read(source/'pipeline.json')
    config=copy.deepcopy(cfg['training']);config['updates']=updates
    if check:config.update(warmup_steps=0,checkpoint_every=updates)
    hashes={p.name:digest(p) for p in source.iterdir() if p.is_file()}
    contract, bindings, support=validate_package(source,entry)
    documents=data.sources(entry)
    spec=dict(training=config,contract=contract,normalizer=read(entry['normalization']['path'])['normalizer'],
        candidate_config=metadata['config'],public_context=read(cfg['public_context']),skill_id=entry['skill_id'])
    context=dict(contract=contract,binding_support=support)
    request=dict(config=config,entry=entry,source_hashes=hashes,
        data=entry['sources'],seed=0,framework_source=cfg['framework_sha256'],interface_check=check)
    if context:request['policy_context']=context
    if (output/'result.json').exists():
        result=read(output/'result.json')
        if read(output/'request.json')!=request or digest(output/'last.pt')!=result['checkpoint_sha256']:
            raise ValueError('Completed training differs from requested operation')
        return result
    if (output/'request.json').exists() and read(output/'request.json')!=request:raise ValueError('Exact training resume mismatch')
    atomic(output/'request.json',request)
    device=verify_cuda();deterministic_numerics();torch.set_num_threads(2)
    lockdown(source,output)
    torch.manual_seed(0);torch.cuda.manual_seed_all(0);np.random.seed(0);random.seed(0)
    adapters=module_at(source,'adapters')
    arrays,examples,checks=data.prepare(adapters,documents,bindings,spec)
    checks.update(check_conversion(adapters,examples,spec))
    atomic(output/'adapter_checks.json',checks);atomic(output/'spec.json',spec)
    arrays=tree_map(lambda x:x.to('cuda:0'),arrays)
    module=module_at(source);model=module.build_model(spec).cuda()
    count=sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not 0<count<=64000000:raise ValueError('Parameter budget: 1..64 million')
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    initial_hash=tensor_hash(initial)
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    generator=torch.Generator(device='cuda:0').manual_seed(0)
    noise_schedule=scheduler(config);start=0;elapsed=0.;exposures=0
    if (output/'last.pt').exists():
        c=torch.load(output/'last.pt',map_location='cpu',weights_only=True)
        if c['request_hash']!=object_hash(request):raise ValueError('Checkpoint/request mismatch')
        model.load_state_dict(c['model']);ema.load_state_dict(c['ema']);optimizer.load_state_dict(c['optimizer'])
        generator.set_state(c['generator']);torch.set_rng_state(c['torch_rng']);torch.cuda.set_rng_state_all(c['cuda_rng'])
        random.setstate(c['python_rng']);r=c['numpy_rng'];np.random.set_state((r[0],np.asarray(r[1],dtype=np.uint32),*r[2:]))
        start=c['step'];elapsed=c['elapsed_seconds'];initial=c['initial_weights'];initial_hash=c['initial_weights_sha256'];exposures=c['effective_action_targets']
        event(output/'events.jsonl','exact_resume',step=start)
    model.train();began=time.monotonic();losses=[];gradients=[];components={}
    for step in range(start+1,updates+1):
        warmup=config['warmup_steps']
        factor=step/warmup if step<=warmup else .5*(1+math.cos(math.pi*(step-warmup)/(updates-warmup)))
        for group in optimizer.param_groups:group['lr']=config['learning_rate']*factor
        index=torch.randint(len(arrays['encoded_action']),(config['batch_size'],),device=generator.device,generator=generator)
        batch=tree_map(lambda x:x[index],arrays);exposures+=int(batch['mask'].sum())
        encoded=batch['encoded_action']
        noise=torch.randn(encoded.shape,device=generator.device,generator=generator)
        t=torch.randint(config['denoising_train_steps'],(len(index),),device=generator.device,generator=generator)
        batch.update(encoded_action=encoded,noise=noise,timesteps=t,
            noisy_action=noise_schedule.add_noise(encoded,noise,t),
            alpha_bar=noise_schedule.alphas_cumprod.to('cuda:0')[t])
        values=module.compute_loss(model,batch,spec)
        if not {'loss','diffusion_loss','prior_loss'}<=set(values):raise ValueError('Loss must report total, diffusion and prior terms')
        loss=values['loss']
        for key in ('loss','diffusion_loss','prior_loss'):
            if values[key].ndim!=0 or not torch.isfinite(values[key]):raise ValueError('Nonfinite or nonscalar '+key)
        if not loss.requires_grad or not values['diffusion_loss'].requires_grad:raise ValueError('Detached training objective')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['max_grad_norm'],error_if_nonfinite=True)
        if step<=3 or step==updates:
            if not float(norm)>0:raise ValueError('No effective neural gradient')
            gradients.append(dict(step=step,l2_norm=float(norm)))
        optimizer.step()
        with torch.no_grad():
            for p,q in zip(ema.parameters(),model.parameters(),strict=True):p.lerp_(q,1-config['ema_decay'])
            for p,q in zip(ema.buffers(),model.buffers(),strict=True):p.copy_(q)
        losses.append(float(loss.detach()))
        components={k:float(values[k].detach()) for k in ('diffusion_loss','prior_loss')}
        if step%100==0 or step==1 or step==updates:
            progress=dict(step=step,updates=updates,loss_mean=float(np.mean(losses)),components=components,
                elapsed_seconds=elapsed+time.monotonic()-began,effective_action_targets=exposures)
            atomic(output/'progress.json',progress);event(output/'learning.jsonl','progress',**progress);losses.clear()
        if step%config['checkpoint_every']==0 or step==updates:
            r=np.random.get_state()
            save(output/'last.pt',dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=optimizer.state_dict(),
                generator=generator.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                python_rng=random.getstate(),numpy_rng=[r[0],r[1].tolist(),*r[2:]],step=step,spec=spec,
                policy_context=context,
                request_hash=object_hash(request),elapsed_seconds=elapsed+time.monotonic()-began,
                initial_weights=initial,initial_weights_sha256=initial_hash,effective_action_targets=exposures))
    delta=sum((v.detach().cpu()-initial[k]).double().square().sum().item() for k,v in model.state_dict().items())**.5
    if delta<=0:raise ValueError('Weights did not change')
    ema.eval();raw=tree_map(lambda x:x[:2],arrays['model_inputs'])
    predicted=sample(ema,raw,spec,torch.Generator(device='cuda:0').manual_seed(913),2)
    reloaded=module.build_model(spec).cuda();reloaded.load_state_dict(torch.load(output/'last.pt',map_location='cpu',weights_only=True)['ema']);reloaded.eval().requires_grad_(False)
    replay=sample(reloaded,raw,spec,torch.Generator(device='cuda:0').manual_seed(913),2)
    error=float((predicted-replay).abs().max())
    if error>1e-6:raise ValueError('EMA reload mismatch')
    for i in range(2):
        e=examples[i]
        for slot in range(1,9):
            decoded_checked(adapters,predicted[i,slot].cpu().tolist(),e['history'][-1],e['chunk_context'],e['call_args'],spec)
    result=dict(neural_training_verified=True,optimizer_steps=updates,trainable_parameters=count,
        weight_l2_change=delta,initial_weights_sha256=initial_hash,final_weights_sha256=tensor_hash(model.state_dict()),
        finite_nonzero_gradient_checks=gradients,reload_max_action_error=error,device=device,
        checkpoint=str(output/'last.pt'),checkpoint_sha256=digest(output/'last.pt'),
        training_elapsed_seconds=elapsed+time.monotonic()-began,source_hashes=hashes,
        sample_exposures=updates*config['batch_size'],effective_action_targets=exposures,
        interface_check=check,checkpoint_selection='last EMA at the declared update budget')
    if context:result['policy_context']=context
    atomic(output/'result.json',result);return result


class LoadedPolicy:
    def __init__(self,source,checkpoint,seed,device="cuda:0"):
        self.device = torch.device(device)
        deterministic_numerics()
        c=torch.load(checkpoint,map_location='cpu',weights_only=True);self.spec=c['spec']
        module=module_at(source);self.model=module.build_model(self.spec).to(self.device)
        self.model.load_state_dict(c['ema']);self.model.eval().requires_grad_(False)
        self.adapters=module_at(source,'adapters')
        self.generator=torch.Generator(device=self.device).manual_seed(seed)
        self.queue=deque();self.binding=None;self.chunk_context=None
        self.timings=[]

    def reset(self):self.queue.clear();self.binding=None;self.chunk_context=None

    def action(self,history,call_args,reset=False):
        import jsonschema
        jsonschema.validate(call_args,self.spec['contract']['call_args_schema'])
        binding=object_hash(call_args)
        if reset or binding!=self.binding:self.reset()
        self.binding=binding
        if not self.queue:
            started=time.monotonic()
            built=self.adapters.build_inputs(history,call_args,self.spec['public_context'],self.spec)
            inputs=inputs_checked(built['model_inputs'],self.spec['contract']['input_shapes'])
            self.chunk_context=finite_json(built['chunk_context'])
            value=sample(self.model,tree_map(lambda x:x[None].to(self.device),inputs),self.spec,self.generator,1)[0].cpu().numpy()
            self.timings.append(time.monotonic()-started)
            start=self.spec['training']['observation_steps']-1
            self.queue.extend(value[start:start+self.spec['training']['execution_steps']].tolist())
        encoded=self.queue.popleft()
        result=decoded_checked(self.adapters,encoded,history[-1],self.chunk_context,call_args,self.spec)
        return dict(**result,represented_action=encoded,chunk_context=self.chunk_context,binding_hash=binding)


def decoded_checked(adapters,slot,state,chunk,args,spec):
    value=adapters.decode_action(slot,copy.deepcopy(state),copy.deepcopy(chunk),copy.deepcopy(args),spec['public_context'],spec)
    if set(value)!={'native_action','diagnostics'}:raise ValueError('decode_action returns native_action and diagnostics')
    action=np.asarray(value['native_action'],dtype=np.float64)
    if action.shape!=(8,) or not np.isfinite(action).all():raise ValueError('Decoded action must be finite [8]')
    return dict(native_action=action.tolist(),diagnostics=finite_json(value['diagnostics']))


def check_conversion(adapters,examples,spec):
    config=spec['contract']['conversion_check'];residuals=[]
    for e in examples:
        for slot in range(16):
            if not e['window']['mask'][slot][0]:continue
            decoded=decoded_checked(adapters,e['actions'][slot].tolist(),e['window']['action_observations'][slot],
                e['chunk_context'],e['call_args'],spec)['native_action']
            native=e['window']['native_action'][slot]
            if config['mode']=='native_roundtrip':
                error=float(np.max(np.abs(np.asarray(decoded)-native)))
            else:
                from .kinematics import residual
                error=residual(decoded,native,spec['public_context']['robot'])
            residuals.append(error)
    worst=max(residuals)
    if worst>config['tolerance']:raise ValueError('Encode/decode consistency failed: '+str(worst)+' > '+str(config['tolerance']))
    return dict(conversion_mode=config['mode'],conversion_probes=len(residuals),conversion_max_residual=worst)
