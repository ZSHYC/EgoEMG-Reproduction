from pathlib import Path
import datetime,hashlib,json,os,resource,sys
import numpy as np

def sha(path):
 with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def stamp(path):
 path=Path(path);assert path.is_file()and not path.is_symlink()
 s=path.stat();return[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
def actor():
 z=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
 return dict(pid=os.getpid(),start_ticks=int(z[19]))
def live(handle):
 try:z=Path('/proc',str(handle['pid']),'stat').read_text().rsplit(')',1)[1].split();return z[19]==str(handle['start_ticks'])and z[0]not in('Z','X')
 except FileNotFoundError:return False
def decompose(original,truth):
 delta=truth-original;translation=delta.mean(axis=1);centered=delta-translation[:,None,:]
 return translation,np.linalg.norm(delta,axis=2).mean(axis=1),np.linalg.norm(centered,axis=2).mean(axis=1),np.mean(np.sum(delta**2,axis=2),axis=1),np.mean(np.sum(centered**2,axis=2),axis=1)
def rigid_decompose(original,truth):
 x=original-original.mean(axis=1,keepdims=True);y=truth-truth.mean(axis=1,keepdims=True)
 u,_,vh=np.linalg.svd(x.transpose(0,2,1)@y)
 correction=np.broadcast_to(np.eye(3),(len(x),3,3)).copy()
 correction[:,2,2]=np.where(np.linalg.det(u@vh)<0,-1.,1.)
 rotation=u@correction@vh
 delta=x@rotation-y
 return rotation,np.linalg.norm(delta,axis=2).mean(axis=1),np.mean(np.sum(delta**2,axis=2),axis=1)
def similarity_decompose(original,truth,rotation):
 x=original-original.mean(axis=1,keepdims=True);y=truth-truth.mean(axis=1,keepdims=True)
 rotated=x@rotation;scale=np.sum(rotated*y,axis=(1,2))/np.sum(x*x,axis=(1,2))
 assert np.isfinite(scale).all()and(scale>0).all()
 delta=rotated*scale[:,None,None]-y
 return scale,np.linalg.norm(delta,axis=2).mean(axis=1),np.mean(np.sum(delta**2,axis=2),axis=1)

# Known translations and a single-vertex deformation distinguish the hypothesis;
# they are not dataset rows and never enter the actual result population.
toy=np.zeros((2,778,3),dtype=np.float64);shift=np.array([[1.,2.,3.],[-1.,0.,1.]])
t,_,centered,_,_=decompose(toy,toy+shift[:,None,:]);assert np.array_equal(t,shift)and np.array_equal(centered,np.zeros(2))
deformed=toy.copy();deformed[:,0,0]=1.;assert(decompose(toy,deformed)[2]>0).all()
ids=np.arange(778);cloud=np.stack([ids%17,(ids//17)%13,ids//221],axis=1).astype(np.float64)[None]
known=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
rotation,error,_=rigid_decompose(cloud,cloud@known+shift[:1,None,:])
assert np.allclose(rotation[0],known,rtol=0,atol=1e-12)and error[0]<1e-12
mirror=cloud.copy();mirror[:,:,0]*=-1
rotation,error,_=rigid_decompose(cloud,mirror)
assert np.linalg.det(rotation[0])>0 and error[0]>0
rotation,_,_=rigid_decompose(cloud,2.*(cloud@known)+shift[:1,None,:])
scale,error,_=similarity_decompose(cloud,2.*(cloud@known)+shift[:1,None,:],rotation)
assert np.allclose(scale,[2.],rtol=0,atol=1e-12)and error[0]<1e-12

os.nice(10);root=Path(__file__).resolve().parents[1];q=json.loads((root/'private/request.json').read_text())
assert sha(__file__)==q['worker_sha256']and not any(name in sys.modules for name in('torch','chumpy','smplx'))
assert all(not live(h)for h in q['closed_actors'])
available=int(next(line.split()[1]for line in Path('/proc/meminfo').read_text().splitlines()if line.startswith('MemAvailable:')))*1024
assert available>272*1024**3
before={path:stamp(path)for path in q['input_sha256']}
for path,digest in q['input_sha256'].items():assert sha(path)==digest,path
before_payload=Path(q['original_root'])/'private/full'
vertices=np.load(before_payload/'original_vertices.npy',mmap_mode='r',allow_pickle=False)
joints=np.load(before_payload/'original_joints.npy',mmap_mode='r',allow_pickle=False)
saved_errors=np.load(before_payload/'row_errors.npy',mmap_mode='r',allow_pickle=False)
assert vertices.shape==(36520,778,3)and joints.shape==(36520,21,3)and saved_errors.shape==(36520,6)
assert all(a.dtype==np.float64 for a in(vertices,joints,saved_errors))
data={}
for name,identity in q['raw_files'].items():
 with Path(identity['path']).open()as stream:data[name]=np.asarray(json.load(stream))
 assert data[name].dtype==np.float64 and np.isfinite(data[name]).all()
assert data['verts'].shape==(3960,778,3)and data['xyz'].shape==(3960,21,3)
assert data['K'].shape==(3960,3,3)and data['mano'].shape==(3960,1,61)
original_v=vertices[32560:];original_j=joints[32560:]
translation,raw_vertex,centered_vertex,raw_energy,centered_energy=decompose(original_v,data['verts'])
assert np.array_equal(raw_vertex,saved_errors[32560:,0])
raw_joint=np.linalg.norm(data['xyz']-original_j,axis=2).mean(axis=1)
assert np.array_equal(raw_joint,saved_errors[32560:,1])
root_delta=data['xyz'][:,9,:]-original_j[:,9,:]
root_aligned_vertex=np.linalg.norm(data['verts']-original_v-root_delta[:,None,:],axis=2).mean(axis=1)
joint_after_vertex_translation=np.linalg.norm(data['xyz']-original_j-translation[:,None,:],axis=2).mean(axis=1)
rotation,rigid_vertex,rigid_energy=rigid_decompose(original_v,data['verts'])
fitted_scale,similarity_vertex,similarity_energy=similarity_decompose(original_v,data['verts'],rotation)
angle_degrees=np.arccos(np.clip((np.trace(rotation,axis1=1,axis2=2)-1)/2,-1,1))*180/np.pi
focal=(data['K'][:,0,0]+data['K'][:,1,1])/2
mano=data['mano'][:,0,:];assert(mano[:,60]>0).all()
recovered_root=np.stack([mano[:,58]-data['K'][:,0,2],mano[:,59]-data['K'][:,1,2],focal],axis=1)/mano[:,60,None]
columns=['original_vertex_mean_m','centroid_translation_norm_m','centroid_translation_removed_vertex_mean_m',
 'original_vertex_squared_error_mean_m2','centroid_translation_removed_vertex_squared_error_mean_m2',
 'joint9_translation_removed_vertex_mean_m','original_joint_mean_m','vertex_translation_removed_joint_mean_m',
 'joint9_vs_centroid_translation_norm_m','recovered_root_vs_original_joint9_norm_m',
 'rigid_rotation_and_translation_removed_vertex_mean_m','rigid_rotation_and_translation_removed_squared_error_mean_m2','fitted_rotation_angle_degrees',
 'proper_similarity_removed_vertex_mean_m','proper_similarity_removed_squared_error_mean_m2','fitted_positive_uniform_scale']
rows=np.stack([raw_vertex,np.linalg.norm(translation,axis=1),centered_vertex,raw_energy,centered_energy,
 root_aligned_vertex,raw_joint,joint_after_vertex_translation,np.linalg.norm(root_delta-translation,axis=1),
 np.linalg.norm(recovered_root-original_j[:,9,:],axis=1),rigid_vertex,rigid_energy,angle_degrees,
 similarity_vertex,similarity_energy,fitted_scale],axis=1)
assert rows.shape==(3960,16)and np.isfinite(rows).all()
out=root/'private/diagnostic';out.mkdir(mode=0o700)
np.savez(out/'rows.npz',rows=rows,centroid_translation=translation,joint9_translation=root_delta,
 role_row=np.arange(3960,dtype=np.int64))
summary={name:dict(mean=float(rows[:,i].mean()),minimum=float(rows[:,i].min()),maximum=float(rows[:,i].max()),median=float(np.median(rows[:,i])),p95=float(np.quantile(rows[:,i],.95)))for i,name in enumerate(columns)}
summary['translation_explained_squared_error_fraction']=float(1-centered_energy.sum()/raw_energy.sum())
summary['rigid_explained_original_squared_error_fraction']=float(1-rigid_energy.sum()/raw_energy.sum())
summary['rotation_explained_translation_centered_squared_error_fraction']=float(1-rigid_energy.sum()/centered_energy.sum())
summary['fitted_rotation_determinant_minimum']=float(np.linalg.det(rotation).min())
summary['similarity_explained_original_squared_error_fraction']=float(1-similarity_energy.sum()/raw_energy.sum())
summary['scale_explained_rigid_residual_squared_error_fraction']=float(1-similarity_energy.sum()/rigid_energy.sum())
summary['pure_translation_hypothesis_criterion']='diagnostic decomposition only; no tuned pass threshold, corrected GT, new model or replacement metric'
assert all(stamp(path)==old and sha(path)==q['input_sha256'][path]for path,old in before.items())
assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<16*1024**3
assert not any(name in sys.modules for name in('torch','chumpy','smplx'))
value=dict(format='model_free_original_freihand_evaluation_residual_positive_similarity_diagnostic_v1',
 observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),actor=actor(),worker_sha256=q['worker_sha256'],
 original_full_geometry_source_sha256=q['original_worker_sha256'],role='evaluation',all_original_rows=3960,
 columns=columns,summary=summary,payload_sha256=sha(out/'rows.npz'),original_saved_per_row_errors_exact=True,
 all_original_inputs_before_after_sha_and_generation_exact=True,closed_original_actors_rechecked=True,
 actual_gpu_actions=0,new_model_constructions=0,new_optimizer_updates=0,
 input_roles_or_values_changed=False,canonical_outputs_or_scores_replaced=False,full_paper_complete=False,
 peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
with(out/'result.json').open('x')as stream:json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
print(json.dumps(value),flush=True)
