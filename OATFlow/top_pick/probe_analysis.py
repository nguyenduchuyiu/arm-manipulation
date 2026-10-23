"""Summarize E130 saved probe arrays without another model forward pass."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from OATFlow.data_tools.collect_shuffle import BASE
from OATFlow.top_pick.cloud_data import write
from OATFlow.top_pick.features import sha
from OATFlow.top_pick.geometry_probe import OUT, fractions


def main():
    records=json.loads((OUT/'attention_records.json').read_text())
    translation=json.loads((OUT/'translation_records.json').read_text())
    summary=json.loads((OUT/'summary.json').read_text())
    with np.load(OUT/'attention.npz') as z:
        pred=z['predicted_clouds_base_m'];xyz=z['truth_xyz'];offsets=z['truth_offsets'];weights=z['weights'];masks=z['instance']
    support=fractions(masks).numpy()>0;pair_results=[]
    initial=[i for i,r in enumerate(records) if r['frame']==0]
    for pair in range(8):
        a,b=[next(i for i in initial if records[i]['pair']==pair and records[i]['group']==g) for g in ('inside','outside')]
        truth=[xyz[offsets[i]:offsets[i+1]].mean(0) for i in (a,b)]
        centers=pred[[a,b]].mean(1);delta=truth[1]-truth[0];response=centers[1]-centers[0]
        error=(centers[1]-truth[1])*1000;inward=error[0]*(1 if pair<4 else -1)
        pair_results.append(dict(pair=pair,target=records[a]['target'],edge=records[a]['edge'],
            measured_surface_centroid_shift_mm=(delta*1000).tolist(),predicted_centroid_shift_mm=(response*1000).tolist(),
            physical_X_gain=float(response[0]/delta[0]),outside_inward_X_error_mm=float(inward),
            outside_prediction_centroid_world_m=(centers[1]+BASE).tolist()))
    head4=[]
    for i in initial:
        t=records[i]['target'];head4.append(dict(path=records[i]['path'],group=records[i]['group'],
            target_mass=float(weights[i,0,3]@support[i,t]),
            peak_hit=bool(support[i,t,weights[i,0,3].argmax()])))
    jacobians=[]
    for name in sorted({r['path'] for r in translation}):
        columns=[]
        for axis in 'XYZ':
            slopes=[]
            for mm in (5,10,20):
                plus=next(r for r in translation if r['path']==name and r['axis']==axis and r['input_mm']==mm)
                minus=next(r for r in translation if r['path']==name and r['axis']==axis and r['input_mm']==-mm)
                slopes.append((np.array(plus['output_shift_mm'])-minus['output_shift_mm'])/(2*mm))
            columns.append(np.mean(slopes,axis=0))
        jacobians.append(np.array(columns).T)
    matrix=np.median(jacobians,axis=0)
    ten={axis:np.median([r['output_shift_mm'] for r in translation if r['axis']==axis and r['input_mm']==10],axis=0).tolist() for axis in 'XYZ'}
    extras=dict(paired_physical=pair_results,first_layer_head4_initial=head4,
        median_physical_X_gain=float(np.median([r['physical_X_gain'] for r in pair_results])),
        outside_inward_X_cases=sum(r['outside_inward_X_error_mm']>0 for r in pair_results),
        mean_outside_inward_X_error_mm=float(np.mean([r['outside_inward_X_error_mm'] for r in pair_results])),
        median_output_for_plus10mm_input=ten,local_central_difference_median_J=matrix.tolist(),
        J_convention='rows=outputXYZ,columns=inputXYZ;central difference at +/-5,10,20mm,mean within scene then median over16scenes')
    write(OUT/'analysis.json',extras);summary['analysis']=extras;write(OUT/'summary.json',summary)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    im=axes[0].imshow(matrix,cmap='coolwarm',vmin=-1,vmax=1)
    for i in range(3):
        for j in range(3):axes[0].text(j,i,f'{matrix[i,j]:.3f}',ha='center',va='center')
    axes[0].set(xticks=range(3),xticklabels=list('XYZ'),yticks=range(3),yticklabels=list('XYZ'),xlabel='Input XYZ axis',ylabel='Output centroid axis',title='Local response matrix (ideal: identity)')
    fig.colorbar(im,ax=axes[0],label='mm output / mm input')
    x=np.arange(8);axes[1].bar(x,[r['outside_inward_X_error_mm'] for r in pair_results],color=['#2585b7']*4+['#e28737']*4)
    axes[1].set(xlabel='Pair (0-3 x_low; 4-7 x_high)',ylabel='Outside cloud centroid inward X error (mm)',title='Physical pairs: predicted cloud pulled inward')
    fig.savefig(OUT/'geometry_bias.png',dpi=180);plt.close(fig)
    for edge,pairs in [('x_low',range(4)),('x_high',range(4,8))]:
        # Crop unchanged first-layer panels from the original side-by-side sheet.
        source=Image.open(OUT/f'attention_{edge}.png');contact=Image.new('RGB',(800,1480),'white')
        for row,pair in enumerate(pairs):
            for column,original_x in enumerate((0,800)):
                contact.paste(source.crop((original_x,row*370,original_x+400,(row+1)*370)),(column*400,row*370))
        contact.save(OUT/f'target_attention_{edge}.png')
    config=json.loads((OUT/'config.json').read_text());config['analysis_source_sha256']=sha(Path(__file__));write(OUT/'config.json',config)
    print(json.dumps({k:extras[k] for k in ('median_physical_X_gain','outside_inward_X_cases','mean_outside_inward_X_error_mm','median_output_for_plus10mm_input','local_central_difference_median_J')}))


if __name__=='__main__':main()
