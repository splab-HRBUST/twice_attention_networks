"""
Utility script to fuse different output files.
"""

import numpy as np
import pandas as pd
import argparse
'''
   	1) AUDIO_FILE_NAME:	LA_****, name of the audio file
   	2) SYSTEM_ID: 		ID of the speech spoofing system (A01 - A19),  or, for bonafide speech SYSTEM-ID is left blank ('-')
	3) KEY: 		    'bonafide' for genuine speech, or, 'spoof' for spoofing speech
    4) score:           得分。           
'''
def read_frame(fname):
    data_np = np.genfromtxt(fname, dtype=str)
    if data_np.shape[1] == 4:
	    cols = ['fname', 'sysid', 'key', 'score']

    else:
	    cols = ['fname', 'score']
    df = pd.DataFrame(index=data_np[:,0],data=data_np, columns=cols)
    df['score'] = df['score'].astype(np.float32, copy=False)
    return df

def fuse(file_list):
    frames = [read_frame(f) for f in file_list]
    if frames[0].shape[1] == 4:
	    merge_cols = ['fname', 'sysid', 'key']
    else:
	    merge_cols = ['fname']
    result_df = pd.concat(frames).groupby(merge_cols, as_index=False)["score"].mean()
    return result_df
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Result Fusion utility')
    parser.add_argument('--input', type=str, nargs='+', required=True)
    parser.add_argument('--output', type=str,required=True)
    args = parser.parse_args()
    print('Processing input files : ', args.input)
    fuse_result = fuse(args.input)
    fuse_result.to_csv(args.output,  sep=' ', header=False, index=False)
    print('Result saved to {}'.format(args.output))
