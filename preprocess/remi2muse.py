import argparse
import pickle
import glob
import os


def pickle_load(f):
  return pickle.load(open(f, 'rb'))


def pickle_dump(obj, f):
  pickle.dump(obj, open(f, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


def remi2muse(path_infile, path_outfile):
    events = pickle_load(path_infile)
    '''
    for event in events:
        if event["name"] == "Note_Velocity":
            event["value"] = min(max(40,event["value"]),80)
    '''
    bar_idx = []
    for idx, event in enumerate(events):
        if event["name"] == "Bar":
            bar_idx.append(idx)
    result = (bar_idx, events)

    # save
    fn = os.path.basename(path_outfile)
    os.makedirs(path_outfile[:-len(fn)], exist_ok=True)
    pickle_dump(result, path_outfile)


args_parser = argparse.ArgumentParser()
args_parser.add_argument('--input', type=str, default='./jazz-midi/REMI/events')
args_parser.add_argument('--output', type=str, default='./jazz-midi/REMI/muse')
args = args_parser.parse_args()
os.makedirs(args.output, exist_ok=True)

# list files
midifiles = traverse_dir(args.input, extension=('pkl'), is_pure=True, is_sort=True)
n_files = len(midifiles)
print('num files:', n_files)
midifiles = sorted(midifiles)

for fidx in range(n_files):
    path_midi = midifiles[fidx]
    print('{}/{}'.format(fidx, n_files))

    # paths
    path_infile = os.path.join(args.input, path_midi)
    path_outfile = os.path.join(args.output, path_midi)

    remi2muse(path_infile, path_outfile)
