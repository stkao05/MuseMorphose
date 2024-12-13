Preprocessing

0. Setup

   `$ pip install -r requirements.txt`

   The output file tree will be the same as the original directory, so tidy up the directory if needed.

1. synchronizer.py 

   --input (root directory of raw midi files)

   --output (root directory of midi_synchronized) 

   --start (index of file to start processing)

   --amount (number of files to process)
   
   synchronizer.py expects 4 beats in a bar, and will produce wrong results otherwise.

   You can use start & amount to open multiple processes to run, but the temp wav filename must be changed for each process to get correct results.

2. analyzer.py 

   --input (root directory of midi_synchronized)

   --output (root directory of midi_analyzed) 

3. midi2corpus.py 

   --input (root directory of midi_analyzed)

   --output (root directory of corpus) 

4. corpus2events.py 

   --input (root directory of corpus)

   --output (root directory of REMI/events) 

5. remi2muse.py 

   --input (root directory of REMI/events)

   --output (root directory of REMI/muse) 

`synchronizer.py`, `analyzer.py`, `midi2corpus.py`, `corpus2events.py` are modified from [YatingMusic/compound-word-transformer](https://github.com/YatingMusic/compound-word-transformer).

`remi2muse.py` is modified from [@eri24816's comment in YatingMusic/MuseMorphose](https://github.com/YatingMusic/MuseMorphose/issues/1#issuecomment-1178514744).