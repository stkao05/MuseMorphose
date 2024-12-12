Preprocessing

0. Setup

   `$ pip install -r requirements.txt`

   The output file tree will be the same as the original directory, so tidy up the directory if needed.

1. synchronizer.py 

   --input (root directory of raw midi files)

   --output (root directory of output midi files) 

   --start (index of file to start processing)

   --amount (number of files to process)
   
   synchronizer.py expects 4 beats in a bar, and will produce wrong results otherwise.

   You can use start & amount to open multiple processes to run, but the temp wav filename must be changed for each process to get correct results.
