## Setup jazz dataset

First download jazz dataset from
https://drive.google.com/file/d/1DL60nip8JEBhNS_Rf9k8Jz-Ukg3wdbsB/view?usp=share_link


```
tar xzvf jazz.tar.gz
```

The `jazz` directory contains:
- `meta`: Metadata, such as tokenizer configuration, song mapping, etc.
- `midi`: Original jazz MIDI files.
- `pickles`: Tokenized song data in REMI format. This follows the same file format as the original `remi_dataset`. Each pickle contains a tuple: `(bar_ids, tokens)`.

## Training

Run the training script with the following command:

```
python train.py config/jazz.yaml
```

## Known Issues with jazz Dataset

The tokenizer used to create the original dataset `remi_dataset` differs from the tokenizer used in the jazz dataset. Although both follow the REMI format, they use different token string formats:

- `Beat` vs. `Position`
- `Note_Pitch` vs. `Pitch`
- `Chord_B:7aug` vs. `Chord_A#_+`

As a result of these differences, we are unable to extract the song "attributes" using the `attributes.py` script (song attributes are a special feature used to feed the model). Additionally, data augmentation (`do_augment` in the config file) has been disabled as it currently does not work with the jazz dataset's token string format.

It should be possible to resolve the token format differences through conversion, but further investigation is required.

To see how we currently tokenize the jazz dataset: `data_process.ipynb`.


## TODO

- [ ] Fix the token format so that we can run `attributes.py` on the jazz dataset
- [ ] Fix the token format so that we data argumentation is possible (you can try data argumentation by setting `do_augment`)