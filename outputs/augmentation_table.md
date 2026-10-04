# Augmentations (train only)

From `shared/transforms.py` (AUGMENTATIONS).

| augmentation | parameters | justification |
|---|---|---|
| RandomResizedCrop | output 224x224, area 0.8 to 1.0, aspect 3/4 to 4/3 | zoom and framing vary between photographers; at least 80% of the frame stays, so the centred lesion is not cut out |
| RandomHorizontalFlip | p = 0.5 | a lesion has no left or right side; the mirror image has the same diagnosis |
| RandomVerticalFlip | p = 0.5 | the dermatoscope or camera can be held either way up |
| RandomQuarterTurn | 0, 90, 180 or 270 degrees | orientation is arbitrary; quarter turns add no interpolation or black corners |
| ColorJitter | brightness 0.1, contrast 0.1, saturation 0, hue 0 | exposure differs between devices; hue is never touched because colour (red, brown, blue-grey) is diagnostic |
