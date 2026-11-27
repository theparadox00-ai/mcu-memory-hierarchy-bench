# Placements

One folder per model and SRAM budget:

```
placements/<model>/s<KB>/nn_place.c
placements/<model>/s<KB>/nn_place.h
```

Generate with:
```
python3 tools/make_placement.py models/<model>.tflite <plan.html|plan.json> <board>/placements/<model>/s<KB> models/<model>/nn_model.c
```
