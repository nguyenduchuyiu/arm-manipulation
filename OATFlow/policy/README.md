# Shared action components

`model.py` contains the context decoder and six-coordinate FM expert used by
`OATFlow.prior` and reach experiments. `loader.py` supplies video decoding and
minibatch helpers. The former TCOW mask/FM pipeline and its train/eval/rollout
wrappers have been removed. Amodal XYZ is in [tracking](../tracking/README.md).
