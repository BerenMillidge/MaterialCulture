# MaterialCulture

This repository contains an active inference simulation of the interaction of cognition and increasingly complex material culture. We simulate scanpaths showing visual foraging on four levels of pattern complexity, with learning of the underlying transition (B) matrix. Moreover, we show transfer learning to a psychophysics task demonstrating that exposure to material culture may impact psychophysical observables.

This work also contains a two level hierarchy of pixels (bottom-level) and motifs (second-level) such that motifs contextualise the inference at lower levels. Moreover, we introduce the 'likelihood remapping' technique to active inference whereby agents only have access to a local window on the wider world, which is implemented through a state-dependent likelihood (A) matrix, which is thus remapped every timestep as the agent explores its environment. 

The paper can be found here: https://psyarxiv.com/rchaf/

This is joint work with Axel Constant, [Alec Tschantz](https://github.com/alec-tschantz) and Andy Clark.

If you find this paper or the code useful, please cite us as:
```
@article{constant2020acquisition,
  title={The Acquisition of Culturally Patterned Attention Styles under Active Inference},
  author={Constant, Axel and Tschantz, Alexander and Millidge, Beren and Criado-Boado, Felipe and Martinez, Luis M and M{\"u}ller, Johannes and Clark, Andy},
  year={2020},
  publisher={PsyArXiv}
}

```


)
