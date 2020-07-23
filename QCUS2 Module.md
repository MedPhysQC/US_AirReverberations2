#QC US Module

##Abstract
The QC module for Ultra Sound analyses the pattern of reverberations in air. Analysis is done in two parts: a sensitivity analysis and a uniformity analysis. The results of the QC are strongly dependent on the probe and the acquisition protocol used.

##Status
Validating

##Dependencies
ocr

##Acquisition protocol
US machines have many different applications (acquisition protocols, or presets) and a user is used to change many settings to optimize the displayed image. For QC it is vital that all image acquisition parameters are constant between QC acquisitions. 

The exact choice for the acquisition parameters should not matter too much, but since the objective is to detect flaws, it is recommended to pick settings that do not try to enhance an image.

As a guideline, the QC acquisition protocol can be build as follows. Notice that each probe should have its own QC protocol.

1. Choose the most used preset for the probe; e.g. for C5-1 pick the Generic Abdomen preset.
2. Choose a Linear LUT; e.g. for Philips this is M3.
3. Choose full sector width (all crystals should be used)
4. Switch off all fancy settings like Harmonics, XRes, SonoCT, AutoQuality etc. This is important, as using Harmonics will general result in nicer images, it will also confound small faults in the probe.
5. Switch off persistence
6. Prefer the Res(olution) setting over the Gen(eric) and Pen(etration).
7. Prefer Resolution over Speed
8. Pick a power of 0 dB
9. Do not use color maps (Chroma)
10. Pick a zoom that is as large as possible, but it should not result in a any objects or annotations to overlap the reverberation pattern.
11. Choose a fixed dynamic range (sometimes called compression) ; e.g. 55 for Philips.
12. Choose a gain that does just not result in clipping of high intensities. Note this gain, because it will be one of the few remaining user variables.
13. Store the above settings to a new preset, so each QC acquisition uses those settings.

As it it common practice to exchange probes between machines, mark the probes so that each time a QC acquisition is performed with a probe, it is used with the same machine.

Before making a QC acquisition, make sure that:

1. The particular probe is connected to the same machine as before.
2. The probe is properly connected, and the lens is clean.
3. All TGC settings are set to neutral.
4. The correct QC protocol is selected.
5. The correct Gain is selected.
