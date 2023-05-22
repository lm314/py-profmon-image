# py-profmon-image

The following class is for reading image data that is stored in a MATLAB structure using the `scipy.io` module as well as displaying it using `matplotlib` and saving it outside the MATLAB fiel format using the `PIL` package.

The MATLAB data structure has the following variable and fields contained in the ProfMon-Example.mat file:

<pre>
├── data
    ├── img
    ├── roiXN
    ├── roiYN
    └── res
</pre>

An example use case can be found [here](examples/DistGen_2dFile.ipynb), which runs through how to use this class to create a particle distribution at a cathode using the [distgen](https://github.com/ColwynGulliford/distgen) pacakge

 
