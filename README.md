# SMBPANN
Self-Modifying Backpropagation (Feed Forward) Neural Network 

The goal.
Rather than implementing function
PREDICTOR = FUNCTION (TrainingSet, TestingSet, NN_Topology, Error, ActivationFunction, JitterLevel),
where NN_Topology is the hardcoded architecture of the net (number of hidden layers, form and number of edges),
we want instead:
SELF_MODIFYING_PREDICTOR = FUNCTION (TrainingSet, TestingSet, Error)
returning other parameters back to the network as a feedback (according to the performance function)

Language choice. Why java and not C++ and not ANSI C?
1) According to a measurement 10 years ago, I program the same functionality 3+ times faster in Java than in ANSI C. 
Also, I'm not an expert in CPP, so it will be even slower than in ANSI C.
Note: if more developers will participate who write in C faster than in Java, then this item might be revised;
2) According to my brief tests (http://siberean.livejournal.com/2253.html), in-memory operations with already in-memory structures (preloaded) are not slower written, compiled and running in Java than in C, and the existence of efficient free algorithm libraries such as HashMaps (red-black trees) etc makes the task easier;
3) Easier portability;
4) More potential code participants if using Java comparing with niche ANSI C.
Anyway, having a succesfull Java prototype, written in 1/3 time, it is possible to rewrite (in case of success) in C.

Frameworks, code style ideology.
It is suggested to keep frameworks at bay (for performance, simplicity, debugging reasons). No Spring, no Java8 annotations, no declarative programming. Only imperative programming, debuggable by a traditional debugger (not by gdb but at least by Eclipse). In complex code no space and time to guess. It must be a way to plainly debug. And automatic regression tests covering everything, from bottom-up, from simplest cases, from the very beginning.

No external dependencies, no Maven (since no frameworks - no need in it). Plain single ant build file. Target distribution is in plain jar file. It must be easy to run the neural network from command line (from scripts).

Plugins. Neural Network is a general purpose processor, so inputs might be some sound tracks as well as images, 
or even DNA sequences. It must be an extremely easy way to add a new plugin to read an input from any source: from files, from a database or from any other data sources. Clean API well documented must exists for easy integrating the NN.

Parameters such as bias range or initial noise level etc - all will be put in one place for simplicity.

SIMLICITY, KISS principle, unit tests covering everything - this is what I'm thinking all the time. If a business logic requires 1 bit of entropy, 1 "if" while a framework generates 2 bits for the sake of technology - this is a bad technology: the overall complexity of the system should not be much bigger than the complexity of the business requirements (a valid exception - logging levels in log4j been expressed in a plain set of parameters in one separate file).

Backprop Feed-Forward Neural Networks Tutorial (old but still valid, nothing changed since then, except the performace of the machines), but you can get the same information on Wikipedia nowadays, I'm putting it here - just for my reference (and sorry for that): 
https://drive.google.com/file/d/0B9Ee4Kn3OEaRTF91Y0dBQ29ldlU/view

Respecting both algorithmical approach (starting programming from the processing function) and OO approach (starting from the domain objects), yin and yang of programming, for the sake of clarity, we'll start from the common (in the business domain of ANN) terminology, naming classes in familiar terms: Network, Layer, Neuron (better to separate it into: Node + Axons/Edges), ActivationPhase(), BackPropagationPhase().
To keep the code maintainable - we keep class inheritance at bay, only when necessary, using aggregation instead (for dynamic properties - Network consists of an ArrayList of Layers, Layer consists of Nodes, Nodes contain ArrayLists of Edges or Axons, also containing Activation function and encapsulated activation, backpropagation convenience routine). This approach will make it easier to discuss and debug the processing. 

No need in GPU support. There are no matrix (in math terms) operations envisioned for now: the neurons are linked with neigbours and calculations are to be done by chain rules, iterating, in sequential loops. No need in threading right now (the processing of a Network must be processed by the same process/thread anyway, according to the Chain Rule). But future optimized ("garbage collector" for unused neurons might run in a separate thread, but it is early to discuss it for now). On the contrary, different Networks (instances with different architecture running against the same data sets) might run in-parallel, so Processor, processing the Network better to be a Thread - to benefit from parallel processing and future clastering.

The first unit test should be probably the classical Perceptron anyway, for the sake of tradition and as the very first running test.