# Self-Modifying Back Propagation Feed-Forward Neural Network

The project was started in the early 2000s, after the author graduated from university with a specialization in Back Propagation Feed-Forward neural networks. The tensor implementation of the delta rule was one of the subjects of the final work. The project was abandoned after the appearance of TensorFlow and due to the author's lack of spare time for implementation.

## The Goal

Rather than implementing a function like this:

```text
PREDICTOR = FUNCTION (
    TrainingSet,
    TestingSet,
    NN_Topology,
    Error,
    ActivationFunction,
    JitterLevel
)
```

where `NN_Topology` is the hardcoded architecture of the net: the number of hidden layers, the form and number of edges, and numerical parameters.

We want instead:

```text
SELF_MODIFYING_PREDICTOR = FUNCTION (
    TrainingSet,
    TestingSet,
    Error
)
```

returning other parameters back to the network as feedback, according to the performance function. In other words, we want hyper-parameterization of the ANN.

## Some Literature

*Not in order of significance!*

* Discussion about hyper-parameters and a lot of practical considerations:
  http://yann.lecun.com/exdb/publis/pdf/bengio-lecun-07.pdf

* Why Deep Learning, and why not to experiment first with shallow architectures such as SVM; also useful discussions:
  http://arxiv.org/pdf/1206.5533v2.pdf

* A very basic introductory Back Propagation Feed-Forward Neural Networks tutorial. It is old but still valid; nothing has changed since then, except the performance of the machines. You can get the same information on Wikipedia nowadays. I'm putting it here just for my reference, and sorry for that:
  https://drive.google.com/file/d/0B9Ee4Kn3OEaRTF91Y0dBQ29ldlU/view

* A good tutorial:
  http://karpathy.github.io/neuralnets/

## Some Preliminary Design Considerations

### Language Choice

Why Java, and not C++ or ANSI C?

* According to a measurement from 10 years ago, I program the same functionality 3+ times faster in Java than in ANSI C. Also, I'm not an expert in C++, so it would be even slower than in ANSI C.
  Note: if more developers participate who write in C faster than in Java, then this item might be revised.

* According to my brief tests:
  http://siberean.livejournal.com/2253.html
  in-memory operations with already in-memory structures, pre-loaded, are not slower when written, compiled, and run in Java than in C. The existence of efficient free algorithm libraries such as `HashMap`, red-black trees, etc. makes the task easier.

* Easier portability.

* More potential code participants when using Java compared with niche ANSI C. Anyway, having a successful Java prototype, written in one third of the time, it is possible to rewrite it in C in case of success.

### Frameworks and Code Style Ideology

It is suggested to keep frameworks at bay for performance, simplicity, and debugging reasons.

No Spring, no Java 8 annotations, no declarative programming. Only imperative programming, debuggable by a traditional debugger: not by `gdb`, but at least by the Eclipse debugger.

In complex code, there is no space and time for guesses. There must be a way to plainly debug the code by observing the processing sequence and isolating the particular execution thread in case of multi-threading.

Automatic regression tests should already cover everything, from bottom to top, from the simplest cases, from the very beginning.

### Dependencies and Build

No external dependencies, no Maven. Since there are no frameworks, there is no need for it.

A plain single Ant build file.

The target distribution is a plain `.jar` file.

It must be easy to run the neural network from the command line, from scripts.

### Plugins

A Neural Network is a general-purpose processor, so inputs might be sound tracks as well as images, or even DNA sequences.

There must be an extremely easy way to add a new plug-in to read input from any source: files, a database, or any other data sources.

A clean, well-documented API must exist for easy integration with the NN.

### Parameters

Parameters such as bias range, initial noise level, etc. will all be put in one place for simplicity.

### Simplicity

SIMPLICITY, the KISS principle, and unit tests covering everything — this is what I'm thinking about all the time.

If a business logic requires 1 bit of entropy, 1 `if`, while a framework generates 2 bits for the sake of technology, this is bad technology. The overall complexity of the system should not be much bigger than the complexity of the business requirements.

A valid exception: logging levels in `log4j` being expressed in a plain set of parameters in one separate file.

### Maintainability

To keep the code maintainable, we keep class inheritance at bay and use it only when necessary, using aggregation instead.

For example, for dynamic properties:

* `Network` consists of an `ArrayList` of `Layers`.
* `Layer` consists of `Nodes`.
* `Nodes` contain `ArrayLists` of `Edges` or `Axons`.
* Nodes also contain an activation function and encapsulated activation/backpropagation convenience routines.

This approach will make it easier to discuss and debug the processing.

### GPU Support and Parallelism

There is no need for GPU support.

There are no matrix operations, in mathematical terms, envisioned for now. The neurons are linked with neighbours, and calculations are to be done by chain rules, iterating in sequential loops.

There is no need for threading right now. The processing of a `Network` must be done by the same process/thread anyway, according to the Chain Rule.

In the future, the network dynamic optimizations, the "garbage collector" for unused neurons, might run in a separate thread. But it is too early to discuss this for now.

On the contrary, different `Network` instances, with different topologies/architectures running against the same data sets, might run in parallel. So the `Processor` processing the `Network` had better be a `Thread`, to benefit from parallel processing and future clustering.

## First Unit Test

The first unit test should probably be the classical Perceptron anyway, for the sake of tradition and as the very first running test.
