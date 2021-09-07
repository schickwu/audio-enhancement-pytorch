# audio-enhancement-pytorch
Here, i have customize two different loss functions for ideal ratio mask (IRM) estimation. And I also utilised the MSEloss(from pytorch).

The two custimised loss function are structured based on the paper from Yan Zhao, Buye Xu, Ritwik Giri, Tao Zhang 'Perceptually Guided Speech Enhancement Using Deep Neural Networks', ICASSP 2018, Calgary, AB, Canada.

Specifically, 'stoiloss_16k' refers to the stoi intelligibility based loss function,
and 'infoloss_16k' refers to the information theory intelligibility based loss function.


And for the Mask training stage, the python file named 'MaskTraining' is used, and it includes the training procedures for all three different loss functions.

And the named 'MaskTesting' python file is used for testing the pesq values and stoi values for the reconstructed enhanced (through trained deep neural network) speech signal.

The 'MaskChecking' python file is used for reconstructing some example enhanced speech signals.






Besides that, there is also a mapping approach in the 'mapping' python file, based on the paper from Yong Xu, Jun Du, Li-Rong Dai, Chin-Hui Lee 'A Regression Approach to Speech Enhancement Based on Deep Neural Networks'  IEEE/ACM Transactions on Audio, Speech, and Language Processing, 10.2014, and this could be independently utilized for mapping+proposed loss enhancement.
