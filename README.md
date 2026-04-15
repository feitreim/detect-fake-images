# AI Video Detector

My driving idea here is something I have observed while training GANs, and that
is that training a discriminator model, that is able to detect if an image is
the output of a generator model, isn't actually very difficult\*. When training
a GAN, you need to achieve a balance between generator and the discriminator,
and subsequently I have noticed that having a discriminator that is just as
large and powerful as the generator is not a very good way to achieve this,
because the discriminator will wildly outperform the discriminator. I also know
from training these models that often the discriminator is able to function
purely off of errors that are not very perceptible by humans.

This means that training a model that is able to detect if outputs come from a
specific video generation model may not be all that difficult, and I propose
that we can do it with a fraction of the compute resources needed to train such
a video generator. I believe that we can train small models that operate on tiny
chunks of video, like a small crop from three consecutive frames of video 

# Proof of Concept

The current strategy is to train a single model to predict a single models outputs, currently I am most limited in the supply of generated videos, due to my tiny supply of videos, deep approaches like training a ViT or Conv discriminator were wildly overfitting, however a logistic regression on some simple features does a great job:

 confusion matrix:
             pred_real  pred_fake
    real          8448       2192
    fake           147        413

  - 74% of fakes detected (413/560)
  - 79% of reals correctly kept (8448/10640)

| Model                                            | Train Acc | Val Acc | Val Balanced Acc | Val AUROC | Params |
| ------------------------------------------------ | --------- | ------- | ---------------- | --------- | ------ |
| Logistic regression | 67%   | 75% | 75%          | 0.84  | 25 |

As simple as that is, I think it demonstrates that there is a ton of merit to my original claim. The dataset for those tests is video timelapses of plants growing, and the fake video fakes are seedance2.0 text+image -> video generations, seeded by the label of the video and the first frame.

we train and test on 3 frame x rgb x 64 x 64 crops from the videos.
