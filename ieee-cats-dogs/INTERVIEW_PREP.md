# Explain this project in your own words

## A 45-second introduction

This project classifies cat and dog images using a pretrained ResNet-18 in PyTorch. I am using the model's existing ImageNet knowledge, without training it again. The supplied dataset uses breed filenames, so the code maps breeds to cat or dog labels. It evaluates 200 images of each species and groups the relevant ImageNet scores into two species scores. Grad-CAM then highlights regions that contributed to the predicted species score. The recorded run reached 94% accuracy, but it also showed that a model can look at the animal's face and still predict the wrong species.

Use this explanation only after running the notebook and understanding the steps. Describe assistance and your own contribution honestly if asked.

## Ground zero glossary

- **Dataset:** the collection of example images.
- **Label:** the known answer, cat or dog, derived from the breed name.
- **Preprocessing:** converting each image into the numeric format the model expects.
- **Tensor:** an array of numbers. One input batch has shape `[batch, 3, 224, 224]`.
- **Pretrained:** someone already learned the model weights using another dataset.
- **Inference:** using learned weights to predict an answer.
- **Logit:** a model output score before conversion to probabilities.
- **Feature map:** a grid showing where a learned visual pattern responds.
- **Gradient:** how much a target score would change with a small change in another quantity.
- **Grad-CAM:** a method that uses gradients and feature maps to make a coarse explanation heatmap.

## Questions you should be able to answer

**Did you train your model?** No. The pretrained ImageNet ResNet-18 weights stay fixed. There is no training loop or optimizer. Computing gradients for Grad-CAM does not itself train a model.

**Why ResNet-18?** It is a standard pretrained ResNet and is small enough for a quick Colab or CPU demonstration. Residual connections let information pass around blocks and help deeper networks learn.

**Does it naturally output cat and dog?** No. It outputs 1,000 ImageNet scores. The code groups five domestic-cat categories and 118 domestic-dog categories. Softmax over the grouped logsumexp scores gives conditional cat/dog scores. This is an inference baseline, not a trained binary head.

**Why not assume indices 0 and 1 mean cat and dog?** Those positions in ImageNet mean tench and goldfish. Output meanings come from the model's category metadata.

**How do you know the correct answer?** The dataset filename contains a known breed. An explicit map converts that breed to species. Model outputs are not used as ground truth.

**Why normalize?** The pretrained model expects the same input scaling used during its training. Normalization adjusts each channel using ImageNet mean and standard deviation.

**Why `eval()` and `inference_mode()`?** `eval()` switches model behavior such as batch normalization to inference behavior. `inference_mode()` avoids gradient tracking for ordinary predictions. Grad-CAM needs gradient tracking, so it runs outside that context with gradients explicitly enabled.

**What is your Grad-CAM target?** The grouped raw score of the predicted species. This makes the explanation consistent with the binary decision rule. It is not just the highest individual breed score.

**Explain the heatmap calculation.** Capture last-block feature maps, differentiate the target score, average gradients per channel, weight and sum channels, apply ReLU, resize and normalize.

**Why align to the crop?** ResNet sees the center crop, not the entire photo. Overlaying on the original full image would misalign locations unless the crop transformation is reversed correctly.

**What were the results?** 376/400 correct: 176/200 cats and 200/200 dogs. All 24 errors were cats predicted as dogs. The exact CSV and metrics are included.

**What did the errors teach you?** Sphynx examples can be assigned dog even when activation covers their head. Seeing the right region is not the same as interpreting its appearance correctly. Category imbalance or limited cat coverage may contribute, but these were not causally tested.

**Does red mean the model is certain?** No. Red means higher relative positive contribution within that image's normalized map. Maps are independently normalized.

**Can you claim 94% on all images?** No. It is the result on the fixed selected 400 images. We did not use the official Oxford benchmark split or audit overlap with ImageNet pretraining.

**How could you improve it?** Evaluate a larger fixed subset and breed-level error rates, compare another pretrained ResNet, investigate crop effects, and only if permitted fine-tune a binary classification head.

## Before the screening

Run the portable notebook in Colab. Open one correct and one incorrect heatmap. Trace a CSV row back to its image and verify its breed label. Be able to explain `group_logits()` and the six main steps of `gradcam()`. Do not memorize an explanation you cannot connect to the code.
