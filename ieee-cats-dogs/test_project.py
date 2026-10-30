"""Small checks for label mapping and explanation math; no dataset download."""
import unittest
from pathlib import Path
import numpy as np
import torch
from torchvision.models import resnet18
from project import label_for, CAT_BREEDS, DOG_BREEDS, CAT_IDS, DOG_IDS, group_logits, gradcam

class ProjectTests(unittest.TestCase):
    def test_labels(self):
        for b in CAT_BREEDS: self.assertEqual(label_for(Path('images')/(b+'_1.jpg')),0)
        for b in DOG_BREEDS: self.assertEqual(label_for(Path('images')/(b+'_1.jpg')),1)
        self.assertIsNone(label_for(Path('images/unknown_1.jpg')))
    def test_grouping_matches_probability_sums(self):
        torch.manual_seed(42)
        z=torch.randn(8,1000)
        p=z.softmax(1)
        expected=torch.stack([p[:,CAT_IDS].sum(1),p[:,DOG_IDS].sum(1)],dim=1)
        expected/=expected.sum(1,keepdim=True)
        torch.testing.assert_close(group_logits(z).softmax(1),expected)
    def test_cam_finite_and_weights_unchanged(self):
        torch.set_num_threads(2)
        model=resnet18(weights=None).eval().requires_grad_(False)
        before={k:v.clone() for k,v in model.state_dict().items()}
        for target in [0,1]:
            cam,blank=gradcam(model,torch.rand(1,3,224,224),target)
            self.assertEqual(cam.shape,(224,224))
            self.assertTrue(np.isfinite(cam).all())
            self.assertGreaterEqual(cam.min(),0);self.assertLessEqual(cam.max(),1)
        for k,v in model.state_dict().items():self.assertTrue(torch.equal(v,before[k]))
        self.assertEqual(len(model.layer4[-1]._forward_hooks),0)

if __name__=='__main__':unittest.main()
