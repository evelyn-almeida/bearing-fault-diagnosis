| model                                         |   Leave one load out |   Random segment split |   Unseen fault size |
|:----------------------------------------------|---------------------:|-----------------------:|--------------------:|
| All hand-crafted features                     |                0.999 |                  1     |               0.646 |
| Envelope fault frequencies (physics)          |                0.806 |                  0.792 |               0.461 |
| Envelope order spectrum + logistic regression |                0.919 |                  0.938 |               0.642 |
| Time-domain statistics                        |                0.975 |                  0.974 |               0.549 |