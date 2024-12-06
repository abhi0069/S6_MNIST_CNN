# MNIST Model Test Results

## Model Architecture Tests
### Parameter Count Test
✓ Total parameters: 13,697 (Under 20,000 limit)

### Layer Tests
✓ Uses Batch Normalization  
✓ Uses Dropout (0.05)  
✓ Uses Global Average Pooling  

## Test Results
### Model Performance
- Total samples tested: 10,020
- Correct predictions: XXXX
- Final accuracy: XX.XX%
- Average Loss: 0.XXXX

### Target Metrics Status
- ✓ Parameters: Under 20,000 limit (13,697)
- ✓ Uses Batch Normalization
- ✓ Uses Dropout
- ✓ Uses GAP
- ✓/✗ Accuracy: XX.XX% (Target: 99.40%)

## Training Configuration
- Batch Size: 1024
- Optimizer: Adam (lr=0.004)
- Learning Rate Scheduler: ReduceLROnPlateau
- Max Epochs: 19 (Early stopping when target accuracy reached)

## Test Logs
Test Set - Average Loss: 0.0337, Accuracy: 9923/10020 (99.03%)

## Training Logs
Epoch: 1 | Train Loss: 0.4818 | Batch: 48: 100%|██████████| 49/49 [01:55<00:00,  2.36s/it]

Epoch: 1
Training Set - Average Loss: 1.2137, Accuracy: 35467/49980 (70.96%)

Validation Results:

Validation Set - Average Loss: 0.5899, Accuracy: 8653/10020 (86.36%)

New best model saved! Accuracy: 86.36%
Epoch: 2 | Train Loss: 0.1560 | Batch: 48: 100%|██████████| 49/49 [01:51<00:00,  2.28s/it]

Epoch: 2
Training Set - Average Loss: 0.2707, Accuracy: 47719/49980 (95.48%)

Validation Results:

Validation Set - Average Loss: 0.1815, Accuracy: 9650/10020 (96.31%)

New best model saved! Accuracy: 96.31%
Epoch: 3 | Train Loss: 0.1093 | Batch: 48: 100%|██████████| 49/49 [01:51<00:00,  2.27s/it]

Epoch: 3
Training Set - Average Loss: 0.1282, Accuracy: 48673/49980 (97.38%)

Validation Results:

Validation Set - Average Loss: 0.1284, Accuracy: 9711/10020 (96.92%)

New best model saved! Accuracy: 96.92%
Epoch: 4 | Train Loss: 0.0640 | Batch: 48: 100%|██████████| 49/49 [01:50<00:00,  2.25s/it]

Epoch: 4
Training Set - Average Loss: 0.0882, Accuracy: 49009/49980 (98.06%)

Validation Results:

Validation Set - Average Loss: 0.0960, Accuracy: 9756/10020 (97.37%)

New best model saved! Accuracy: 97.37%
Epoch: 5 | Train Loss: 0.0601 | Batch: 48: 100%|██████████| 49/49 [01:50<00:00,  2.26s/it]

Epoch: 5
Training Set - Average Loss: 0.0696, Accuracy: 49167/49980 (98.37%)

Validation Results:

Validation Set - Average Loss: 0.1058, Accuracy: 9705/10020 (96.86%)

Epoch: 6 | Train Loss: 0.0468 | Batch: 48: 100%|██████████| 49/49 [01:53<00:00,  2.31s/it]

Epoch: 6
Training Set - Average Loss: 0.0574, Accuracy: 49263/49980 (98.57%)

Validation Results:

Validation Set - Average Loss: 0.0812, Accuracy: 9802/10020 (97.82%)

New best model saved! Accuracy: 97.82%
Epoch: 7 | Train Loss: 0.0352 | Batch: 48: 100%|██████████| 49/49 [01:52<00:00,  2.30s/it]

Epoch: 7
Training Set - Average Loss: 0.0510, Accuracy: 49353/49980 (98.75%)

Validation Results:

Validation Set - Average Loss: 0.0709, Accuracy: 9816/10020 (97.96%)

New best model saved! Accuracy: 97.96%
Epoch: 8 | Train Loss: 0.0464 | Batch: 48: 100%|██████████| 49/49 [01:53<00:00,  2.32s/it]

Epoch: 8
Training Set - Average Loss: 0.0435, Accuracy: 49435/49980 (98.91%)

Validation Results:

Validation Set - Average Loss: 0.0601, Accuracy: 9849/10020 (98.29%)

New best model saved! Accuracy: 98.29%
Epoch: 9 | Train Loss: 0.0422 | Batch: 48: 100%|██████████| 49/49 [01:51<00:00,  2.28s/it]

Epoch: 9
Training Set - Average Loss: 0.0387, Accuracy: 49487/49980 (99.01%)

Validation Results:

Validation Set - Average Loss: 0.0559, Accuracy: 9868/10020 (98.48%)

New best model saved! Accuracy: 98.48%
Epoch: 10 | Train Loss: 0.0374 | Batch: 48: 100%|██████████| 49/49 [01:51<00:00,  2.27s/it]

Epoch: 10
Training Set - Average Loss: 0.0361, Accuracy: 49519/49980 (99.08%)

Validation Results:

Validation Set - Average Loss: 0.0633, Accuracy: 9845/10020 (98.25%)

Epoch: 11 | Train Loss: 0.0378 | Batch: 48: 100%|██████████| 49/49 [01:51<00:00,  2.28s/it]

Epoch: 11
Training Set - Average Loss: 0.0334, Accuracy: 49537/49980 (99.11%)

Validation Results:

Validation Set - Average Loss: 0.0416, Accuracy: 9891/10020 (98.71%)

New best model saved! Accuracy: 98.71%
Epoch: 12 | Train Loss: 0.0340 | Batch: 48: 100%|██████████| 49/49 [01:50<00:00,  2.26s/it]

Epoch: 12
Training Set - Average Loss: 0.0310, Accuracy: 49551/49980 (99.14%)

Validation Results:

Validation Set - Average Loss: 0.0534, Accuracy: 9849/10020 (98.29%)

Epoch: 13 | Train Loss: 0.0252 | Batch: 48: 100%|██████████| 49/49 [01:50<00:00,  2.25s/it]

Epoch: 13
Training Set - Average Loss: 0.0303, Accuracy: 49576/49980 (99.19%)

Validation Results:

Validation Set - Average Loss: 0.0437, Accuracy: 9884/10020 (98.64%)

Epoch: 14 | Train Loss: 0.0304 | Batch: 48: 100%|██████████| 49/49 [01:50<00:00,  2.26s/it]

Epoch: 14
Training Set - Average Loss: 0.0287, Accuracy: 49571/49980 (99.18%)

Validation Results:

Validation Set - Average Loss: 0.0511, Accuracy: 9858/10020 (98.38%)

Epoch: 15 | Train Loss: 0.0223 | Batch: 48: 100%|██████████| 49/49 [01:51<00:00,  2.28s/it]

Epoch: 15
Training Set - Average Loss: 0.0226, Accuracy: 49692/49980 (99.42%)

Validation Results:

Validation Set - Average Loss: 0.0369, Accuracy: 9900/10020 (98.80%)

New best model saved! Accuracy: 98.80%
Epoch: 16 | Train Loss: 0.0259 | Batch: 48: 100%|██████████| 49/49 [01:51<00:00,  2.28s/it]

Epoch: 16
Training Set - Average Loss: 0.0197, Accuracy: 49747/49980 (99.53%)

Validation Results:

Validation Set - Average Loss: 0.0356, Accuracy: 9910/10020 (98.90%)

New best model saved! Accuracy: 98.90%
Epoch: 17 | Train Loss: 0.0203 | Batch: 48: 100%|██████████| 49/49 [01:52<00:00,  2.29s/it]

Epoch: 17
Training Set - Average Loss: 0.0194, Accuracy: 49744/49980 (99.53%)

Validation Results:

Validation Set - Average Loss: 0.0337, Accuracy: 9923/10020 (99.03%)

New best model saved! Accuracy: 99.03%
Epoch: 18 | Train Loss: 0.0225 | Batch: 48: 100%|██████████| 49/49 [01:51<00:00,  2.28s/it]

Epoch: 18
Training Set - Average Loss: 0.0193, Accuracy: 49729/49980 (99.50%)

Validation Results:

Validation Set - Average Loss: 0.0389, Accuracy: 9900/10020 (98.80%)
