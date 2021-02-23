import torch
import torch.nn as nn

def train(model, train_set, loss, batch_size=32, learning_rate=1e-3, report_rate=50, epoch_num=20):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    iter_in_epoch = 50
    running_loss = 0.0

    for epoch in range(epoch_num):
        for i in range(iter_in_epoch):
            # ===================forward=====================
            output = model.forward(train_set)
            total_loss = loss(output, train_set)
            # ===================backward====================
            optimizer.zero_grad()
            total_loss.backward(retain_graph=True)
            optimizer.step()

            running_loss += total_loss.item()
            if (i+1) % report_rate == 0:    
                print('[%d, %5d] loss: %.5f' %
                    (epoch + 1, i + 1, running_loss))
                running_loss = 0.0

    print('Finished Training')