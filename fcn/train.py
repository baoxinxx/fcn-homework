from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import visdom

from FCN import FCN8s, FCN16s, FCN32s, FCN, VGGNet

def train(epo_num=50, show_vgg_params=False):
    vis = visdom.Visdom()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vgg_model = VGGNet(requires_grad=True, show_params=show_vgg_params)
    fcn_model = FCN(pretrained_net=vgg_model, n_class=2)
    fcn_model = fcn_model.to(device)
    criterion = nn.BCELoss().to(device)
    optimizer = optim.SGD(fcn_model.parameters(), lr=1e-2, momentum=0.7)

    all_train_iter_loss = []
    all_test_iter_loss = []

    prev_time = datetime.now()
    for epo in range(epo_num):
        train_loss = 0
        fcn_model.train()

        for index, (horse, horsemsk) in enumerate(train_dataloader):

            horse = horse.to(device)
            horsemsk = horsemsk.to(device)

            optimizer.zero_grad()
            output = fcn_model(horse)
            output = torch.sigmoid(output)
            loss = criterion(output, horsemsk)
            loss.backward()
            iter_loss = loss.item()
            all_train_iter_loss.append(iter_loss)
            train_loss += iter_loss
            optimizer.step()

            output_np = output.cpu().detach().numpy().copy()
            output_np = np.argmin(output_np, axis=1)
            horse_mask_np = horsemsk.cpu().detach().numpy().copy()
            horse_mask_np = np.argmin(horse_mask_np, axis=1)

            if np.mod(index, 15) == 0:
                print('epoch{},{}/{},train loss is{}'.format(epo, index, len(train_dataloader),iter_loss))
                #vis.close()
                vis.images(output_np[:,None,:,:], win='train_pred', opts=dict(title='train prediction'))
                vis.images(horse_mask_np[:,None,:,:], win='train_label', opts=dict(title='label'))
                vis.line(all_train_iter_loss, win='train_iter_loss', opts=dict(title='train iter loss'))



        test_loss = 0
        fcn_model.eval()
        with torch.no_grad():
            for index, (horse, horsemsk) in  enumerate(test_dataloader):
                horse = horse.to(device)
                horsemsk = horsemsk.to(device)

                optimizer.zero_grad()
                output = fcn_model(horse)
                output = torch.sigmoid(output)
                loss = criterion(output, horsemsk)
                iter_loss = loss.item()
                all_test_iter_loss.append(iter_loss)
                test_loss += iter_loss

                output_np = output.cpu().detach().numpy().copy()
                output_np = np.argmin(output_np, axis=1)
                horse_mask_np = horsemsk.cpu().detach().numpy().copy()
                horse_mask_np = np.argmin(horse_mask_np, axis=1)

                if np.mod(index, 15) == 0:
                    print(r'Testing... Open http://localhost:8097/ to see test result')
                    #vis.close()
                    vis.images(output_np[:, None, :, :], win='test_pred', opts=dict(title='test prediction'))
                    vis.images(horse_mask_np[:, None, :, :], win='test_label', opts=dict(title='label'))
                    vis.line(all_test_iter_loss, win='test_iter_loss', opts=dict(title='test iter loss'))

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time%02d:%02:%02d"%(h, m, s)
        prev_time = cur_time

        print('epoch train loss = %f, epoch test loss = %f, %s'%(train_loss/len(train_dataloader),test_loss/len(test_dataloader), time_str))

        if np.mod(epo, 5) == 0:
            torch.save(fcn_model, 'checkpoint/fcn_model_{}.pt'.format(epo))
            print('saving checkpoints/fcn_model_{}.pt'.format(epo))

if __name__ == "__main__":
    train(epo_num=100, show_vgg_params=False)



