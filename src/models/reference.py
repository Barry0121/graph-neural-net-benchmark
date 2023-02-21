rnn.train()
output.train()
loss_sum = 0
for batch_idx, data in enumerate(data_loader):
    rnn.zero_grad()
    output.zero_grad()
    x_unsorted = data['x'].float()
    y_unsorted = data['y'].float()
    y_len_unsorted = data['len']
    y_len_max = max(y_len_unsorted)
    x_unsorted = x_unsorted[:, 0:y_len_max, :]
    y_unsorted = y_unsorted[:, 0:y_len_max, :]
    # initialize lstm hidden state according to batch size
    rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))
    # output.hidden = output.init_hidden(batch_size=x_unsorted.size(0)*x_unsorted.size(1))

    # sort input
    y_len,sort_index = torch.sort(y_len_unsorted,0,descending=True)
    y_len = y_len.numpy().tolist()
    x = torch.index_select(x_unsorted,0,sort_index)
    y = torch.index_select(y_unsorted,0,sort_index)

    # input, output for output rnn module
    # a smart use of pytorch builtin function: pack variable--b1_l1,b2_l1,...,b1_l2,b2_l2,...
    y_reshape = pack_padded_sequence(y,y_len,batch_first=True).data
    # reverse y_reshape, so that their lengths are sorted, add dimension
    idx = [i for i in range(y_reshape.size(0)-1, -1, -1)]
    idx = torch.LongTensor(idx)
    y_reshape = y_reshape.index_select(0, idx)
    y_reshape = y_reshape.view(y_reshape.size(0),y_reshape.size(1),1)

    output_x = torch.cat((torch.ones(y_reshape.size(0),1,1),y_reshape[:,0:-1,0:1]),dim=1)
    output_y = y_reshape
    # batch size for output module: sum(y_len)
    output_y_len = []
    output_y_len_bin = np.bincount(np.array(y_len))
    for i in range(len(output_y_len_bin)-1,0,-1):
        count_temp = np.sum(output_y_len_bin[i:]) # count how many y_len is above i
        output_y_len.extend([min(i,y.size(2))]*count_temp) # put them in output_y_len; max value should not exceed y.size(2)
    # pack into variable
    x = Variable(x).cuda()
    y = Variable(y).cuda()
    output_x = Variable(output_x).cuda()
    output_y = Variable(output_y).cuda()
    # print(output_y_len)
    # print('len',len(output_y_len))
    # print('y',y.size())
    # print('output_y',output_y.size())


    # if using ground truth to train
    h = rnn(x, pack=True, input_len=y_len)
    h = pack_padded_sequence(h,y_len,batch_first=True).data # get packed hidden vector
    # reverse h
    idx = [i for i in range(h.size(0) - 1, -1, -1)]
    idx = Variable(torch.LongTensor(idx)).cuda()
    h = h.index_select(0, idx)
    hidden_null = Variable(torch.zeros(args.num_layers-1, h.size(0), h.size(1))).cuda()
    output.hidden = torch.cat((h.view(1,h.size(0),h.size(1)),hidden_null),dim=0) # num_layers, batch_size, hidden_size
    y_pred = output(output_x, pack=True, input_len=output_y_len)
    y_pred = F.sigmoid(y_pred)
    # clean
    y_pred = pack_padded_sequence(y_pred, output_y_len, batch_first=True)
    y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
    output_y = pack_padded_sequence(output_y,output_y_len,batch_first=True)
    output_y = pad_packed_sequence(output_y,batch_first=True)[0]
    # use cross entropy loss
    loss = binary_cross_entropy_weight(y_pred, output_y)
    loss.backward()
    # update deterministic and lstm
    optimizer_output.step()
    optimizer_rnn.step()
    scheduler_output.step()
    scheduler_rnn.step()


    if epoch % args.epochs_log==0 and batch_idx==0: # only output first batch's statistics
        print('Epoch: {}/{}, train loss: {:.6f}, graph type: {}, num_layer: {}, hidden: {}'.format(
            epoch, args.epochs,loss.data[0], args.graph_type, args.num_layers, args.hidden_size_rnn))

    # logging
    log_value('loss_'+args.fname, loss.data[0], epoch*args.batch_ratio+batch_idx)
    feature_dim = y.size(1)*y.size(2)
    loss_sum += loss.data[0]*feature_dim