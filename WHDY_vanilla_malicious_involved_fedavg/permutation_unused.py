# prepare test data
		if not self.shard_test_data:
			test_data = torch.tensor(mnist_dataset.test_data)
			test_label = torch.argmax(torch.tensor(mnist_dataset.test_label), dim=1)
			test_data_loader = DataLoader(TensorDataset(test_data, test_label), batch_size=100, shuffle=False)
		else:
			test_data = mnist_dataset.test_data
			test_label = mnist_dataset.test_label
			 # shard test
			shard_size_test = mnist_dataset.test_data_size // self.num_of_clients // 2
			shards_id_test = np.random.permutation(mnist_dataset.test_data_size // shard_size_test)