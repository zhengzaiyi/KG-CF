import torch

class LSTMClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_relation, num_entity, output_dim):
        # TODO: use pretrained embeddings
        super(LSTMClassifier, self).__init__()
        self.entity_embedding = torch.nn.Embedding(num_entity, input_size)
        self.relation_embedding = torch.nn.Embedding(num_relation, input_size)
        self.lstm = torch.nn.LSTM(4 * input_size, hidden_size, num_layers)
        self.fc = torch.nn.Linear(hidden_size, output_dim)
    
    def forward(self, batch):
        # sequence: [(e1, e2, r1, r_q), ...]
        # use concat(e1, e2, r1, r_q) as input
        lstm_input = torch.cat([
            self.entity_embedding(batch['lstm_path'][:, :, 0]),
            self.entity_embedding(batch['lstm_path'][:, :, 1]),
            self.relation_embedding(batch['lstm_path'][:, :, 2]),
            self.relation_embedding(batch['lstm_path'][:, :, 3]),
        ], dim=-1)
        lstm_output, _ = self.lstm(lstm_input)
        lstm_output = lstm_output[:, -1, :]
        logits = self.fc(lstm_output)
        logits = torch.sigmoid(logits)
        return logits

# if __name__ == '__main__':
#     classifier = LSTMClassifier(
#         input_size=kg.entity_embedding_dim, 
#         hidden_size=50, 
#         num_layers=args.num_hop, 
#         num_relation=num_relation,
#         num_entity=num_entity,
#         output_dim=1)
#     classifier.to(device)
#     optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
#     criterion = F.binary_cross_entropy_with_logits
#     lstm_dataloader = DataLoader(
#         dataset=lstm_dataset,
#         batch_size=16,
#         shuffle=True,)
#     if os.path.exists('models/LSTM_Classifier.pth'):
#         classifier.load_state_dict(torch.load('models/LSTM_Classifier.pth'))
#     else:
#         for epoch in range(args.num_epochs):
#             for i, batch in enumerate(lstm_dataloader):
#                 for key in ['lstm_path', 'id_query', 'label']:
#                     batch[key] = batch[key].to(device)
#                 logits = classifier(batch)
#                 loss = criterion(logits, batch['label'])
#                 loss.backward()
#                 optimizer.step()
#                 optimizer.zero_grad()
#             print(f'lstm: epoch {epoch} loss: {loss.item()}')
#         torch.save(classifier.state_dict(), 'models/LSTM_Classifier.pth')
    
    