import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import mlflow
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, confusion_matrix, recall_score, precision_score, accuracy_score, cohen_kappa_score, matthews_corrcoef



# define the model architecture
class IntegerTransformer3(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_heads, num_layers):
        super().__init__()
        self.conv = nn.Conv2d(kernel_size=(1, embedding_dim), out_channels=1, in_channels=1)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(num_embeddings, num_heads),
            num_layers
        )

        self.fc1 = nn.Linear(100, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 2)


    def forward(self, input_ids):
        embedded = self.conv(input_ids)
        output = self.transformer(embedded.squeeze(3))
        out = self.fc1(output.squeeze(1))
        out = self.relu(out)
        out = self.fc2(out).softmax(dim=1)
        return out

def train_model(model, mlflow, writer,train_dataloader,val_dataloader ,log=True, num_epochs=100 , learning_rate = 0.0001):
    model = model.to('cuda:0')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # train the model
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_accuracy = 0
        n_sample = 0

        p = []
        r = []

        for i, (X_batch, y_batch) in enumerate(train_dataloader):
            n_sample += len(X_batch)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(X_batch.unsqueeze(1).to('cuda:0'))
            loss = criterion(outputs.squeeze(1), y_batch.type(torch.LongTensor).to('cuda:0'))
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)

            p += list(predicted.cpu())
            r += [y_batch]
            running_accuracy += torch.sum(predicted == y_batch.to('cuda:0')).item()

          # update the running loss
            running_loss += loss.item() * len(X_batch)

        y_true = []
        for q in r:
            y_true += q

        y_pred = []
        for q in p:
            y_pred += [q.item()]


        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_micro = f1_score(y_true, y_pred, average='micro')
        accuracy = running_accuracy / n_sample
        # print the average loss for the epoch
        epoch_loss = running_loss / n_sample
        if log:
            writer.add_scalar('Train Loss', epoch_loss, epoch)
            writer.add_scalar('Train Accuracy', accuracy, epoch)
            writer.add_scalar('Train F1_Macro', f1_macro, epoch)
            writer.add_scalar('Train F1_micro', f1_micro, epoch)
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f} Accuracy: {accuracy:.4f} F1_Macro: {f1_macro:.4f} F1_micro: {f1_micro:.4f}')

        if epoch % 20 == 0:
            if log:
                print("#######################################")
                print(' val evaliation : ')
            model.eval()  # Set the model to evaluation mode
            val_accuracy = 0
            val_num_samples = 0

            val_true = []
            val_pred = []


            for i, (X_batch, y_batch) in enumerate(val_dataloader):

                outputs = model(X_batch.unsqueeze(1).to('cuda:0'))

                _, predicted = torch.max(outputs, 1)

                val_pred += list(predicted.cpu())
                val_true += [y_batch]
                val_accuracy += torch.sum(predicted == y_batch.to('cuda:0')).item()

                val_num_samples += len(X_batch)

            avg_accuracy = val_accuracy / val_num_samples


            if log:
                print(f"Accuracy: {avg_accuracy:.4f}")

            val_y_true = []
            for q in val_true:
                val_y_true += q

            val_y_pred = []
            for q in val_pred:
                val_y_pred += [q.item()]

            tn, fp, fn, tp = confusion_matrix(val_y_true, val_y_pred).ravel()
            false_positive_rate = fp / (fp + tn)
            false_negative_rate = fn / (tp + fn)
            true_negative_rate = tn / (tn + fp)
            false_discovery_rate = fp/ (tp + fp)
            recall = recall_score(val_y_true, val_y_pred)
            precision = precision_score(val_y_true, val_y_pred)
            acc = accuracy_score(val_y_true, val_y_pred)
            cohen_kappa = cohen_kappa_score(val_y_true, val_y_pred)
            matthews_corr = matthews_corrcoef(val_y_true, val_y_pred)

            if log:
                writer.add_scalar('Val precision', precision, epoch)
                writer.add_scalar('Val Accuracy', acc, epoch)
                writer.add_scalar('Val recall', f1_macro, epoch)
                print('false_positive_rate : ', false_positive_rate)
                print('false_negative_rate : ', false_negative_rate)
                print('true_negative_rate : ', true_negative_rate)
                print('false_discovery_rate : ', false_discovery_rate)
                print('recall : ', recall)
                print('precision : ', precision)
                print('acc : ', acc)
                print('cohen_kappa : ', cohen_kappa)
                print('matthews_corr : ', matthews_corr)
                print("#######################################")
                mlflow.log_metric("false_positive_rate", false_positive_rate, step=epoch)
                mlflow.log_metric("false_negative_rate", false_negative_rate, step=epoch)
                mlflow.log_metric("true_negative_rate", true_negative_rate, step=epoch)
                mlflow.log_metric("false_discovery_rate", false_discovery_rate, step=epoch)
                mlflow.log_metric("recall", recall, step=epoch)
                mlflow.log_metric("precision", precision, step=epoch)
                mlflow.log_metric("acc", acc, step=epoch)
                mlflow.log_metric("cohen_kappa", cohen_kappa, step=epoch)
                mlflow.log_metric("matthews_corr", matthews_corr, step=epoch)

    # End the MLflow run
    mlflow.end_run()
    writer.close()




if __name__ == '__main__':
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Simple PyTorch Model Training')
    parser.add_argument('--data_path', type=str, default='', help='Path to the data directory')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--version_data', type=int, default=1, help='version of data we trained on')
    parser.add_argument('--window_size', type=int, default=14, help='window of candles')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')

    # Parse the command-line arguments
    args = parser.parse_args()
    train_x = torch.load(f'{args.data_path}X_train_V{args.version_data}.pt')
    train_y = torch.load(f'{args.data_path}y_train_V{args.version_data}.pt')
    val_x = torch.load(f'{args.data_path}X_val_V{args.version_data}.pt')
    val_y = torch.load(f'{args.data_path}y_val_V{args.version_data}.pt')
    train_dataset = TensorDataset(train_x, train_y)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = TensorDataset(val_x, val_y)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size , shuffle=True)
    print(args.window_size)
    model = IntegerTransformer3(num_embeddings=args.window_size, embedding_dim=9, num_heads=4, num_layers=2)
    writer = SummaryWriter()

    # Call the training function with the provided arguments
    train_model(model, mlflow, writer , train_dataloader , val_dataloader , num_epochs = args.num_epochs, learning_rate = args.learning_rate)


    torch.save(model.state_dict(), f'modelV{args.version_data}.pth')