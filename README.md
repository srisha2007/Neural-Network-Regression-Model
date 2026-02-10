

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

The objective of this experiment is to develop a neural network regression model using a dataset created in Google Sheets with one numeric input and one numeric output. Regression is a supervised learning technique used to predict continuous values. A neural network is chosen because it can effectively learn both linear and non-linear relationships between input and output by adjusting its weights during training.

The model is trained using backpropagation to minimize a loss function such as Mean Squared Error (MSE). During each iteration, the training loss is calculated and updated. The training loss vs iteration plot is used to visualize the learning process of the model, where a decreasing loss indicates that the neural network is learning properly and converging toward an optimal solution.

## Neural Network Model

.
<img width="1115" height="695" alt="546306997-1a3163ce-4b0f-4fa5-9b13-61396699d3a9" src="https://github.com/user-attachments/assets/75c8b8af-fd98-4c8f-91f8-6249d4585e78" />

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: SRISHA
### Register Number: 212224040328
```
#creating model class
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(1, 8)
        self.fc2=nn.Linear(8, 10)
        self.fc3=nn.Linear(10, 1)
        self.relu=nn.ReLU()
        self.history={'loss':[]}

  def forward(self,x):
        x=self.relu(self.fc1(x))
        x=self.relu(self.fc2(x))
        x=self.fc3(x)
        return x

# Initialize the Model, Loss Function, and Optimizer
ai_brain = NeuralNet()
criterion=nn.MSELoss()
optimizer=optim.RMSprop(ai_brain.parameters(), lr=0.001)

#Function to train model
def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):

    for epoch in range(epochs):
      optimizer.zero_grad()
      loss=criterion(ai_brain(X_train),y_train)
      loss.backward()
      optimizer.step()

      ai_brain.history['loss'].append(loss.item())
      if epoch % 200 == 0:
          print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')    

```
## Dataset Information


<img width="274" height="523" alt="546759068-2dc3fa2d-c453-4512-8dc9-6dcbbfce96a7" src="https://github.com/user-attachments/assets/74db5b37-0750-43d4-bd9a-7e12e91d41c5" />

## OUTPUT

<img width="784" height="492" alt="546759137-e3fbdc58-644e-4ee4-a05f-c6e30a8d2eb0" src="https://github.com/user-attachments/assets/48aedf42-984f-42ce-a170-5b3fe277c033" />

Include your plot here

<img width="720" height="572" alt="546759156-87168463-d0c2-4470-ba1d-f35e92192db8" src="https://github.com/user-attachments/assets/91b5b305-4b21-4c8f-8c0b-7566e6dae03e" />

Include your sample input and output here

<img width="793" height="400" alt="546199086-5d566fda-2ea2-4352-a96f-d9b8c625496b" src="https://github.com/user-attachments/assets/616da18c-dd4f-40ba-bd03-bed3eaa11d15" />


## RESULT
Thus the neural network regression model is developed using the given dataset.
