import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
from boruta import BorutaPy
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFE
from sklearn.preprocessing import FunctionTransformer


class DataProcessor:
    def __init__(self, input_data: pd.DataFrame):
        self.input_data = input_data
        self.le = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_names = None

    def preprocessing(self):
        """
        Preprocess the input data
        Get the feature names, remove labels, split the data into testing
        and training.

        The data is then normalized.
        """
        # only select the column with numbers
        numeric_only = self.input_data.select_dtypes(include=['number'])
        col_info = self.input_data[numeric_only.columns]

        self.feature_names = numeric_only.columns.tolist()

        # values of the column label
        input_data_label = self.input_data['Label'].values

        # transform labels to numerical values
        encoded_test = self.le.fit_transform(input_data_label)

        # 33% of data used for testing
        # random state = 42 for reproducibility
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            col_info, encoded_test, test_size=0.33, random_state=42)
        
        self._scale_data()

    def set_scaler_method(self, method: str):
        """
        Set the scaler method to use
        :param method: "standard", "minmax", "log"
        if function not run, automatically assume standard
        """
        if method == "standard":
            self.scaler = StandardScaler()
        elif method == "minmax":
            self.scaler = MinMaxScaler()
        elif method == "log":
            self.scaler = FunctionTransformer(np.log1p, validate = True)
        elif isinstance(method, object):
            self.scaler = method
        else:
            print("Invalid scaler method, Defult is standard")

    def _scale_data(self):
        """
        Normalize the data according to the selected method
        Plot the Curve before and after normalization
        """
        plt.figure(figsize=(12, 6))
        sns.kdeplot(self.X_train.mean(), bw_adjust=0.5)  # Adjust bw_adjust for smoothing
        plt.title('Curve of Column Means before Normalization')
        plt.xlabel('Mean Value')
        plt.ylabel('Density')
        plt.savefig('PNG_output/before_normalization', format="png")

        plt.show()

        self.scaler.fit(self.X_train)
        self.X_train_scaled = self.scaler.transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        normalized_df = pd.DataFrame(self.X_train_scaled, columns = self.X_train.columns)
        plt.figure(figsize=(12, 6))
        sns.kdeplot(normalized_df.mean(), bw_adjust=0.5)  # Adjust bw_adjust for smoothing
        plt.title('Curve of Column Means after Normalization')
        plt.xlabel('Mean Value')
        plt.ylabel('Density')
        plt.savefig('PNG_output/after_normalization', format="png")

        plt.show()

    def feature_selection(self, feature_method, number = None):
        """
        :param feature_method: "RFE", "FeatureImportance", "Boruta", "PCA"
        :param number: Only relevant if feature if RFE, Select number of 
        features to include
        """
        # Initialize the RandomForestClassifier
        self.rf = RandomForestClassifier(n_jobs=-1, 
                                         class_weight='balanced',
                                         n_estimators=50,
                                         max_depth = None,
                                         random_state=42)
        
        # Fit the model first
        self.rf.fit(self.X_train_scaled, self.y_train)

        if feature_method == "RFE":
            self._apply_rfe(number)
        elif feature_method == "FeatureImportance":
            self._apply_FI()
        elif feature_method == "Boruta":
            self._apply_boruta()
        elif feature_method == "PCA":
            self._apply_pca()
        
        # Refit again using the filtered/selected data
        self.rf.fit(self.X_train_selected, self.y_train)

    def _apply_pca(self):
        """
        Use PCA to select for the features to used in the model
        """
        pca = PCA()
        pca.fit(self.X_train_scaled)

        explained_variance_ratio = pca.explained_variance_ratio_
        # Calculate cumulative explained variance
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(1, len(explained_variance_ratio) + 1), 
                 explained_variance_ratio, 
                 marker='o', 
                 linestyle='--')
        plt.title('Explained Variance Ratio by Principal Component')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.xticks(np.arange(1, len(explained_variance_ratio) + 1))
        plt.grid(True)
        plt.show()

        # Plot cumulative explained variance plot
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(1, len(cumulative_variance_ratio) + 1), 
                 cumulative_variance_ratio, 
                 marker='o', 
                 linestyle='--')
        plt.title('Cumulative Explained Variance Ratio by Principal Component')
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.axhline(y=0.95, color='r', linestyle='--', label='95% Threshold')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Find number of component meet 95% threshold
        num_components = np.argmax(cumulative_variance_ratio >= 0.95) + 1
        self.feature = PCA(n_components=num_components)

        self._fit_n_transform()

    def _apply_rfe(self, number):
        """
        :param number: Number of features to include
        Perform RFE for feature select
        """
        self.feature = RFE(self.rf, n_features_to_select = number)
        self._fit_n_transform()

    def _apply_FI(self):
        """
        Apply Feature Importance to select the feature
        """
        importance = self.rf.feature_importances_
        sorted_indicies = importance.argsort()[::-1]

        selected_indicies = sorted_indicies[importance[sorted_indicies] > 0]

        # transform the data
        self.X_train_selected = self.X_train_scaled[:, selected_indicies]
        self.X_test_selected = self.X_test_scaled[:, selected_indicies]

    def _apply_boruta(self):
        """
        Apply Boruta method to filter and select for features to include
        """
        self.feature = BorutaPy(estimator = self.rf,
                                n_estimators = "auto",
                                verbose = 2,
                                random_state = 42)
        self._fit_n_transform()
        
    def _fit_n_transform(self):
        """
        Fit the training data according to feature selection method.
        Transform the training and the testing data to filter only desired
        features
        """
        self.feature.fit(self.X_train_scaled, self.y_train)

        self.X_train_selected = self.feature.transform(self.X_train_scaled)
        self.X_test_selected = self.feature.transform(self.X_test_scaled)

    def feature_selected(self):
        """
        Return the Number of feature selected in the model, and the list
        of features that were selected in the model
        """
        number_of_features = 0
        features_selected = []
        for feature, support in list(zip(self.feature_names,
                                         self.feature.support_)):
            if support: # Count number of features selected by the boruta
                number_of_features += 1
                features_selected.append(feature)
        
        self.features_selected = features_selected
        return number_of_features, features_selected
            
    def model_prediction(self):
        """
        Run the model to predict the accuracy of the model. Return the Accuracy
        of the model and the classification report
        """

        # Predict on test set and print accuracy score
        self.y_pred = self.rf.predict(self.X_test_selected)
        accuracy = accuracy_score(self.y_test, self.y_pred), 
        report = classification_report(self.y_test, self.y_pred)

        return accuracy, report

    def visualize_importance(self, file_path):
        """
        Visualize the importance of the selected feature in a graph.
        """
        # feature_selected function must be run before the graph can be plotted
        _, _ = self.feature_selected()

        importance = self.rf.feature_importances_
        
        # Sort the features by importance in descending order
        sorted_indices = importance.argsort()[::-1]
        non_zero_indices = sorted_indices[importance[sorted_indices] > 0]

        # Visualize feature importances
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(non_zero_indices)), 
                importance[non_zero_indices])
        plt.xticks(range(len(non_zero_indices)), 
                   [self.features_selected[i] for i in non_zero_indices], 
                   rotation=90)
        plt.xlabel("Features")
        plt.ylabel("Feature Importance")
        plt.title("Feature Importances from the selected method")
        plt.tight_layout()
        
        # Save the plot as a PNG file
        plt.savefig(file_path, format="png")

        # Show the plot
        plt.show()

    def plot_confusion_matrix(self, file_path):
        """
        Plot the confusion matrix.
        """
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(cm, annot=True)

        # Set labels and Title
        plt.title("Confusion Matrix")

        # Save the heatmap as PNG file
        plt.savefig(file_path, format="png")

        # Show the plot
        plt.show()

    def neural_network_model(self, epochs_number = 10000):
        """
        param: epochs_number = int. Number of training loop.
        Establish a neural network model from our inputted training data.
        """
        X_train_tensor = torch.tensor(self.X_train_scaled, dtype=torch.float32)
        y_train_tensor = torch.tensor(self.y_train, dtype=torch.long)

        input_size = self.X_train_scaled.shape[1]
        hidden_size = 64 # Alter later
        num_classes = len(set(self.y_train))

        self.model = NeuralNetwork(input_size, hidden_size, num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.01)

        train_losses = []
        train_accuracies = []

        # Training Loop
        epochs = epochs_number

        for epoch in range(epochs):
            self.model.train()
            y_pred = self.model(X_train_tensor)
            loss = criterion(y_pred, y_train_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            _,predicted = torch.max(y_pred, 1)
            train_accuracies.append(
                accuracy_score(y_train_tensor.numpy(), predicted.numpy()))

            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        
        self.model.eval()

        self.train_losses = train_losses
        self.train_accuracies = train_accuracies
    
    def test_nnm(self):
        """
        Test the neural network model established on the testing dataset.
        Return the accuracy of the NNM.
        """
        X_test_tensor = torch.tensor(self.X_test_scaled, dtype=torch.float32)
        y_test_tensor = torch.tensor(self.y_test, dtype=torch.long)

        with torch.no_grad():
            y_pred = self.model(X_test_tensor)
            _, predicted_classes = torch.max(y_pred, 1)
            accuracy = accuracy_score(y_test_tensor.numpy(), 
                                      predicted_classes.numpy())
    
        return accuracy
    
    def plot_nnm_result(self, file_path):
        """
        Plot the result of the NNM training into 2 different plot
        The loss curve and the accuracy curve.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()

        plt.savefig(file_path + "training_loss_curve.png", format="png")
        plt.show()
    
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_accuracies, label='Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Curve')
        plt.legend()

        plt.savefig(file_path + "Accuracy_curve.png", format="png")
        plt.show()
    
    def predict_eval(self, unknown_data):
        
        unknown_label = unknown_data['Label'].values
        self.y_test = self.le.fit_transform(unknown_label)

        # Filter the feature to features used to scale model
        unknown_data_filtered = unknown_data[self.feature_names]

        # Scale the unknown data using the established scaled
        self.X_test_scaled = self.scaler.transform(unknown_data_filtered)

        # Transform the unknown data to retain feature selected
        unknown_data_transformed = self.feature.transform(self.X_test_scaled)      

        # Use the model to predict our unknown data
        self.y_pred = self.rf.predict(unknown_data_transformed)
        
        accuracy = accuracy_score(self.y_test, 
                                  self.y_pred)
        
        report = classification_report(self.y_test,
                                       self.y_pred)

        return accuracy, report



class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, num_classes)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x
    

if __name__ == "__main__":
    # Load the data here
    
    dna_meth = pd.read_csv("G13_kidney_dna-meth.csv")
    gene_expr = pd.read_csv("G13_kidney_gene-expr.csv")
    mystery_dna_meth = pd.read_csv("mystery_dna-meth.csv")
    mystery_gene_expr = pd.read_csv("mystery_gene-expr.csv")

    meth_model = DataProcessor(dna_meth)
    gene_expr_model = DataProcessor(gene_expr)

    # Set the model to normalize using standard
    meth_model.set_scaler_method("standard")
    gene_expr_model.set_scaler_method("standard")
    
    # Preprocess the data
    meth_model.preprocessing()
    gene_expr_model.preprocessing()

    # Use Boruta Method for feature selection
    meth_model.feature_selection("Boruta")
    gene_expr_model.feature_selection("Boruta")

    # Number of features selected by the feature selection methods
    meth_feat_num, meth_feat_list = meth_model.feature_selected()
    gene_feat_num, gene_feat_list = gene_expr_model.feature_selected()

    # Predict the accuracy of the model
    meth_acc, meth_report = meth_model.model_prediction()
    gene_acc, gene_report = gene_expr_model.model_prediction()

    # Train the neural Network model
    meth_model.neural_network_model()
    gene_expr_model.neural_network_model()

    meth_nnm_acc = meth_model.test_nnm()
    gene_nnm_acc = gene_expr_model.test_nnm()

    print("Number of features selected in DNA-methylation data:", meth_feat_num)
    print("Accuracy of Random Forest Model on DNA-methylation Data:", meth_acc)
    print("Classification report on RF on DNA methylation data: \n", 
          meth_report)
    print("Accuracy of Neural Network Model on DNA methylation Data:", 
          meth_nnm_acc)
    print("\n")
    print("Number of features selected in Gene-expression data:", gene_feat_num)
    print("Accuracy of Random Forest Model on Gene-expression Data:", gene_acc)
    print("Classification report on RF on Gene-expression Data:", gene_report)
    print("Accuracy of Neural Network Model on Gene-expression data:",
          gene_nnm_acc)

    # Visualization of plot
    print("Plot for DNA-methylation Data")
    meth_model.visualize_importance("PNG_output/DNA_meth_boruta_features.png")
    meth_model.plot_confusion_matrix("PNG_output/DNA_meth_CM.png")
    meth_model.plot_nnm_result("PNG_output/DNA_meth_")

    print("Plot for Gene-expression data")
    gene_expr_model.visualize_importance(
        "PNG_output/Gene_expr_boruta_features.png")
    gene_expr_model.plot_confusion_matrix("PNG_output/Gene_expr_CM.png")
    gene_expr_model.plot_nnm_result("PNG_output/Gene_expr_")

    # Run model on Unknown Data
    un_meth_acc, un_meth_report = meth_model.predict_eval(mystery_dna_meth)
    un_expr_acc, un_expr_report = gene_expr_model.predict_eval(mystery_gene_expr)
    un_nnm_meth_acc = meth_model.test_nnm()
    un_nnm_expr_acc = gene_expr_model.test_nnm()

    print("Accuracy of RF model on unknown DNA-methylation Data:", un_meth_acc)
    print("Classification Report of RF on unknown DNA-meth Data: \n",
          un_meth_report)
    print("Accuracy of NNM on unknown DNA-methylation Data:", un_nnm_meth_acc)
    
    print("Accuracy of RF model on unknown Gene Expression Data:", un_expr_acc)
    print("Classification Report of RF on unknown Gene Expression Data: \n",
          un_expr_report)
    print("Accuracy of NNM on unknown Gene Expression Data: \n",
          un_nnm_expr_acc)
    
    print("Plot for Unknown DNA-methylation Data")
    meth_model.plot_confusion_matrix("PNG_output/unknown_DNA_meth_CM.png")

    print("Plot for unknown Gene expression Data")
    gene_expr_model.plot_confusion_matrix("PNG_output/unknown_Gene_expr_CM.png")


    

    