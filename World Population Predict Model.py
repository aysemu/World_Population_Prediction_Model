import sys  # For system-level operations (e.g., application exit)
import pandas as pd  # Powerful data manipulation and analysis library
import numpy as np  # Efficient numerical computing
import matplotlib.pyplot as plt  # Plotting and visualization library

# PyQt5 modules for GUI components and styling
from PyQt5 import QtWidgets, QtGui, QtCore

# Scikit-learn: Machine learning tools
from sklearn.preprocessing import PolynomialFeatures  # Generate polynomial features
from sklearn.linear_model import Ridge  # Ridge regression (L2 regularization)
from sklearn.pipeline import Pipeline  # Combine preprocessing and model in a pipeline
from sklearn.model_selection import train_test_split, GridSearchCV  # Data splitting and hyperparameter tuning
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error  # Performance metrics

# Load the population dataset
df = pd.read_csv("population_total_long.csv")

# Extract and sort all unique country names from the dataset
all_countries = sorted(df["Country Name"].dropna().unique())

# Main GUI Application
class PopulationApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        # Set the title and size of the main window
        self.setWindowTitle("🌍 Population Prediction Tool")
        self.setGeometry(100, 100, 700, 500)
        self.setFixedSize(800, 600)

        # Set background color of the window using CSS syntax
        self.setStyleSheet("background-color: #d5d8db;")

        # Call the method to build the user interface
        self.initUI()

    def initUI(self):
        # Main vertical layout to hold everything in the window
        main_layout = QtWidgets.QVBoxLayout()

        # Add a title label at the top of the window
        title = QtWidgets.QLabel("🌍 Population Prediction & Analysis")
        title.setFont(QtGui.QFont("Arial", 20, QtGui.QFont.Bold))
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet("color: #2c3e50; margin-bottom: 20px;")
        main_layout.addWidget(title)
        
        # Country Prediction Section
        country_group = QtWidgets.QGroupBox("🔍 Country Prediction")
        country_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: #34495e;
                font-size: 12pt;
            }
            QLabel {
                font-size: 10pt;
            }
            QComboBox {
                padding: 2px;
                font-size: 10pt;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                font-weight: bold;
                font-size: 10pt;
                border-radius: 4px;
            }
        """)

        layout1 = QtWidgets.QVBoxLayout()
        layout1.setSpacing(3)
        layout1.setContentsMargins(10, 8, 10, 8)

        # Label for dropdown
        self.label = QtWidgets.QLabel("Select a country 🌍:")
        self.label.setContentsMargins(0, 120, 0, 0)
        layout1.addWidget(self.label)

        # Dropdown (ComboBox) with country names
        self.combo = QtWidgets.QComboBox()
        self.combo.addItems(all_countries)
        self.combo.setFixedHeight(26)
        self.combo.setStyleSheet("""
            QComboBox {
                background-color: #ecf0f1;
                selection-background-color: #cde0fa;
                font-size: 10pt;
            }
            QComboBox QAbstractItemView {
                selection-background-color: #2ecc71;
                font-size: 10pt;
            }
        """)
        layout1.addWidget(self.combo)

        # Button to trigger prediction
        self.button = QtWidgets.QPushButton("Generate Prediction")
        self.button.setFixedHeight(50)
        self.button.setFixedWidth(200)
        layout1.addWidget(self.button, alignment=QtCore.Qt.AlignHCenter)

        country_group.setLayout(layout1)

        # Global & Comparative Analysis Section
        analysis_group = QtWidgets.QGroupBox("📊 Global & Comparative Analysis")
        analysis_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: #34495e;
                font-size: 12pt;
            }
        """)

        layout2 = QtWidgets.QVBoxLayout()

        # Define buttons for various analysis features
        self.analysis_button = QtWidgets.QPushButton("Global Analysis")
        self.top5_button = QtWidgets.QPushButton("Top 5 Populated Countries")
        self.bottom5_button = QtWidgets.QPushButton("Bottom 5 Populated Countries")
        self.stats_button = QtWidgets.QPushButton("Country Statistics")
        self.compare_button = QtWidgets.QPushButton("Compare Two Countries")

        # Apply consistent styling to all buttons
        for btn in [self.analysis_button, self.top5_button, self.bottom5_button, self.stats_button, self.compare_button]:
            btn.setStyleSheet("""
                background-color: #3d5a80;
                color: white;
                font-size: 16px;
                font-weight: bold;
                margin: 2px;
                min-width: 100px;
                min-height: 40px;
            """)
            layout2.addWidget(btn)

        analysis_group.setLayout(layout2)

        # Layout Connection

        # Combine the two group sections side-by-side
        group_layout = QtWidgets.QHBoxLayout()
        group_layout.addWidget(country_group, 1)
        group_layout.addWidget(analysis_group, 2)

        # Add everything to the main layout
        main_layout.addLayout(group_layout)

        # Button Event Connections

        self.button.clicked.connect(self.predict_and_plot)
        self.analysis_button.clicked.connect(self.show_analysis)
        self.top5_button.clicked.connect(self.show_top5_countries)
        self.bottom5_button.clicked.connect(self.show_bottom5_countries)
        self.stats_button.clicked.connect(self.show_country_stats)
        self.compare_button.clicked.connect(self.compare_two_countries)

        # Finalize layout
        self.setLayout(main_layout)

          # Create a vertical layout for analysis buttons
        layout2 = QtWidgets.QVBoxLayout()

        # Define buttons for various data analysis functions
        self.analysis_button = QtWidgets.QPushButton("Global Analysis")              # Displays global population trends
        self.top5_button = QtWidgets.QPushButton("Top 5 Populated Countries")        # Shows top 5 most populated countries
        self.bottom5_button = QtWidgets.QPushButton("Bottom 5 Populated Countries")  # Shows bottom 5 least populated countries
        self.stats_button = QtWidgets.QPushButton("Country Statistics")              # Displays statistics for a selected country
        self.compare_button = QtWidgets.QPushButton("Compare Two Countries")         # Compares population between two selected countries

        # Apply consistent styling to all analysis buttons
        for btn in [self.analysis_button, self.top5_button, self.bottom5_button, self.stats_button, self.compare_button]:
            btn.setStyleSheet("""
            background-color: #3d5a80;
            color: white;
            font-size: 16px;
            font-weight: bold;
            margin: 2px;
            min-width: 100px;
            min-height: 40px;
            """)

        # Add all buttons to the vertical layout
        layout2.addWidget(self.analysis_button)
        layout2.addWidget(self.top5_button)
        layout2.addWidget(self.bottom5_button)
        layout2.addWidget(self.stats_button)
        layout2.addWidget(self.compare_button)

        # Set the layout for the analysis group box
        analysis_group.setLayout(layout2)

        # Create horizontal layout to position country selection and analysis buttons side by side
        group_layout = QtWidgets.QHBoxLayout()
        group_layout.addWidget(country_group, 1)     # Takes 1 unit of horizontal space
        group_layout.addWidget(analysis_group, 2)    # Takes 2 units of horizontal space for wider display

        # Add this group layout to the main layout of the window
        main_layout.addLayout(group_layout)

        # Connect buttons to their corresponding methods
        self.button.clicked.connect(self.predict_and_plot)
        self.analysis_button.clicked.connect(self.show_analysis)
        self.top5_button.clicked.connect(self.show_top5_countries)
        self.bottom5_button.clicked.connect(self.show_bottom5_countries)
        self.stats_button.clicked.connect(self.show_country_stats)
        self.compare_button.clicked.connect(self.compare_two_countries)

        # Set the full layout to the main window
        self.setLayout(main_layout)


        # Population prediction logic with detailed visualization
    def predict_and_plot(self):
            country = self.combo.currentText()  # Get selected country name from ComboBox
            data = df[df["Country Name"] == country].dropna(subset=["Year", "Count"])  # Filter data and remove missing values

            if data.empty:
                QtWidgets.QMessageBox.warning(self, "Error", "No data found.")  # Show error if no data found
                return

            # Prepare features (X) and target variable (y)
            X = data["Year"].values.reshape(-1, 1)  # Feature: year
            y_raw = data["Count"].values           # Actual population
            y = np.log1p(y_raw)                    # Log transform to stabilize variance

            # Split data into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

            # Build ML pipeline with polynomial features and Ridge regression
            pipeline = Pipeline([
                ("poly", PolynomialFeatures()),   # Generates polynomial features (x, x^2, ...)
                ("ridge", Ridge())                # Applies Ridge regression
            ])

            # Define parameter grid for model selection
            param_grid = {
                "poly__degree": [1, 2, 3],        # Try linear, quadratic, cubic models
                "ridge__alpha": [0.1, 10, 100, 1000]  # Regularization strengths
            }

            # Use GridSearchCV to find best model parameters via cross-validation
            search = GridSearchCV(pipeline, param_grid, scoring="neg_mean_squared_error", cv=3)
            search.fit(X_train, y_train)
            best_model = search.best_estimator_   # Extract the best model

            # Make predictions on training and test sets
            y_pred_train = np.expm1(best_model.predict(X_train))  # Inverse log transform
            y_pred_test = np.expm1(best_model.predict(X_test))
            y_train_original = np.expm1(y_train)
            y_test_original = np.expm1(y_test)

            # Evaluate model performance using R², MAE, and RMSE
            train_r2 = r2_score(y_train_original, y_pred_train)
            test_r2 = r2_score(y_test_original, y_pred_test)
            train_mae = mean_absolute_error(y_train_original, y_pred_train)
            train_rmse = np.sqrt(mean_squared_error(y_train_original, y_pred_train))
            test_mae = mean_absolute_error(y_test_original, y_pred_test)
            test_rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_test))

            # Predict future population from 1970 to 2035
            future_years = np.arange(1970, 2036).reshape(-1, 1)
            future_preds = np.expm1(best_model.predict(future_years))  # Future predictions (inverse log)

            # Plot real vs predicted values
            plt.figure(figsize=(10, 5))
            plt.scatter(X.flatten(), y_raw, color="orange", label="Actual")             # Actual data points
            plt.plot(future_years.flatten(), future_preds, color="blue", label="Prediction")  # Prediction curve
            plt.title(f"{country} - Population Prediction")
            plt.xlabel("Year")
            plt.ylabel("Population")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show(block=True)

            # Display performance results and predictions in a popup message
            msg = f"📊 {country} - Model Performance (degree={search.best_params_['poly__degree']}, alpha={search.best_params_['ridge__alpha']}):\n"
            msg += f"Train R²: {train_r2:.4f} | MAE: {train_mae:,.2f} | RMSE: {train_rmse:,.2f}\n"
            msg += f"Test R²: {test_r2:.4f} | MAE: {test_mae:,.2f} | RMSE: {test_rmse:,.2f}\n"

            # List future prediction results year by year
            msg += "\n📈 2016-2035 Predictions:\n"
            for year, pred in zip(future_years.flatten(), future_preds):
                if year >= 2016:
                    msg += f"{int(year)}: {int(pred):,}\n"

            # Warn user if the model might be overfitting
            if abs(train_r2 - test_r2) > 0.05:
                msg += "\n⚠️ The model may be overfitting.\n"

            # Show final message with metrics and predictions
            QtWidgets.QMessageBox.information(self, "Model Results", msg)

 # Function to show global population analysis and predictions
    def show_analysis(self):
        df_cleaned = df.dropna(subset=["Year", "Count"])
        world = df_cleaned.groupby("Year")["Count"].sum().reset_index()

        X = world["Year"].values.reshape(-1, 1)
        y_raw = world["Count"].values
        y = np.log1p(y_raw)

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

        # Create a pipeline with PolynomialFeatures and Ridge regression
        pipeline = Pipeline([
            ("poly", PolynomialFeatures()),
            ("ridge", Ridge())
        ])
        param_grid = {
            "poly__degree": [1, 2, 3],
            "ridge__alpha": [0.1, 10, 100, 1000]
        }
        # Hyperparameter search using cross-validation
        search = GridSearchCV(pipeline, param_grid, scoring="neg_mean_squared_error", cv=3)
        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        # Predictions for training and testing sets
        y_pred_train = np.expm1(best_model.predict(X_train))
        y_pred_test = np.expm1(best_model.predict(X_test))
        y_train_original = np.expm1(y_train)
        y_test_original = np.expm1(y_test)

        # Predict future values from 1960 to 2035
        future_years = np.arange(1960, 2036).reshape(-1, 1)
        future_preds = np.expm1(best_model.predict(future_years))

        # Plot actual vs predicted values
        plt.figure(figsize=(10, 5))
        plt.scatter(X.flatten(), y_raw, color="orange", label="Actual")
        plt.plot(future_years.flatten(), future_preds, color="blue", label="Prediction")
        plt.title("World Population with Forecast (1960–2035)")
        plt.xlabel("Year")
        plt.ylabel("Total Population")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show(block=False)

        # Evaluation metrics
        train_r2 = r2_score(y_train_original, y_pred_train)
        test_r2 = r2_score(y_test_original, y_pred_test)
        train_mae = mean_absolute_error(y_train_original, y_pred_train)
        test_mae = mean_absolute_error(y_test_original, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train_original, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_test))

        # Generate summary message
        msg = f"🌍 World Population Forecast (degree={search.best_params_['poly__degree']}, alpha={search.best_params_['ridge__alpha']}):\n"
        msg += f"Train R²: {train_r2:.4f} | MAE: {train_mae:,.2f} | RMSE: {train_rmse:,.2f}\n"
        msg += f"Test R²: {test_r2:.4f} | MAE: {test_mae:,.2f} | RMSE: {test_rmse:,.2f}\n"
        msg += "\n📈 2016–2035 Predictions:\n"
        for year, pred in zip(future_years.flatten(), future_preds):
            if year >= 2016:
                msg += f"{int(year)}: {int(pred):,}\n"

        # Warn if model is likely overfitting
        if abs(train_r2 - test_r2) > 0.05:
            msg += "\nThe model may be overfitting.\n"

      # Display the summary message in a popup
        QtWidgets.QMessageBox.information(self, "Global Prediction", msg)

# Function to show top 5 most populated countries over time
    def show_top5_countries(self):
        df_cleaned = df.dropna(subset=["Year", "Count"])
        latest_year = df_cleaned["Year"].max()
        latest_data = df_cleaned[df_cleaned["Year"] == latest_year]
        top5 = latest_data.nlargest(5, "Count")["Country Name"]
        top5_data = df_cleaned[df_cleaned["Country Name"].isin(top5)]
    
        plt.figure(figsize=(10, 5))
        for country in top5:
            subset = top5_data[top5_data["Country Name"] == country]
            plt.plot(subset["Year"], subset["Count"], label=country)
        plt.title(f"Top 5 Most Populated Countries ({latest_year})")
        plt.xlabel("Year")
        plt.ylabel("Population")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show(block=False)
    
    # Function to show bottom 5 least populated countries over time
    def show_bottom5_countries(self):
        df_cleaned = df.dropna(subset=["Year", "Count"])
        latest_year = df_cleaned["Year"].max()
        latest_data = df_cleaned[df_cleaned["Year"] == latest_year]
        bottom5 = latest_data.nsmallest(5, "Count")["Country Name"]
        bottom5_data = df_cleaned[df_cleaned["Country Name"].isin(bottom5)]

        plt.figure(figsize=(10, 5))
        for country in bottom5:
            subset = bottom5_data[bottom5_data["Country Name"] == country]
            plt.plot(subset["Year"], subset["Count"], label=country)
        plt.title(f"Bottom 5 Least Populated Countries ({latest_year})")
        plt.xlabel("Year")
        plt.ylabel("Population")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show(block=False)

    # Function to show statistics of the selected country
    def show_country_stats(self):
        country = self.combo.currentText()
        data = df[df["Country Name"] == country].dropna(subset=["Year", "Count"])

        if data.empty:
            QtWidgets.QMessageBox.warning(self, "Error", "No data found.")
            return

        msg = f"📌 {country} Statistics:\n"
        msg += f"Year Range: {int(data['Year'].min())} - {int(data['Year'].max())}\n"
        msg += f"Average Population: {data['Count'].mean():,.0f}\n"
        msg += f"Min Population: {data['Count'].min():,.0f} ({int(data.loc[data['Count'].idxmin(), 'Year'])})\n"
        msg += f"Max Population: {data['Count'].max():,.0f} ({int(data.loc[data['Count'].idxmax(), 'Year'])})\n"

        QtWidgets.QMessageBox.information(self, "Country Statistics", msg)

        # Show bar chart of country's population over time
        plt.figure(figsize=(10, 5))
        plt.bar(data["Year"], data["Count"], color="skyblue")
        plt.title(f"{country} - Population by Year")
        plt.xlabel("Year")
        plt.ylabel("Population")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show(block=False)

    def compare_two_countries(self):
       dialog = QtWidgets.QDialog(self)
       dialog.setWindowTitle("Compare Two Countries")
       layout = QtWidgets.QVBoxLayout()    
    
       # Create two combo boxes for selecting countries
       combo1 = QtWidgets.QComboBox()
       combo1.addItems(all_countries)
       combo2 = QtWidgets.QComboBox()
       combo2.addItems(all_countries)  
    
       layout.addWidget(QtWidgets.QLabel("Select first country:"))
       layout.addWidget(combo1)
       layout.addWidget(QtWidgets.QLabel("Select second country:"))
       layout.addWidget(combo2)    
    
       # Create compare button
       compare_btn = QtWidgets.QPushButton("Compare")
       compare_btn.setStyleSheet("background-color: #9b59b6; color: white; font-weight: bold;")
       layout.addWidget(compare_btn)   
    
       dialog.setLayout(layout)    
    
       # Define comparison function
       def do_compare():
           c1 = combo1.currentText()
           c2 = combo2.currentText()
           data1 = df[df["Country Name"] == c1].dropna(subset=["Year", "Count"])
           data2 = df[df["Country Name"] == c2].dropna(subset=["Year", "Count"])
           plt.figure(figsize=(10, 5))
           plt.plot(data1["Year"], data1["Count"], label=c1, color="blue")
           plt.plot(data2["Year"], data2["Count"], label=c2, color="red")
           plt.title(f"{c1} vs {c2} - Population Comparison")
           plt.xlabel("Year")
           plt.ylabel("Population")
           plt.grid(True)
           plt.legend()
           plt.tight_layout()
           plt.show(block=False)
           dialog.accept()
    
       # Connect button after function is defined
       compare_btn.clicked.connect(do_compare)
       
       dialog.exec_()
    
if __name__ == "__main__":  # This block runs only if the script is executed directly
    app = QtWidgets.QApplication(sys.argv)  # Create the main QApplication instance required for any PyQt5 app
    win = PopulationApp()  # Instantiate the main window from the PopulationApp class
    win.show()  # Display the main application window
    sys.exit(app.exec_())  # Start the event loop and exit the application when it's closed