ens = joblib.load("ens_mod_reg.joblib")

# Load test data
test = pd.read_csv("mars-private_test-reg.csv")

test.head(50)
# Make predictions
X_test = test
y_pred_ens = ens.predict(X_test)

# Create a DataFrame for predictions and save to CSV
result_df = pd.DataFrame({'Доля сигнала в ВП': y_pred_ens})
result_df.to_csv('pred_ens.csv', index=False)
