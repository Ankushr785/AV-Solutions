#selecting the best lag
lag1 = series_d.shift()
lag2 = lag1.shift()
lag3 = lag2.shift()
lag4 = lag3.shift()
lag5 = lag4.shift()
lag6 = lag5.shift()
lag7 = lag6.shift()

lag_df_d = pd.DataFrame({'series':series_d, 'lag1':lag1, 'lag2':lag2, 'lag3':lag3, 'lag4':lag4, 'lag5':lag5, 'lag6':lag6, 'lag7':lag7})
lag_df_d = lag_df_d.drop(lag_df_d.index[[0, 1, 2, 3, 4, 5, 6]])

comparison_d = []
for i in range(7):
    comparison_d.append(np.corrcoef(lag_df_d.iloc[:, 0], lag_df_d.iloc[:, i+1])[0,1])

for i in range(7):
    if comparison_d[i] == np.max(comparison_d):
        lag = i+1
    else:
        count = 0