import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv("./data_processed.csv")

# print(df.head())

fontsize = "12"
fontweight = "bold"


# plt.figure(figsize=(14,8))
# plt.plot(df["Date"], df["Close"], color="#607787")
# plt.title("Estadística de Walmart entre fecha y el precio de cierre")
# plt.xlabel("Fecha", fontsize=fontsize, fontweight=fontweight, color="#212121")
# plt.ylabel("Precio de cierre", fontsize=fontsize, fontweight=fontweight, color="#212121")
# plt.xticks(df['Date'][::500], rotation=45)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(10, 6))
# sns.histplot(df['Volume'], bins=50, kde=True, color="#212121")
# plt.title('Distribucion del volumen de proceso o frecuencia')
# plt.xlabel('Volumen')
# plt.ylabel('Frecuencia')
# plt.tight_layout()
# plt.show()

# df['MA50'] = df['Close'].rolling(window=50).mean()
# df['MA200'] = df['Close'].rolling(window=200).mean()
 
# plt.figure(figsize=(12, 6))
# plt.plot(df['Date'], df['Close'], color='b', label='Closing Price')
# plt.plot(df['Date'], df['MA50'], color='r', label='50-Day MA')
# plt.plot(df['Date'], df['MA200'], color='g', label='200-Day MA')
# plt.title('Moving Averages of Walmart Stock Prices')
# plt.xlabel('Date')
# plt.ylabel('Price')
# plt.xticks(df['Date'][::500], rotation=45)
# plt.legend()
# plt.tight_layout()
# plt.show()

df['Daily_Return'] = df['Close'].pct_change()

plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Daily_Return'].rolling(window=30).std(), color='m')
plt.title('Volatility of Walmart Stock Prices')
plt.xlabel('Date')
plt.ylabel('Volatility (30-day Std Dev)')
plt.xticks(df['Date'][::500], rotation=45)
plt.tight_layout()
plt.show()