import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression as lr

data = pd.read_csv("bram.csv", delimiter=';') #veri okuma
print(data) #veriyi görelim

# Sadece sayısal değerleri içeren satırları seçin
data = data[pd.to_numeric(data["Allele_one_RP_count"], errors='coerce').notnull()]



x = data["Allele_one_RP_count"].values 
y = data["Allele_two_RP_count"].values 
x = np.array(x).reshape(2093, 1)  # Doğru boyutlandırma

# Eksik değerleri ortalama ile doldur
imputer_x = SimpleImputer(strategy='mean')
x = imputer_x.fit_transform(x)

# Ref_ID sütunundaki eksik değerleri en sık görülen değerle doldur
imputer_y = SimpleImputer(strategy='most_frequent')
y = imputer_y.fit_transform(y.reshape(2093, 1))

# Ref_ID sütununu sayısal değerlere dönüştürün
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y.ravel())

linearregresyon = lr()
linearregresyon.fit(x, y)

linearregresyon.predict(x)

print('Eğim: \n', linearregresyon.coef_)
print("Y de kesiştiği yer: ", linearregresyon.intercept_)
m = linearregresyon.coef_
b = linearregresyon.intercept_
print("denklem")
print("y=", m, "x+", b)

a = np.arange(25)
plt.scatter(x, y)
plt.plot(a, m * a + b, color='red')
plt.show()

#tahmin deneme
z=int(input("kaç metrekare="))
tahmin=m*z+b
print(tahmin)
plt.scatter(z,tahmin,c="red",marker=">")
plt.scatter(x,y)

plt.show()
print("y=",m,"x+",b)
