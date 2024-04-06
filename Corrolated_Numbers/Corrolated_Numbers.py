import numpy as np
from sklearn.decomposition import FastICA


def generate_correlated_sequence(n, correlation):
    # تولید دو دنباله‌ی رندوم
    random_sequence_1 = np.random.normal(size=n)
    random_sequence_2 = np.random.normal(size=n)

    # ترکیب خطی دو دنباله‌ی رندوم برای ایجاد همبستگی
    mixed_sequence = correlation * random_sequence_1 + np.sqrt(1 - correlation ** 2) * random_sequence_2

    # اعمال ICA بر روی مخلوط
    ica = FastICA(n_components=2)
    unmixing_matrix = ica.fit_transform(mixed_sequence.reshape(-1, 1))
    independent_sequence = unmixing_matrix[:, 0]  # انتخاب یکی از مولفه‌های مستقل

    # برگرداندن دنباله‌ی مستقل
    return independent_sequence


# مثال استفاده
n = 1000  # تعداد نمونه‌ها
correlation = 0.8  # مقدار همبستگی
correlated_sequence = generate_correlated_sequence(n, correlation)

# چاپ چند عضو از دنباله‌ی مستقل
print(correlated_sequence[:10])
