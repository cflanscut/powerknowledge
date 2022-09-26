from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'SimHei'

names = [
    'Fan', 'Table Lamp', 'Electric Blanket', 'Air Conditioner', 'Fridge',
    'Clothes Dryer', 'Electric kettle', 'Hair Dryer', 'Rice Cooker', 'Lamp',
    'Oven', 'Phone Charger', 'Microwave', 'Hot Pot', 'Drinking Fountain'
]
names_chinese = [
    '风扇', '台灯', '电热毯', '空调', '冰箱', '干衣机', '电热水壶', '电吹风', '电饭煲', '白炽灯', '微波炉',
    '手机充电器', '微波炉', '电火锅', '饮水机'
]
r1 = np.array([
    4, 2.5, 50, 2324, 20.417, 100.5, 600, 1000.0, 220.0, 2.0, 775, 5, 700, 600,
    420
])
r2 = np.array([
    61.25, 17.5, 101.0, 6972, 45, 2000, 2000, 1800, 1300, 120, 2600, 65, 1100,
    2100, 2200
])

fig = plt.figure(figsize=(16, 9))
ax1 = plt.axes([0.1, 0.1, 0.85, 0.85])

b1 = ax1.barh(names_chinese, r2 - r1, color='skyblue', left=r1)
for i, b in enumerate(b1):
    ax1.annotate(r1[i], (b.get_x(), b.get_height() + i - 1),
                 FontProperties='SimHei')
    ax1.annotate(r2[i], (b.get_x() + b.get_width(), b.get_height() + i - 1),
                 FontProperties='SimHei')
ax1.set_xlabel('功率/W')
plt.semilogx()
plt.grid(axis='x')
matplotlib.rcParams['font.family'] = 'SimHei'
plt.show()
