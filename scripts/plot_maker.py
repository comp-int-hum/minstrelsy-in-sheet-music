import matplotlib.pyplot as plt

# Years and their respective sums
years = [1800, 1810, 1820, 1830, 1840, 1850, 1860, 1870, 1880, 1890, 1900, 1910, 1920]
totals = [1248, 1136, 13343, 52157, 128411, 204902, 283098, 152148, 140437, 215966, 211507, 262179, 71013]

plt.figure(figsize=(10, 6))
plt.bar(years, totals, color='blue')
plt.xlabel('Year')
plt.ylabel('Total Textual Distribution')
plt.title('Total Textual Distribution Over Time')
plt.grid(axis='y')
plt.tight_layout()

plt.show()
