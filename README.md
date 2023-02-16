# Davydov Research of H2O in biological tissues by optical spectroscopy
____
## Описание ноутбуков обработки

- mcml_data_knn_train_test.ipynb – ноутбук для обучения моделей knn для моделированию спектров диффузного отражения
- spectra_modeling_by_knn.ipynb – ноутбук по моделированию спектров диффузного отражения с помощью обученных моделей knn
- inverse_problem_random_forest_mua1.ipynb – ноутбук по обучению модели random forest для определения коэффициента поглощения для первого слоя по коэффициентам отражения.
- inverse_problem_dermis_thickness.ipynb – ноутбук по обучении модели xgboostregressor по определению толщины первого слоя по коэффициентам отражения

## Описание данных моделирования

Моделирование проводилось с помощью метода Монте-Карло (MCML, Monte Carlo Modeling of Light Transport) (https://omlc.org/software/mc/) для двухслойных сред с различной толщиной первого слоя d1 и коэффициентами поглощения и рассеяния каждого слоя 𝜇a1, 𝜇s1, 𝜇a2, 𝜇s2, где 𝜇a1, 𝜇s1 – коэффициенты поглощения и рассеяния для первого слоя, а 𝜇a2, 𝜇s2 для второго, соответственно. Для данных сред были рассчитаны и коэффициенты диффузного отражения R в зависимости от расстояния между детектируемой областью и источником излучения r. Коэффициент диффузного отражения R=R(r) для каждого заданного набора значения 𝜇a1, 𝜇s1, 𝜇a2, 𝜇s2 был рассчитан как отношение числа фотонных пакетов, вышедших из области в форме кольца радиуса r, толщиной dr = 0.005 см и площадью S=2πrdr, к числу фотонных пакетов, вошедших в среду. При этом расстояние r варьировалось в диапазоне от 0.0025 до 1.5 см с равномерным шагом, равным dr = 0.005 см. Для каждого слоя значение показателя поглощения 𝜇ai (i=1,2) варьировалось в диапазоне от 0.1 до 10 см-1 на логарифмической сетке (расчёты были проведены для 20 различных значений в указанном диапазоне), а показателя рассеяния 𝜇si (i=1,2) в диапазоне от 100 до 1000 см-1 (также 20 значений на логарифмической сетке). Помимо этого, варьировалась толщина первого слоя d1 в диапазоне от 0.075 до 0.275 см с шагом 0.025 см, при этом толщина второго слоя d2 было фиксировано и равно 0.5 см. В результате всего было проведено1440000 MCML-моделирований с числом фотонных пакетов 107 каждое.
Данные представляют собой таблицу c, каждая строчка которого есть результат моделирования распространения света в двухслойной среде методом Монте-Карло с заданными значениями 𝜇a1, 𝜇s1, d1 𝜇a2, 𝜇s2 ,d2.Первые 300 столбцов это коэффициенты отражения коэффициент диффузного отражения R=R(r) для каждого заданного набора значений  𝜇a1, 𝜇s1, d1 𝜇a2, 𝜇s2 ,d2 рассчитаные как отношение числа фотонных пакетов, вышедших из области в форме кольца радиуса r, толщиной dr = 0.005 см и площадью S=2*π*r*dr, к числу фотонных пакетов, вошедших в среду, нормированные на площадь данной области. Значения r варьировалось в диапазоне от 0.0025 до 1.5 см с равномерным шагом, равным dr = 0.005 см. Последние 6 столбцов это значения 𝜇a1, 𝜇s1, d1 𝜇a2, 𝜇s2 ,d2 для каждой среды.
