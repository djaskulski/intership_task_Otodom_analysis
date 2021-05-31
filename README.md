## intership_task_Otodom_analysis

# LIQUIDITY TASK
Zadanie polega na obliczeniu wskaźnika liquidity, który pozwala nam dużo lepiej zrozumieć czy
dane ogłoszenia cieszą się popularnością ze strony poszukujących.
W oparciu o to prosimy o przygotowanie pełnej analizy z użyciem dostępnych danych.
Samo zadanie dzieli się na 2 części:

### Część techniczna:
1. Przygotowanie zapytań SQL, które pozwalają na obliczenie liquidity - może być w formie
osobnego pliku (.sql) lub jako część skryptu w pythonie/R. Skonfigurowanie połączenia
do bazy danych (mysql, postgesql, etc.) nie jest konieczne, ale mile widziane.
2. Przygotowanie również w Pythonie/R obliczenia liquidity dla wszystkich użytkowników,
czyli chcemy dostać listę z informacją ile wynosi dokładnie liquidity dla każdego
użytkownika.

### Część analityczna:
1. Proszę o przygotowanie pełnej analizy danych, które zostały przesłane, wraz z
odpowiedziami na poniższe pytania
a. Jakie dostrzegasz różnice pomiędzy segmentami w zakresie danych, które masz
dostępne (w tym liquidity)?
b. Co wg Ciebie może mieć wpływ na wyższy lub niższy poziom liquidity?

### Forma:
1. Preferowany Jupyter/R Markdown do analizy
2. Skrypty mogą być w oddzielnych plikach, albo jako część notebooka w zależności od
wybranych metod
3. Ostateczne wyniki i najważniejsze wnioski proszę przedstawić w formie prezentacji (np.
Google slides)

### Jak obliczyć liquidity:
Liquidity rozumiane będzie jako % ogłoszeń, które otrzymały co najmniej 1 odpowiedź
(telefoniczną lub mailową) w okresie 7 dni (wliczamy w to także dzień 0 - dzień dodania
ogłoszenia)

### Przykład:
Użytkownik dodał 1 kwietnia do serwisu 10 ogłoszeń w dniach od 1 do 7 kwietnia otrzymał odpowiedzi na 6 ogłoszeń.
W dniu 2 kwietnia dodał kolejne 5 ogłoszeń i otrzymał odpowiedzi na wszystkie z nich w czasie
7 dni od pojawienia się tych ofert w serwisie
Liquidity wynosi (6+5)/(10+5) = 73%

### Dane, które mamy do dyspozycji:
1. Data_ads - tutaj znajdują się informacje o ogłoszeniach
2. Data_replies - informacje o odpowiedzi per ogłoszenie per dany dzień
3. Data_categories - mapowanie do drzewa kategorii
4. Data_segments - mapowanie do segmentacji dla każdego użytkownika

Nazwy kolumn dla data_ads
● date
● user_id
● ad_id
● category_id
● params

Nazwy kolumn dla data_replies
● date
● user_id
● ad_id
● mails
● phones

Nazwy kolumn dla data_segmentation
● user_id
● segment

Nazwy kolumn dla data_categories
● category_id
● category_name
