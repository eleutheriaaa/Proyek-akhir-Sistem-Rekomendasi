# Laporan Proyek Machine Learning - Nathanael Dennis Gunawan

## Project Overview

Ponsel pintar atau cellphone telah menjadi kebutuhan penting dalam kehidupan modern, digunakan untuk komunikasi, bekerja, hiburan, hingga aktivitas produktif lainnya. Banyaknya pilihan ponsel dengan berbagai merek, spesifikasi, dan fitur yang terus berkembang menjadikan proses pemilihan perangkat yang tepat menjadi semakin kompleks bagi konsumen. Informasi yang terlalu banyak (information overload) dapat menyebabkan pengguna kesulitan dalam mengambil keputusan pembelian yang sesuai dengan kebutuhan mereka

Salah satu solusi untuk membantu pengguna dalam memilih ponsel yang sesuai adalah dengan sistem rekomendasi. Sistem ini bekerja dengan menyarankan produk berdasarkan karakteristik atau preferensi pengguna sebelumnya. Salah satu pendekatan yang umum digunakan adalah content based filtering, di mana sistem menganalisis atribut produk seperti merek, sistem operasi, RAM, memori internal, dan harga untuk menemukan kemiripan antar perangkat

**Rumusan Masalah dan Penyelesaian Masalah**  
Memilih cellphone yang sesuai dengan kebutuhan merupakan tantangan yang semakin kompleks seiring berkembangnya teknologi. Maka dari itu diperlukan sistem yang mendukung sistem yang membantu sistem seleksi yang berguna untuk memengaruhi produktivitas dan kepuasan user
Penyelesaian masalah ini penting karena:  
- Efisiensi Pengambilan Keputusan: Banyaknya pilihan di pasar membuat proses pemilihan ponsel menjadi tidak efisien dan membingungkan bagi pengguna awam
- Kesesuaian Kebutuhan: Tidak semua pengguna memahami spesifikasi teknis. Tanpa bantuan sistem rekomendasi, mereka bisa memilih perangkat yang tidak sesuai dengan kebutuhan atau gaya hidup mereka
- Kepuasan Pengguna: Rekomendasi yang tidak tepat dapat menurunkan kepuasan pengguna dan menumbuhkan rasa kekecewaan setelah pembelian

Penyelesaian masalah dengan Sistem Rekomendasi berbasis Content Based Filtering dengan memanfaatkan Machine Learning:  
- Menganalisis Fitur Cellphone: Sistem memanfaatkan atribut seperti brand, sistem operasi, RAM, dan memori internal untuk menghitung kemiripan antar perangkat
- Memberikan Rekomendasi Personal: Jika seorang pengguna tertarik dengan satu merek atau model, sistem akan merekomendasikan ponsel lain yang memiliki spesifikasi serupa
- Meningkatkan Keputusan Pembelian: Dengan bantuan rekomendasi, pengguna bisa lebih percaya diri dalam memilih ponsel yang sesuai dengan kebutuhannya

[1] Menurut Anggoro dan Izzatillah (2022), sistem rekomendasi merupakan salah satu cabang dari machine learning yang lebih spesifik karena tidak hanya memprediksi nilai berdasarkan input baru, tetapi lebih berfokus pada penyediaan daftar produk yang sesuai dengan preferensi pengguna berdasarkan riwayat data sebelumnya  
[2] Menurut Nurhidayat dan Zulfikar (2020), penerapan content-based filtering dalam sistem rekomendasi smartphone terbukti efektif terutama ketika data interaksi pengguna masih terbatas. Mereka menyatakan bahwa pendekatan ini tetap dapat memberikan hasil rekomendasi yang relevan meskipun belum tersedia banyak ulasan atau riwayat pembelian. Dalam studi perbandingan yang dilakukan, pendekatan content-based filtering menghasilkan akurasi yang lebih tinggi dibandingkan dengan demographic filtering, karena mampu mempertimbangkan kesamaan fitur perangkat yang lebih spesifik dan relevan  

[1] Anggoro, M. V., & Izzatillah, M. (2022). Sistem Rekomendasi Musik dengan Metode Collaborative Filtering Berbasis Android. STRING (Satuan Tulisan Riset dan Inovasi Teknologi), 7(1), 1-8.  
[2] Nurhidayat, A., & Zulfikar, M. (2020). Perbandingan metode demographic dan content-based filtering pada sistem rekomendasi smartphone Android. Jurnal E-Proc, 7(3), 456–462.  

## Business Understanding

### Problem Statements

- Apa yang menjadi kesulitan utama pengguna dalam memilih smartphone?  
Pengguna sering mengalami kebingungan karena terlalu banyaknya pilihan smartphone dengan berbagai fitur dan harga yang berbeda  
- Mengapa sistem rekomendasi yang ada belum optimal dalam memberikan rekomendasi yang sesuai?  
Karena sebagian besar sistem belum memanfaatkan fitur spesifik perangkat secara maksimal sehingga rekomendasi yang diberikan kurang relevan dengan kebutuhan pengguna  
- Bagaimana keterbatasan data interaksi pengguna mempengaruhi kualitas rekomendasi?  
Minimnya data riwayat interaksi pengguna seperti rating dan pembelian membuat metode collaborative filtering kurang efektif  

### Goals

- Membantu pengguna dalam proses pemilihan smartphone yang tepat dan sesuai dengan preferensi mereka  
Proyek ini bertujuan mengembangkan sistem rekomendasi yang mampu menyajikan daftar smartphone yang paling relevan berdasarkan fitur dan kebutuhan pengguna, sehingga dapat mengurangi kebingungan dan mempercepat pengambilan keputusan
- Meningkatkan akurasi dan relevansi rekomendasi dengan memanfaatkan metode content-based filtering  
Dengan fokus pada kesamaan fitur smartphone, sistem akan menghasilkan rekomendasi yang lebih personal dan sesuai dengan karakteristik yang diinginkan pengguna, sehingga meningkatkan kepuasan pengguna terhadap hasil rekomendasi  
- Mengatasi masalah keterbatasan data interaksi pengguna dengan pendekatan content-based filtering  
Proyek ini mengimplementasikan teknik yang efektif meskipun data riwayat pengguna minim, sehingga sistem dapat tetap memberikan rekomendasi yang bermakna tanpa bergantung pada banyak data interaksi pengguna sebelumnya

    ### Solution statements
- Menggunakan teknik TF-IDF (Term Frequency-Inverse Document Frequency) untuk mengubah fitur teks menjadi representasi numerik  
	- Menggunakan TfidfVectorizer dari scikit-learn untuk mengolah data teks kolom model smartphone  
	- Menghitung bobot TF-IDF untuk setiap kata yang muncul dalam nama model, sehingga fitur teks dapat direpresentasikan dalam bentuk matriks numerik  
Tujuan: Mengubah data teks menjadi representasi numerik yang dapat dianalisis secara matematis sehingga memudahkan pengukuran kemiripan antar smartphone berdasarkan deskripsi model  
- Mengukur sudut kosinus antara dua vektor fitur setelah melakukan TF-IDF  
	- Menghitung cosine similarity antar matriks TF-IDF yang mewakili masing-masing smartphone  
	- Sistem merekomendasikan smartphone yang memiliki nilai similarity tertinggi terhadap smartphone input, artinya smartphone tersebut paling mirip secara fitur model  
Tujuan: Memberikan rekomendasi smartphone yang paling mirip berdasarkan fitur teks model, membantu pengguna menemukan alternatif smartphone yang relevan dengan preferensi mereka  

## Data Understanding
Dataset yang digunakan dalam proyek ini berasal dari Kaggle. Dataset ini berisi informasi lengkap tentang berbagai jenis smartphone, termasuk fitur teknis seperti brand, model, sistem operasi, memori, kamera, baterai, harga, serta data rating dan profil pengguna  
- **URL/Tautan Sumber Data** 
Dataset yang digunakan dalam proyek ini dapat diunduh melalui link ini: https://www.kaggle.com/datasets/meirnizri/cellphones-recommendations?select=cellphones+data.csv  
- **Jumlah Baris dan Kolom**
Dataset yang digunakan total adalah 3 dataset yang tujuan utamanya adalah untuk membuat sistem rekomendasi pemilihan cellphone, yaitu:  
	- Cellphone.csv: dataset ini terdiri dari 33 baris data dan 14 kolom yang berisi mengenai brand, model, os, dan lainnya 
 	- Rating.csv: dataset ini terdiri dari 990 baris data dan 3 kolom yang berisi rating semua pengguna
  	- User.csv: dataset ini terdiri dari 99 baris data dan 4 kolom yang berisi umur, jenis kelamin dan pekerjaan
- **Kondisi Data**
	- Missing Value: Terdapat 10 data missing value pada kolom occupation
	- Duplikat: Tidak terdapat duplikasi baris data

**Variabel-variabel dalam dataset adalah sebagai berikut:**
- cellphone_id : Kode unik yang mengidentifikasi setiap smartphone dalam dataset
- brand : Merek atau produsen smartphone 
- model : Nama atau tipe model smartphone 
- operating system : Sistem operasi yang digunakan smartphone 
- internal memory : Kapasitas penyimpanan internal smartphone dalam gigabyte (GB)
- RAM : Memori akses acak yang tersedia pada smartphone, dalam GB, mempengaruhi kecepatan pemrosesan
- performance : Ukuran atau skor performa smartphone 
- main camera : Spesifikasi kamera utama smartphone, biasanya dalam megapiksel (MP)
- selfie camera : Spesifikasi kamera depan/swatch smartphone, dalam MP
- battery size : Kapasitas baterai smartphone dalam milliampere-hour (mAh)
- screen size : Ukuran layar smartphone dalam inci
- weight : Berat smartphone dalam gram (g)
- price : Harga smartphone, biasanya dalam satuan mata uang tertentu (dalam dollar)
- release date : Tanggal atau tahun rilis smartphone
- user_id : Kode unik yang mengidentifikasi pengguna atau reviewer dalam dataset
- rating : Nilai rating yang diberikan pengguna terhadap smartphone (1-10)
- age : Usia pengguna dalam tahun
- gender : Jenis kelamin pengguna (Male/Female)
- occupation : Pekerjaan atau profesi pengguna

### **Tahapan**
- Exploratory Data Analysis (EDA)
	- Mengecek struktur data (info()), jumlah unik cellphone_id, daftar brand, sistem operasi (operating system), serta rentang harga. Sehingga dapat diketahui brand yang tersedia dan variasi harga produk
	- Melihat ringkasan statistik (describe()), total rating, dan jumlah user serta produk yang diberi rating. Sehingga dapat diketahui distribusi rating
	- Terakhir, mengecek dimensi data user untuk melihat ukuran populasi user dan informasi demografis dasar. Sehingga dapat diketahui persebaran umur, gender dan pekerjaan pengguna
- Data Pre-processing
	- Menggabungkan cellphone_id dari beberapa tabel, menghapus duplikat, dan menyatukan data rating, cellphone, dan user menjadi satu dataframe lengkap (merged_data)
	- Menggunakan .isnull().sum() untuk mengecek nilai hilang, serta groupby() untuk melihat total rating per cellphone
	- Menggabungkan data rating, nama model, dan data user menjadi all_cellphone sebagai dasar sistem rekomendasi 

## Data Preparation
Dalam tahap Data Preparation ini, dilakukan beberapa proses penting untuk memastikan data siap digunakan dalam analisis atau pemodelan Machine Learning. Dataset yang telah digabungkan, dimuat, kemudian diperiksa untuk memastikan tidak terdapat nilai kosong (missing value) dan semua tipe data sesuai dengan karakteristik masing-masing fitur  
- Langkah awal adalah melakukan penggabungan beberapa DataFrame, yaitu rating, cellphone, dan user, menjadi satu kesatuan menggunakan fungsi merge. Setelah digabungkan, disimpan pada variabel all_cellphone_rate
Alasan: Penggabungan kembali rating, user, dan data cellphone untuk memastikan bahwa informasi nama model dan fitur lainnya tersedia untuk setiap interaksi user-ponsel. Hal ini penting untuk pendekatan content-based filtering yang memanfaatkan fitur deskriptif dari ponsel
- Pada all_cellphone ditemukan 10 nilai kosong pada fitur occupation. Nilai-nilai kosong ini diatasi dengan menghapus baris yang mengandung missing value menggunakan metode `dropna()`, karena proporsinya tidak signifikan terhadap keseluruhan data  
Alasan: Menghindari error dalam proses transformasi atau pelatihan model akibat missing value atau tipe data yang tidak sesuai  
- Data diurutkan berdasarkan cellphone_id secara ascending untuk memastikan urutan yang konsisten dan memudahkan proses verifikasi serta pelacakan data di tahap selanjutnya  
Alasan: Mengurutkan data membantu memastikan keteraturan visual dan memudahkan proses debugging, pelacakan data, serta pencocokan antar data ketika dilakukan proses merge atau indexing pada tahap analisis lanjutan  
- Mengecek primary key sesuai kategori seperti mengecek jumlah primary key di dataset all_cellphone, jumlah brand yang unik. Hal ini membantu dalam proses encoding jika diperlukan pada tahap modeling lebih lanjut
Alasan: Mengecek keunikan data pada kolom seperti cellphone_id atau user_id memastikan bahwa tidak ada entri ganda yang tidak disengaja. Agar tidak terjadi redudansi
- Membuat variabel preparation yang berisi dataframe fix_cellphone dan diurutkan berdasarkan cellphone_id. (Jelaskan)
Alasan: Variabel preparation disiapkan sebagai versi bersih dan siap pakai dari data, yang menyimpan informasi dasar cellphone. Ini memudahkan akses data untuk kebutuhan modelling seperti pembuatan sistem rekomendasi content based filtering. Urutan berdasarkan cellphone_id juga menjaga konsistensi  
- Dilakukan pengecekan terhadap duplikasi data berdasarkan fitur unik yaitu `cellphone_id`. Duplikat yang terdeteksi dihapus untuk memastikan setiap entri mewakili satu produk yang unik dan menghindari bias pada analisis  
Alasan: Data duplikat dapat menyebabkan distorsi pada hasil analisis atau pelatihan model, karena informasi yang sama dihitung lebih dari sekali. Dengan memastikan bahwa setiap cellphone_id unik, maka setiap model cellphone hanya direpresentasikan satu kali
- Fitur-fitur penting seperti `cellphone_id`, `brand`, dan `model` dipisahkan dan diubah ke dalam bentuk list, kemudian dikemas ulang dalam bentuk DataFrame baru dengan struktur yang lebih ringkas dan sesuai kebutuhan analisis  
Alasan: Pemisahan dan pembentukan ulang ini membuat data lebih modular dan siap digunakan untuk representasi input dalam model, misalnya untuk proses text vectorization dalam content-based filtering
- Membuat dictionary yang disimpan dalam cellphone_new untuk data cellphone_id menjadi id, cellphone_brand menjadi brand, cellphone_model menjadi model
Alasan: Pemberian nama ulang kolom dilakukan untuk menyederhanakan struktur data, meningkatkan keterbacaan, dan menyesuaikan dengan format yang diperlukan pada tahap modeling selanjutnya, seperti saat membuat rekomendasi berbasis TF-IDF dan cosine similarity
Seluruh tahapan ini dilakukan untuk memastikan bahwa data yang digunakan dalam analisis sudah bersih dari error, duplikat, dan inkonsistensi. Data yang telah melalui tahap preparation ini menjadi dasar yang kuat untuk tahapan berikutnya yaitu modeling
- Mengubah fitur model menjadi representasi numerik menggunakan TfidfVectorizer. Hal ini dilakukan lalu diterapkan pada kolom model untuk mengubah teks menjadi vektor angka berbasis frekuensi. Hasilnya adalah matriks TF-IDF yang menggambarkan sejauh mana kata-kata dalam model berkontribusi dalam membedakan satu produk dengan produk lainnya
Alasan: Transformasi teks ke dalam bentuk numerik sangat penting dalam sistem rekomendasi berbasis Content-Based Filtering, karena model tidak bisa secara langsung memahami data dalam bentuk string. TF-IDF membantu dalam menilai pentingnya kata dalam keseluruhan dataset

## Modeling
Setelah data dipersiapkan dan fitur model dikonversi menjadi representasi vektor menggunakan TF-IDF pada tahap sebelumnya, proses modeling dapat dimulai. Pada tahap Modeling ini, dilakukan pembangunan sistem rekomendasi yang dapat menyarankan smartphone (cellphone) berdasarkan kemiripan model. Dalam proyek ini digunakan pendekatan Content-Based Filtering, yang merekomendasikan item berdasarkan kemiripan kontennya

#### Cosine Similarity
Setelah didapat representasi vektor dari tiap nama model, sekarang dihitung cosine similarity antar vektor. Nilai cosine similarity menunjukkan tingkat kemiripan antar model, dengan nilai mendekati 1 berarti sangat mirip  

- Hasil similarity ini disimpan dalam bentuk matriks yang digunakan sebagai dasar untuk mencari model lain yang paling mirip dengan suatu brand/model tertentu

#### Content-Based Filtering
Pada tahap ini dibuat fungsi cellphone_recommendations() yang berfungsi untuk mencari dan mengembalikan daftar rekomendasi smartphone yang paling mirip berdasarkan cosine similarity dari nama brand dan model yang sudah ditransformasi. Pemodelan ini tidak menggunakan data interaksi pengguna seperti rating, karena pendekatan content-based fokus pada karakteristik item itu sendiri  
Langkah langkah dalam fungsi cellphone_recommendations() adalah berikut:  

- Mengambil nilai kemiripan dari brand yang dipilih terhadap semua brand lain berdasarkan cosine similarity
- Mencari indeks dari brand yang memiliki nilai kemiripan tertingggi lalu memilih k brand terdekat dengan brand tersebut
- Brand yang sama dengan input akan dihapus dari daftar hasil agar tidak direkomendasikan ke dirinya sendiri
- fungsi menggabungkan brand yang direkomendasikan dengan data items (berisi brand dan model), dan menampilkan k hasil teratas sebagai rekomendasi
 
#### Top-N Recommendation
Fungsi cellphone_recommendations() dibuat untuk menghasilkan rekomendasi berdasarkan Top-N kemiripan.
- Input: nama brand, dan jumlah rekomendasi k  
- Output: daftar 5 brand dan model yang paling mirip dengan brand yang dipilih, berdasarkan nilai cosine similarity tertinggi
- Contoh hasil rekomendasi untuk ‘Samsung’:
	- Brand: Apple
  	- Model: iPhone SE (2022), iPhone 13 Mini, iPhone 13, iPhone 13 Pro, iPhone 13 Pro Max 

#### Kelebihan dan Kekurangan Content-Based Filtering
- Kelebihan
	- Tidak bergantung pada user: Content-based filtering dapat digunakan bahkan jika tidak ada data pengguna 
	- Memanfaatkan atribut produk: Rekomendasi tetap dapat dibuat berdasarkan informasi model tanpa perlu rating eksplisit
- Kekurangan:
	- Terbatas pada informasi yang tersedia: Hanya mempertimbangkan kemiripan dari data fitur (misalnya nama model), bukan preferensi pengguna
	- Kurang variasi: Cenderung merekomendasikan item yang sangat mirip, kurang eksploratif dibanding collaborative filtering

## Evaluation

#### Metrik Evaluasi Yang Digunakan 
Dalam proyek sistem rekomendasi berbasis Content-Based Filtering ini, metrik evaluasi yang digunakan adalah Precision@K dan Recall@K. Berikut penjelasannya:  

- **Precision@K**
Precision@K mengukur proporsi rekomendasi yang relevan dari total rekomendasi yang diberikan sebanyak K item.
Rumusnya: Precision@K = Jumlah item relevan dalam top K rekomendasi / K  
Precision@K menunjukkan seberapa akurat sistem dalam memberikan rekomendasi yang sesuai dengan kebutuhan pengguna pada posisi teratas rekomendasi

- **Recall@K**
Recall@K mengukur proporsi item relevan yang berhasil direkomendasikan dari total item relevan yang seharusnya direkomendasikan.
Rumusnya: Recall@K = Jumlah item relevan dalam top K rekomendasi / Total item relevan  
Recall@K menilai kemampuan sistem untuk menangkap sebanyak mungkin item relevan dalam rekomendasi

#### Cara Penggunaan Metrik dalam Proyek
Sistem rekomendasi menghasilkan daftar 5 rekomendasi smartphone paling mirip (K=5) berdasarkan cosine similarity dari nama model. Untuk mengukur kualitas rekomendasi, dilakukan evaluasi dengan membandingkan hasil rekomendasi terhadap daftar model yang dianggap relevan (ground truth) untuk brand tertentu

#### Hasil Evaluasi
Hasil evaluasi menunjukkan bahwa sistem mampu memberikan rekomendasi cellphone yang secara tekstual modelnya paling mirip dengan brand yang dicari. 
Sebagai contoh, untuk brand Samsung, daftar model relevan yang telah ditentukan adalah:
**['iPhone 13', 'iPhone 13 Pro', 'iPhone SE (2022)', 'Galaxy S21', 'Pixel 6']**
Sistem menghasilkan rekomendasi 5 model teratas, dan terdapat 4 model yang sesuai dengan daftar relevan tersebut  

- Maka nilai metrik evaluasi adalah:
	- Precision@5 = 4 / 5 = 0.80  
		Artinya, 80% rekomendasi yang diberikan relevan

	- Recall@5 = 4 / 5 = 0.80  
  		Artinya, sistem berhasil merekomendasikan 80% dari total model relevan
   
Nilai Precision@5 dan Recall@5 yang tinggi mengindikasikan sistem rekomendasi berbasis Content-Based Filtering ini efektif dalam memberikan rekomendasi smartphone yang relevan berdasarkan kemiripan model. Namun, karena evaluasi hanya menggunakan atribut model tanpa mempertimbangkan preferensi pengguna atau faktor lain, sistem ini masih dapat dikembangkan dengan data tambahan seperti rating atau fitur teknis untuk hasil yang lebih akurat

**---Ini adalah bagian akhir laporan---**
