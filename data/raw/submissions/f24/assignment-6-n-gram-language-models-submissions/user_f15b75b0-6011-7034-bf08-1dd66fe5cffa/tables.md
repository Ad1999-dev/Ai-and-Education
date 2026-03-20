# string = `'aababcaccaaacbaabcaa'`

### **Unigram Counts**
\[
f(a) = 11, \, f(b) = 4, \, f(c) = 5, \, \text{size(C)} = 20
\]

---

### **Bigram Counts** (from \( X_1X_2 \))
\[
\begin{aligned}
f(aa) &= 5, & f(ab) &= 3, & f(ac) &= 2, \\
f(ba) &= 2, & f(bb) &= 0, & f(bc) &= 2, \\
f(ca) &= 3, & f(cb) &= 1, & f(cc) &= 1.
\end{aligned}
\]

---

### **Trigram Counts** (from \( X_1X_2X_3 \))
\[
\begin{aligned}
f(aaa) &= 1, & f(aab) &= 2, & f(aac) &= 2, \\
f(aba) &= 1, & f(abc) &= 2, & f(acb) &= 1, \\
f(acc) &= 1, & f(baa) &= 1, & f(bab) &= 1, \\
f(bca) &= 2, & f(caa) &= 2, & f(cac) &= 1, \\
f(cca) &= 1, & f(cba) &= 1.
\end{aligned}
\]

Any other trigram combination has a count of 0.

---

### **Probability Table 1**

For each unigram \( X_1 \), calculate:
\[
P(X_1) = \frac{f(X_1)}{\text{size(C)}}
\]

| \( P(X_1) \) | Formula                | Probability |
|--------------|------------------------|-------------|
| \( P(a) \)   | \( \frac{f(a)}{20} = \frac{11}{20}\)  | \( 0.5500 \) |
| \( P(b) \)   | \( \frac{f(b)}{20} = \frac{4}{20}\)  | \( 0.2000 \) |
| \( P(c) \)   | \( \frac{f(c)}{20} = \frac{5}{20}\)  | \( 0.2500 \) |

---

### **Probability Table 2**

For each bigram \( X_1X_2 \), calculate:
\[
P(X_2 \mid X_1) = \frac{f(X_1X_2)}{f(X_1)}
\]

| \( P(X_2 \mid X_1) \) | Formula                                      | Probability |
|------------------------|----------------------------------------------|-------------|
| \( P(a \mid a) \)      | \( \frac{f(aa)}{f(a)} = \frac{5}{11} \)     | \( 0.4545 \) |
| \( P(b \mid a) \)      | \( \frac{f(ab)}{f(a)} = \frac{3}{11} \)     | \( 0.2727 \) |
| \( P(c \mid a) \)      | \( \frac{f(ac)}{f(a)} = \frac{2}{11} \)     | \( 0.1818 \) |
| \( P(a \mid b) \)      | \( \frac{f(ba)}{f(b)} = \frac{2}{4} \)      | \( 0.5000 \) |
| \( P(b \mid b) \)      | \( \frac{f(bb)}{f(b)} = \frac{0}{4} \)      | \( 0.0000 \) |
| \( P(c \mid b) \)      | \( \frac{f(bc)}{f(b)} = \frac{2}{4} \)      | \( 0.5000 \) |
| \( P(a \mid c) \)      | \( \frac{f(ca)}{f(c)} = \frac{3}{5} \)      | \( 0.6000 \) |
| \( P(b \mid c) \)      | \( \frac{f(cb)}{f(c)} = \frac{1}{5} \)      | \( 0.2000 \) |
| \( P(c \mid c) \)      | \( \frac{f(cc)}{f(c)} = \frac{1}{5} \)      | \( 0.2000 \) |

---

### **Probability Table 3**

For each trigram \( X_1X_2X_3 \), calculate:
\[
P(X_3 \mid X_1X_2) = \frac{f(X_1X_2X_3)}{f(X_1X_2)}
\]

| \( X_1X_2 \) | \( X_3 \) | \( f(X_1X_2X_3) \) | \( f(X_1X_2) \) | \( P(X_3 \mid X_1X_2) \)                   |
|--------------|-----------|--------------------|-----------------|---------------------------------------------------------|
| \( aa \)     | \( a \)   | \( 1 \)            | \( 5 \)         | \( \frac{f(aaa)}{f(aa)} = \frac{1}{5} = 0.2000 \)       |
| \( aa \)     | \( b \)   | \( 2 \)            | \( 5 \)         | \( \frac{f(aab)}{f(aa)} = \frac{2}{5} = 0.4000 \)       |
| \( aa \)     | \( c \)   | \( 1 \)            | \( 5 \)         | \( \frac{f(aac)}{f(aa)} = \frac{1}{5} = 0.2000 \)       |
| \( ab \)     | \( a \)   | \( 1 \)            | \( 3 \)         | \( \frac{f(aba)}{f(ab)} = \frac{1}{3} = 0.3333 \)       |
| \( ab \)     | \( b \)   | \( 0 \)            | \( 3 \)         | \( \frac{f(abb)}{f(ab)} = \frac{0}{3} = 0.0000 \)       |
| \( ab \)     | \( c \)   | \( 2 \)            | \( 3 \)         | \( \frac{f(abc)}{f(ab)} = \frac{2}{3} = 0.6667 \)       |
| \( ac \)     | \( a \)   | \( 0 \)            | \( 2 \)         | \( \frac{f(aca)}{f(ac)} = \frac{0}{2} = 0.0000 \)       |
| \( ac \)     | \( b \)   | \( 1 \)            | \( 2 \)         | \( \frac{f(acb)}{f(ac)} = \frac{1}{2} = 0.5000 \)       |
| \( ac \)     | \( c \)   | \( 1 \)            | \( 2 \)         | \( \frac{f(acc)}{f(ac)} = \frac{1}{2} = 0.5000 \)       |
| \( ba \)     | \( a \)   | \( 1 \)            | \( 2 \)         | \( \frac{f(baa)}{f(ba)} = \frac{1}{2} = 0.5000 \)       |
| \( ba \)     | \( b \)   | \( 1 \)            | \( 2 \)         | \( \frac{f(bab)}{f(ba)} = \frac{1}{2} = 0.5000 \)       |
| \( ba \)     | \( c \)   | \( 0 \)            | \( 2 \)         | \( \frac{f(bac)}{f(ba)} = \frac{0}{2} = 0.0000 \)       |
| \( bc \)     | \( a \)   | \( 2 \)            | \( 2 \)         | \( \frac{f(bca)}{f(bc)} = \frac{2}{2} = 1.0000 \)       |
| \( bc \)     | \( b \)   | \( 0 \)            | \( 2 \)         | \( \frac{f(bcb)}{f(bc)} = \frac{0}{2} = 0.0000 \)       |
| \( bc \)     | \( c \)   | \( 0 \)            | \( 2 \)         | \( \frac{f(bcc)}{f(bc)} = \frac{0}{2} = 0.0000 \)       |
| \( ca \)     | \( a \)   | \( 2 \)            | \( 3 \)         | \( \frac{f(caa)}{f(ca)} = \frac{2}{3} = 0.6667 \)       |
| \( ca \)     | \( b \)   | \( 0 \)            | \( 3 \)         | \( \frac{f(cab)}{f(ca)} = \frac{0}{3} = 0.0000 \)       |
| \( ca \)     | \( c \)   | \( 1 \)            | \( 3 \)         | \( \frac{f(cac)}{f(ca)} = \frac{1}{3} = 0.3333 \)       |
| \( cb \)     | \( a \)   | \( 1 \)            | \( 1 \)         | \( \frac{f(cba)}{f(cb)} = \frac{1}{1} = 1.0000 \)               |
| \( cb \)     | \( b \)   | \( 0 \)            | \( 1 \)         | \( \frac{f(cbb)}{f(cb)} = \frac{0}{1} = 0.0000 \)               |
| \( cb \)     | \( c \)   | \( 0 \)            | \( 1 \)         | \( \frac{f(cbc)}{f(cb)} = \frac{0}{1} = 0.0000 \)               |
| \( cc \)     | \( a \)   | \( 1 \)            | \( 1 \)         | \( \frac{f(cca)}{f(cc)} = \frac{1}{1} = 1.0000 \)               |
| \( cc \)     | \( b \)   | \( 0 \)            | \( 1 \)         | \( \frac{f(ccb)}{f(cc)} = \frac{0}{1} = 0.0000 \)               |
| \( cc \)     | \( c \)   | \( 0 \)            | \( 1 \)         | \( \frac{f(ccc)}{f(cc)} = \frac{0}{1} = 0.0000 \)               |
| \( bb \)     | \( a \)   | \( 0 \)            | \( 0 \)         | \( \frac{f(bba)}{f(bb)} = \frac{0}{0} = \text{undefined} = 0 \)     |
| \( bb \)     | \( b \)   | \( 0 \)            | \( 0 \)         | \( \frac{f(bbb)}{f(bb)} = \frac{0}{0} = \text{undefined} = 0 \)     |
| \( bb \)     | \( c \)   | \( 0 \)            | \( 0 \)         | \( \frac{f(bbc)}{f(bb)} = \frac{0}{0} = \text{undefined} = 0 \)     |

$$P(X_1=x_1, X_2=x_2, \dots, X_n=x_n) = P(x_1) \cdot P(x_2 \mid x_1) \cdot \ldots \cdot P(x_n \mid x_{n-1}) = \frac{f(x_1)}{\text{size(C)}} \cdot \frac{f(x_1, x_2)}{f(x_1)} \cdot \frac{f(x_2, x_3)}{f(x_2)} \cdot \dots \cdot \frac{f(x_{n-1}, x_n)}{f(x_{n-1})}$$

Here, the $$size(C)$$ is the total number of characters in the corpus.