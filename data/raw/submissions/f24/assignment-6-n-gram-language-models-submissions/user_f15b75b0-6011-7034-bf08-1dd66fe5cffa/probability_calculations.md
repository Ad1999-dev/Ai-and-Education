### 1) \( P(X_1 = a, X_2 = a, X_3 = a) \), using the **chain rule** for probability:

\[
P(X_1 = a, X_2 = a, X_3 = a) = P(X_1 = a) \cdot P(X_2 = a \mid X_1 = a) \cdot P(X_3 = a \mid X_1 = a, X_2 = a)
\]


#### **Unigram Probability**:
\[
P(X_1 = a) = \frac{f(a)}{\text{size(C)}} = \frac{11}{20}
\]

#### **Bigram Probability**:
\[
P(X_2 = a \mid X_1 = a) = \frac{f(aa)}{f(a)} = \frac{5}{11}
\]

#### **Trigram Probability**:
\[
P(X_3 = a \mid X_1 = a, X_2 = a) = \frac{f(aaa)}{f(aa)} = \frac{1}{5}
\]

---

Substitute the values:

\[
P(X_1 = a, X_2 = a, X_3 = a) = \frac{11}{20} \cdot \frac{5}{11} \cdot \frac{1}{5}
\]

---

Multiply the numerators and denominators:

\[
P(X_1 = a, X_2 = a, X_3 = a) = \frac{11 \cdot 5 \cdot 1}{20 \cdot 11 \cdot 5}
\]

Cancel out common factors:

\[
P(X_1 = a, X_2 = a, X_3 = a) = \frac{1}{20}
\]

---

### Final Answer:
\[
P(X_1 = a, X_2 = a, X_3 = a) = \frac{1}{20}
\]

---

### 2) \( P(X_1 = a, X_2 = a, X_3 = b) \), using the **chain rule** for probabilities:

\[
P(X_1 = a, X_2 = a, X_3 = b) = P(X_1 = a) \cdot P(X_2 = a \mid X_1 = a) \cdot P(X_3 = b \mid X_1 = a, X_2 = a)
\]



#### **Unigram Probability**:
\[
P(X_1 = a) = \frac{f(a)}{\text{size(C)}} = \frac{11}{20}
\]

#### **Bigram Probability**:
\[
P(X_2 = a \mid X_1 = a) = \frac{f(aa)}{f(a)} = \frac{5}{11}
\]

#### **Trigram Probability**:
\[
P(X_3 = b \mid X_1 = a, X_2 = a) = \frac{f(aab)}{f(aa)} = \frac{2}{5}
\]

---

Substitute the values:

\[
P(X_1 = a, X_2 = a, X_3 = b) = \frac{11}{20} \cdot \frac{5}{11} \cdot \frac{2}{5}
\]

---

Multiply the numerators and denominators:

\[
P(X_1 = a, X_2 = a, X_3 = b) = \frac{11 \cdot 5 \cdot 2}{20 \cdot 11 \cdot 5}
\]

Cancel out common factors:

\[
P(X_1 = a, X_2 = a, X_3 = b) = \frac{2}{20} = \frac{1}{10}
\]

---

### Final Answer:
\[
P(X_1 = a, X_2 = a, X_3 = b) = \frac{1}{10}
\]

---
### 3) \( P(X_1 = a, X_2 = a, X_3 = c) \), using the **chain rule** for probabilities:

\[
P(X_1 = a, X_2 = a, X_3 = c) = P(X_1 = a) \cdot P(X_2 = a \mid X_1 = a) \cdot P(X_3 = c \mid X_1 = a, X_2 = a)
\]



#### **Unigram Probability**:
\[
P(X_1 = a) = \frac{f(a)}{\text{size(C)}} = \frac{11}{20}
\]

#### **Bigram Probability**:
\[
P(X_2 = a \mid X_1 = a) = \frac{f(aa)}{f(a)} = \frac{5}{11}
\]

#### **Trigram Probability**:
\[
P(X_3 = c \mid X_1 = a, X_2 = a) = \frac{f(aac)}{f(aa)} = \frac{1}{5}
\]

---

Substitute the values:

\[
P(X_1 = a, X_2 = a, X_3 = c) = \frac{11}{20} \cdot \frac{5}{11} \cdot \frac{1}{5}
\]

---

Multiply the numerators and denominators:

\[
P(X_1 = a, X_2 = a, X_3 = c) = \frac{11 \cdot 5 \cdot 1}{20 \cdot 11 \cdot 5}
\]

Cancel out common factors:

\[
P(X_1 = a, X_2 = a, X_3 = c) = \frac{1}{20}
\]

---

### Final Answer:
\[
P(X_1 = a, X_2 = a, X_3 = c) = \frac{1}{20}
\]