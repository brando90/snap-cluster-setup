/-
\section{INTRODUCTION}

A satisfactory discussion of the main concepts of analysis (such as convergence, continuity, differentiation, and integration) must be based on an accurately defined number concept. We shall not, however, enter into any discussion of the axioms that govern the arithmetic of the integers, but assume familiarity with the rational numbers (i.e., the numbers of the form $m / n$, where $m$ and $n$ are integers and $n \neq 0$ ).

The rational number system is inadequate for many purposes, both as a field and as an ordered set. (These terms will be defined in Secs. 1.6 and 1.12.) For instance, there is no rational $p$ such that $p^{2}=2$. (We shall prove this presently.) This leads to the introduction of so-called "irrational numbers" which are often written as infinite decimal expansions and are considered to be "approximated" by the corresponding finite decimals. Thus the sequence

$$
1,1.4,1.41,1.414,1.4142, \ldots
$$

"tends to $\sqrt{2}$." But unless the irrational number $\sqrt{2}$ has been clearly defined, the question must arise: Just what is it that this sequence "tends to"? This sort of question can be answered as soon as the so-called "real number system" is constructed.

1.1 Example We now show that the equation

$$
p^{2}=2
$$

is not satisfied by any rational $p$. If there were such a $p$, we could write $p=m / n$ where $m$ and $n$ are integers that are not both even. Let us assume this is done. Then (1) implies

$$
m^{2}=2 n^{2},
$$

This shows that $m^{2}$ is even. Hence $m$ is even (if $m$ were odd, $m^{2}$ would be odd), and so $m^{2}$ is divisible by 4. It follows that the right side of (2) is divisible by 4, so that $n^{2}$ is even, which implies that $n$ is even.

The assumption that (1) holds thus leads to the conclusion that both $m$ and $n$ are even, contrary to our choice of $m$ and $n$. Hence (1) is impossible for rational $p$.

We now examine this situation a little more closely. Let $A$ be the set of all positive rationals $p$ such that $p^{2}<2$ and let $B$ consist of all positive rationals $p$ such that $p^{2}>2$. We shall show that $A$ contains no largest number and $B$ contains no smallest.

More explicitly, for every $p$ in $A$ we can find a rational $q$ in $A$ such that $p<q$, and for every $p$ in $B$ we can find a rational $q$ in $B$ such that $q<p$.

To do this, we associate with each rational $p>0$ the number

$$
q=p-\frac{p^{2}-2}{p+2}=\frac{2 p+2}{p+2}
$$

Then

$$
q^{2}-2=\frac{2\left(p^{2}-2\right)}{(p+2)^{2}}
$$

If $p$ is in $A$ then $p^{2}-2<0$, (3) shows that $q>p$, and (4) shows that $q^{2}<2$. Thus $q$ is in $A$.

If $p$ is in $B$ then $p^{2}-2>0$, (3) shows that $0<q<p$, and (4) shows that $q^{2}>2$. Thus $q$ is in $B$.

1.2 Remark The purpose of the above discussion has been to show that the rational number system has certain gaps, in spite of the fact that between any two rationals there is another: If $r<s$ then $r<(r+s) / 2<s$. The real number system fills these gaps. This is the principal reason for the fundamental role which it plays in analysis. In order to elucidate its structure, as well as that of the complex numbers, we start with a brief discussion of the general concepts of ordered set and field.
-/

-- TODO: proof p^2 = 2 doesn't exist in Q (or sqrt(2) is a "hole"). Plan is to either formalize Q first (or use mathlib def) then proof statement
