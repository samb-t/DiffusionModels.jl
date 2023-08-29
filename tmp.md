## Response

Thanks for the response and for recognising the potential of our approach. We are happy to provide more detail on these points.

> **Relationship between measures and densities**

The Radon-Nikodym theorem tells us the density for a measure $v$ absolutely continuous with respect to a base measure $\mu$, which we can choose as one defined in our previous response. The density when $\mu$ and $v$ are both arbitrary Gaussians can be found in e.g. [1]. As an example, studying the simpler case of two Gaussians with the same covariance for brevity, $\mu=\mathcal{N}(0,TT^*)$ and $v=\mathcal{N}(Tm,TT^*)$, then by [2]
$$
\frac{dv}{d\mu}(x)=\text{exp}\bigg(\langle m,T^{-1}x\rangle-\frac{\|m\|^2}{2}\bigg)
$$

> **Example of the operator $T$ and its inverse**

If we consider $T$ as used in the paper i.e. Gaussian blur, and choose $\mathcal{H}=L^2(\mathbb{R}^d)$, then the blurred output $h=Tx$ is the convolution
$$
h(c)=\int_{\mathbb{R}^d} K(c-y,t)x(y)dy, \text{ where } K(y,t)= \frac{1}{(4\pi t)^{\frac{d}{2}}} e^{-\frac{|y|^2}{4t}},
$$
and $t>0$ determines the extent of the blur. As previous, this operator can be restricted to $L^2([0,1]^d)$ by introducing boundary conditions $m$ such that $x(m(y))$ exists. The existence of the inverse is clear if we consider the Fourier transform of $x(c)$, denoted $\hat{x}(\omega)$, then blurring can instead be defined by $\hat{h}(\omega)=e^{-\omega^2 t}\hat{x}(\omega)$. And so $Tx$ is one-to-one on any class of Fourier transformable functions. Explicitly, the inverse is given by $\hat{x}(\omega)=e^{\omega^2 t}\hat{h}(\omega)$ [3]. John [4] show that $Tx$ being bounded ensures uniqueness and therefore invertibility.

However, inverting is ill-conditioned, with arbitrarily small changes destroying smoothness [3]. To bridge theory and practice, we must address errors induced by floating-point operations during sampling; we utilise the Wiener filter as an approximate inverse, defined as: $\tilde{x}(\omega)=\frac{e^{-\omega^2 t}}{e^{-2(\omega^2 t)}+\epsilon^2}\hat{h}(\omega)$ where $\epsilon$ is an estimate of the inverse SNR [5]. We will update the manuscript with definitions of $T$ and the approximate inverse, along with a related plot/example in the supp. material.

The existence/stability of $T^{-1}$ isn’t crucial to the modelling loss; because covariances are fixed, the goal is to match the Gaussians’ means. Thus, without $T^{-1}$, the minima remains unchanged (lines 121-123). Removing the covariance from the loss, as seen in finite-dimensional diffusion models, improves sample quality (Ho et al. 2020, Hoogeboom and Salimans 2023).

> **Consider existence of backwards in more detail**

We refer to Millet et al. 1989 who consider trace-class noise, whereas Follmer et al. 1986 consider identity-class noise. Millet et al. prove the existence of the backwards process in Theorem 4.3 under conditions H1-H5. This analysis is quite involved—of the concurrent work on infinite dimensional diffusion models, neither Lim et al., 2023, Hagemann et al., 2023, nor Zhang et al. 2023 formally prove the existence of the backward process and their score-matching objectives are heuristic. Our approach is modelled as a discrete time Markov chain, explicitly defining the reverse process (lines 113-114). In Sec. 3.2, we intended to highlight the similarities and differences between finite-dimensional diffusion models when generalising to SDEs. While this isn't the central focus of our work, we acknowledge the section's lack of depth. We will incorporate further details from Millet et al. in our revised manuscript and include the formulation of the SDE, $dy_t=-\frac{1}{2}\beta(t)y_t dt+\sqrt{\beta(t)}dw_t$ where $y_0=Tx_0$, $w_t$ is a $TT^*$-Wiener process on $\mathcal{H}$, and $\beta(t)$ is a continuous noise schedule (making $F^i=-\frac{1}{2}\beta(t)y_t^i$), generalising the VP SDE from Song et al. 2021.

> **Main use-case is super-resolution**

While super-resolution is a natural application of our approach, our main use-case is to support resolution-independent training/generalisation via training on subsets of coordinates without compression. Decoupling memory/training from resolution not only allows our current state-of-the-art results (Table 1) but also paves the way for future applications as in the neural fields literature, such as inference on ultra-high-resolution datasets, large 3D point clouds, and very long videos far exceeding what would otherwise fit into memory.

**References**

[1] Minh. Regularized Divergences Between Covariance Operators and Gaussian Measures on Hilbert Spaces. J. Theor. Probab. 34 (2021): 580-643

[2] Kukush. Gaussian Measures in Hilbert Space: Construction and Properties. 2020

[3] Hummel et al. Deblurring Gaussian Blur. CVGIP 38.1 (1987): 66-80

[4] John. Numerical Solution of the Equation of Heat Conduction for Preceding Times. Ann. Mat. Pura Appl. 40 (1955): 129-142

[5] Biemond et al. Iterative Methods for Image Deblurring." Proc. of the IEEE 78.5 (1990): 856-883










## General Comments
We thank all the reviewers for their insightful feedback and positive remarks, noting our work's novelty and state-of-the-art infinite resolution results. We recognise and agree with all the feedback, especially to clarify our motivations/contributions, enhance mathematical rigor, and provide deeper explanations on certain aspects. In our revised paper, we will:

- Expand on the motivation in Section 1. The motivation is to enable modelling higher resolution data, by treating data as continuous functions, subsampling coordinates to make this computationally feasible. Prior approaches perform poorly and are not scalable due to compression restrictions.
- Summarise the following contributions. 1. We propose a continuous state diffusion framework and 2. design an powerful, scalable function-space architecture that operates on sparsely sampled coordinates. 3. We show that this enables improvements in run time and memory usage, enabling efficient scaling to many sparse coordinates. Finally, 4. our approach substantially outperforms prior function-space generative models, and closes the gap to finite-dimensional approaches, despite being trained on subsets of data at random coordinates.
- Expand on theoretical foundations. Add a new subsection to Section 2, which introduces Gaussian measures defined in Hilbert space; expand Section 3.2, particularly with the points raised by reviewer WXTB such as the restrictions on $T$ and conditions for existence; add a section to the appendix with a simple introduction to Hilbert spaces and probabilities; and expand Section 2.1 on the background of diffusion models to be more comprehensive, with details such as the DDIM formulation.



## Reviewer 7E1P
We thank the reviewer for their constructive feedback and helpful suggestions. We address the weaknesses and questions below, point-by-point.

> **What is possible with the new approach/which experiments were not previously possible?**

To our knowledge, our approach is the only diffusion model that can be trained on subsets of coordinates without compression. This allows training on higher resolution data, provides speed/memory gains over standard diffusion models while also allowing sampling at arbitrary resolutions (Fig. 2) as well as zero-shot super resolution (Fig. 8), neither of which are possible with standard diffusion models. The lack of compression means that our approach is able to scale to much more complex higher resolution data than other infinite dimensional approaches e.g. on FFHQ-256, $\text{FID}_{\text{CLIP}}$ of 3.87 compared to the previous SOTA 24.37, while also being competitive with finite-dimensional diffusion models e.g. DDIMs (Song et al. 2021) on LSUN Church achieves $\text{FID}_{\text{CLIP}}$ of 11.70, higher than our 10.36, despite ours only observing subsets of coordinates during training. 

Additionally, some recent diffusion models (e.g [1]) have proposed to speed up sampling by starting with a low resolution image then iteratively moving to higher resolutions during the diffusion process. These approaches require separate models for each resolution; in contrast, our approach could achieve this with a single model (see Figures 1 and 2 for samples at various resolutions from the same model). 

> **Mollification makes the model finite-dimensional?**

While the Gaussian kernel does indeed introduce some locality and smoothness, it does not make the model finite-dimensional. Discretisation primarily aims to represent continuous functions in terms of a finite set of values (or parameters), while mollification aims to approximate non-smooth functions with smooth functions to fit into the mathematical framework.

> **Presentation difficult to follow**

- **Define $\mathcal{N}$**: Following Da Prato and Zabczyk, 2014, a Gaussian measure $\mu$ can be defined in a real Hilbert space $\mathcal{H}$ for mean $m\in\mathcal{H}$ and covariance operator $C:\mathcal{H}\to\mathcal{H}$, in terms of its characteristic function $\hat{\mu}(x) = \text{exp} \left( i\langle x,m\rangle + \frac{1}{2}\langle Cx,x\rangle \right)$, where $C$ is self-adjoint ($C=C^*$), non-negative ($C\geq 0$), and trace-class ($\int_{\mathcal{H}}\|x\|_{\mathcal{H}}d\mu(x)=\text{tr}(C)<\infty$). For a Gaussian random element $x$ with distribution $\mu$, then $x\sim\mathcal{N}(m,C)$.
- **Why doesn't $\tilde{\beta}_t$ appear... might not equal $\sigma_t$**: The $\sigma_t$ schedule is chosen separately and can therefore vary; however, it is typically set as $\sigma^2_t=\tilde{\beta}_t$ (Ho et al. 2020). We will clarify this in the updated paper.
- **Is $\mathcal{L}$ is a function of $\theta$?**: Yes, this is correct, we will add this for clarity.
- **Why not use one letter $p$ or $q$?**: $q$ represents the forwards diffusion, while $p$ represents the approximate reverse process since the true reverse process is intractable. This is standard notation for diffusion models see e.g. Ho et al. 2020.
- **Unlear whether Eq. 11 [is] the definition of mollification**: Mollification is the process of approximating a non-smooth function with a smooth function via convolution therefore $Tx$ is the act of mollifying and Eq. 11 is the resulting distribution. We will make this definition more explicit.
- **Not clear [what] undoing the process using a Wiener filter [means]**: Although the mollification procedure is theoretically invertible, in practice it is an ill-conditioned problem since the solution is highly sensitive to any errors, see e.g. [2]. Wiener filters are a fast regularised approach to approximately invert this procedure that minimises mean squared error.
- **What is $\phi$ in Eq. 18?**: This should be $\theta$, the parameters of the neural operator. Thanks for spotting this typo.
    
> **Typos**

Thanks for spotting these typos, they will be addressed in the final paper.

**References**

[1] Jing et al., 2022. Subspace diffusion generative models. In ECCV (pp. 274-289).

[2] Biemond et al., 1990. Iterative methods for image deblurring. Proceedings of the IEEE, 78(5), pp.856-883.



## Reviewer NNx8
We thank the reviewer for their time and helpful feedback. We address the weaknesses and questions below, point-by-point.

> **Examples where $\infty$-Diff outperforms both finite dimensional models and other infinite dimensional models**

While $\infty$-Diff outperforms other infinite dimensional models in terms of sample quality, advantages over comparable finite-dimensional models include the improvements in terms of training run time, scalability, and memory usage as a result of being able to train on sparse sets of coordinates, in addition to the extra capabilities such as zero-shot super-resolution and sampling at arbitrary resolutions. Additionally, the official DDIM (Song et al. 2021a) model on LSUN Church achieves an $\text{FID}_{\text{CLIP}}$ score of 11.70 which is higher than our 10.36 demonstrating that $\infty$-Diff is able to outperform finite-dimensional diffusion models despite ours only observing subsets of coordinates during training.

> **Finite dimension models outperform infinite dimension models because assumptions are more suitable for image datasets?**

This gap is primarily due to infinite resolution models not being trained on all coordinates at once, and those coordinates being irregularly spaced. In contrast, finite resolution models are trained for exactly one resolution. Infinite resolution models have the harder job of having to generalise from a different/smaller number of coordinates. The especially large gap present with prior infinite-dimensional approaches is additionally due to the reliance on latent compressed representations, however, our approach does not have this limitation; as such, the findings from our work could be used to improve other types of infinite-dimensional generative models.

> **Clarify how the variance of Gaussian blur influences FID scores**

Gaussian blur with higher variance will be harder to approximate the inverse. Fortunately, we can choose a small value since the requirement is to make non-smooth functions smooth. As an aside, in Fig. 7, we take a model trained on coordinate subsets of 256x256 images then evaluate FID when sampling at lower and higher resolutions and observe only a minor increase correlating with the changing sampling, which indicates an insignificant change with respect to the blur variance.


## Reviewer WXTB
We thank the reviewer for the comprehensive feedback, it is much appreciated and will significantly strengthen our paper. We address the weaknesses and questions below, point-by-point.

>**Gaussian Measures in Hilbert Space**

Following Da Prato and Zabczyk, 2014, a Gaussian measure $\mu$ can be defined in a real Hilbert space $\mathcal{H}$ in terms of its characteristic function $\hat{\mu}$,
$$\hat{\mu}(x)=\text{exp}\Big(i\langle x,m\rangle+\frac{1}{2}\langle Cx,x\rangle\Big),$$
for mean $m\in\mathcal{H}$ and covariance operator ($C:\mathcal{H}\to\mathcal{H}$) which is self-adjoint (denoted $C=C^*$), non-negative (i.e. $C\geq 0$), and trace-class ($\int_{\mathcal{H}}\|x\|_{\mathcal{H}}d\mu(x)=\text{tr}(C)<\infty$). For a Gaussian random element $x$ with distribution $\mu$, $x\sim\mathcal{N}(m,C)$. The distributions introduced in Section 3.2 fit this definition. We will add a subsection to Section 2 with these details and more background including the relationship between $\mu$ and $\hat{\mu}$.

>**Restrictions on the Mollifier**

From the mollifier kernel $k$ and corresponding operator $T$ we get the covariance operator $TT^*$ (Higdon 2002). To ensure this relationship is one-to-one, $k$ must satisfy $\int_{\mathbb{R}^d}k(s)ds<\infty$ and $\int_{\mathbb{R}^d}k^2(s)ds<\infty$. Since $TT^*$ is inherently self-adjoint, non-negative if $T$ is non-negative (as in the Gaussian blur case), the remaining requirement is that $TT^*$ is trace-class (discussed below).

>**Existence of $T^{-1}$**

In practice, we use the space of square integrable functions on the d-dimensional unit cube $\mathcal{H}=L^2([0,1]^d)$ to reflect the data considered. If $T$ is defined with symmetric boundary conditions then $T^{-1}$ exists. If $T$ is compact, then inversion is ill-posed, and a regularision operator must be used to approximate the inverse. In each case, $\text{tr}(TT^*)$ is finite. This connects with the practical perspective of inverting Gaussian blur, an ill-conditioned problem since the solution is highly sensitive to any error [1] therefore necessitating an approximate inverse e.g. a Wiener filter.

>**Conditions under which densities make sense/$x_t$ lies in $\mathcal{H}$**

Consider Eq. 11, $q(x_t|x_0)$. We have shown above that the covariance fits into the definition of $\hat{\mu}$. Since we assume that $x_0\in\mathcal{H}$ and $T:\mathcal{H}\to\mathcal{H}$ then $Tx\in\mathcal{H}$ meaning that $q(x_t|x_0)$ is well defined in Hilbert space. Section 3.1 introduces a natural extension of diffusion models to infinite dimensions (used by Zhuang et al. 2023), however, we observe that this interpretation is problematic ($tr(C_I)$ is not finite and $x_t\notin\mathcal{H}$). The reverse diffusion starts with $x_T$ which must lie in $\mathcal{H}$ since $x_T\sim\mathcal{N}(0,TT^*)$; assuming $x_{t+1}$ lies in $\mathcal{H}$, and $f_\theta$ maps to $\mathcal{H}$, then $x_t\sim p(x_t|x_{t+1})$ will lie in $\mathcal{H}$ as well.

>**Limitations/Motivations**

Our motivation is to model infinite dimensional data, or in practice exceedingly high resolution data, addressing this goal by subsampling coordinates to make processing computationally feasible. Our approach is not strictly limited to images of a fixed resolution; we propose a multi-level architecture (Fig. 4) which operates on sparse irregularly sampled coordinates at the top level using convolution operators (lines 186-190), not Fourier Neural Operators (which have this limitation). Specifically, convolution operators are applied to inputs independent of discretisation, therefore our approach will naturally extend to other domains commonplace in the implicit modelling literature (audio, video, sparse 3D shapes, and solving boundary value problems).

>**Dismissing Related Works**

We believe our approach is complementary to these concurrent works, focusing more on scalability/efficiency to be able to model much more complex, higher resolution data by operating on sparse coordinates without compression, and achieving SOTA quality in terms of FID. We will discuss these merits and draw more in-depth comparisons in the updated paper.

>**Reverse Time SDE**

To be more precise, given the restriction to trace-class covariance operators, we should instead consider [2], where sufficient conditions for time reversal are given in Section 5. 

>**Standard FID Scores**

Standard FID has been shown to be too aligned to ImageNet classes (Kynkaanniemi et al. 2023), and has poor correlation with human evaluation [3], making it a poor choice for assessing quality; as such, we should no longer use it. Alternative feature extractors such as CLIP do not have this limitation and are much better aligned with quality [3].

>**Orthogonal Projection Used**

As mentioned on lines 162-163, orthogonal projection is performed by taking $\\{c_1,...,c_n\\}\subset[0,1]^d$ and evaluating at these points, $x(c_i)$. Usage of this projection is essential to allow application of sparse neural operators (Kovachki et al., 2021), while also being used by other infinite-dimensional generative models.

>**Monte-Carlo Approximation of Eq. 3**

This was a poor explanation; we meant Eq. 3 but where $x_t$ is infinite dimensional. As such we Monte-Carlo approximate the integral over the space of $x_t$, sampling at coordinates $\mathbf{c}$. We will explain this properly in the updated paper.

We have found all these comments helpful and in all cases will update the paper according to these discussions. We hope this adequately addresses these concerns and are happy to take more questions and further elaborate on any points.

**References**

[1] Biemond et al., 1990. Iterative methods for image deblurring. Proc. of the IEEE, 78(5), pp.856-883

[2] Millet et al., 1989. Time reversal for infinite-dimensional diffusions. Probab. Theory Relat. Fields, 82(3), pp.315-347

[3] Stein et al., 2023. Exposing flaws of generative model evaluation metrics and their unfair treatment of diffusion models. arXiv:2306.04675



## Reviewer FFsr
We thank the reviewer for their constructive feedback and helpful suggestions. We address the weaknesses and questions below, point-by-point.

> **Compare with standard finite dimensional diffusion**

We agree, this would be a helpful comparison. In the updated paper we will add scores for a finite resolution diffusion approach to Table 1. For context, the official DDIM (song et al. 2021a) model with 100 sampling steps achieves an $\text{FID}_{\text{CLIP}}$ score of 11.70 on LSUN Church, higher than our 10.36, demonstrating that $\infty$-Diff is able to outperform finite-dimensional diffusion models despite ours only observing subsets of coordinates during training. 

> **Some parts may need a bit more explanation or motivation**

Please see our general comment to all reviewers in which we explain how we will improve the explanation and description of motivation within the paper. As for **"methods in works mentioned in passing"**, thanks for raising this point; we will add more discussion in these cases. In particular, we will add more background details on the prior infinite resolution models e.g. GASP, GEM, etc; introduce Fourier Neural Operators as well as alias-free kernels (Romero et al. 2022) in section 4; and give the details behind DDIMs in section 2.1. If there are any more particularly unclear mentions, please let us know.

> **Unsure why the previously mentioned approach (Sec 3.1) becomes problematic?**

The noise used in Sec. 3.1 is defined with the covariance operator $C_I(z(s), z(s')) = \delta(s - s')$ meaning that at each coordinate the noise is a sample from an independent Gaussian. Intuitively, this noise is not smooth in that it has discontinuities everywhere. As such, it lies outside of the Hilbert space $\mathcal{H}$. Using neural operators which assume that input functions are smooth is therefore not possible and we would be unable to use them to model the denoising function.

> **Some notations do not seem to be introduced**

Thanks for spotting these, we will clarify the meaning of these in the updated paper: $\left(\begin{smallmatrix}D\\\\m\end{smallmatrix}\right)$ is the set of all possible subsets of $D$ with length $m$. $k$ is a neural network that parameterises the kernel.
