# Estimating Sample Mean and SD from Some Quartiles

---

## Overview

This project provides a **neural network–based tool for estimating the sample mean and standard deviation from summary statistics**, including the median, range, and/or quartiles.
The method is designed for situations where individual-level data are unavailable and only limited summary statistics can be accessed, especially when the data is **skewed** or does **not** meet the normality assumption.

The tool is deployed as a **lightweight web application** and can be accessed directly at:

👉 <a href="https://amss-stat.github.io/estimating-sample-mean-standard-deviation/"
     target="_blank"
     rel="noopener noreferrer">
     https://amss-stat.github.io/estimating-sample-mean-standard-deviation/
   </a>

All computations are performed locally in the browser, without requiring data upload or server-side processing.

Typical application scenarios include:

* Meta-analysis
* Systematic reviews
* Evidence synthesis across multiple studies

---

## Methodology

The estimation procedure is based on **neural network models trained on large-scale diverse synthetic data**, covering a wide range of common distributional families and sample sizes.

Given a set of summary statistics (e.g., sample size, median, minimum/maximum, and/or quartiles), the tool:

1. Infers sample mean and standard deviation using pretrained neural networks
2. Reconstruct the latent distribution
3. Selects the best-fitting distribution by comparing theoretical and observed quantiles

This framework enables accurate estimation even under **skewed or non-normal distributions**, where traditional methods often perform poorly.

---

## Performance

Extensive simulation studies show that:

* The proposed method **outperforms classical estimators** (e.g., Luo, Wan et al.) in most settings
* The improvement is particularly pronounced for:

  * **Skewed** data
  * Situations where the **normality** assumption is **violated**

---

## Reference

If you use this tool or method in your research, please cite:

> **Zhang, Qinyuan; Li, Qizhai (2025).**
> *Neural network-based estimation of sample mean and standard deviation from some quartiles.*
> *Journal of Systems Science & Complexity.[accepted]*

📄 Paper link:
[https://sysmath.cjoe.ac.cn/jssc/EN/abstract/abstract56237.shtml](https://sysmath.cjoe.ac.cn/jssc/EN/abstract/abstract56237.shtml)

---

## Notes and Limitations

* This method is **data-driven and highly extensible**, and performs well across a broad range of common distributions and practical scenarios.
* As with any model-based approach, estimation accuracy may degrade under:

  * Unusual data-generating mechanisms
  * Extremely irregular summary statistics

If you encounter such cases, we **strongly encourage you to contact us**.
📧 **[zhangqinyuan19@mails.ucas.ac.cn](mailto:zhangqinyuan19@mails.ucas.ac.cn)**
We actively expand the model library by training additional neural networks to improve performance in newly identified scenarios.

---

## Feedback and Bug Reports

We welcome all forms of feedback, including:

* Unexpected estimation behavior
* Potential bugs or numerical issues
* Suggestions for new distributional settings

Please contact the authors at:
📧 **[zhangqinyuan19@mails.ucas.ac.cn](mailto:zhangqinyuan19@mails.ucas.ac.cn)**

Your feedback will directly help improve the robustness and applicability of this tool.
