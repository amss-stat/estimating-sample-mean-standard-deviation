export const MODEL_CONFIG = {
    s1: {
        beta:     { mu: 'mu_s1_beta_model.onnx',     sigma: 'sigma_s1_beta_model.onnx' },
        lognormal:{ mu: 'mu_s1_lognormal_model.onnx',sigma: 'sigma_s1_lognormal_model.onnx' },
        weibull:  { mu: 'mu_s1_weibull_model.onnx',  sigma: 'sigma_s1_weibull_model.onnx' },
        normal:   {                                 sigma: 'sigma_s1_normal_model.onnx' },
        exp:      { mu: 's1_exp_model.onnx' }
    },
    s2: {
        beta:     { mu: 'mu_s2_beta_model.onnx',     sigma: 'sigma_s2_beta_model.onnx' },
        lognormal:{ mu: 'mu_s2_lognormal_model.onnx',sigma: 'sigma_s2_lognormal_model.onnx' },
        weibull:  { mu: 'mu_s2_weibull_model.onnx',  sigma: 'sigma_s2_weibull_model.onnx' },
        normal:   {                                 sigma: 'sigma_s2_normal_model.onnx' },
        exp:      { mu: 's2_exp_model.onnx' }
    },
    s3: {
        beta:     { mu: 'mu_s3_beta_model.onnx',     sigma: 'sigma_s3_beta_model.onnx' },
        lognormal:{ mu: 'mu_s3_lognormal_model.onnx',sigma: 'sigma_s3_log_model.onnx' },
        weibull:  { mu: 'mu_s3_weibull_model.onnx',  sigma: 'sigma_s3_weibull_model.onnx' },
        normal:   {                                 sigma: 'sigma_s3_normal_model.onnx' },
        exp:      { mu: 's3_exp_model.onnx' }
    }
};

export const PREFERENCE_ORDER_DEFAULT = ['Normal', 'Log-Normal', 'Weibull', 'Exponential', 'Beta'];
