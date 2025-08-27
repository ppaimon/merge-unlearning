# TOFU Experiment Summary

- Generated at: `2025-08-27 17:44:21 +08`
- Repo root: `/scratch/wangtiantong/PUM`

| Task | Model | Method | Forget | Holdout | Retain | Forget Quality | Forget Truth Ratio | Model Utility | Privleak | Extraction Strength |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|
| tofu_Llama-3.1-8B-Instruct_forget01_GradAscent | Llama-3.1-8B-Instruct | GradAscent | forget01 |  |  | 0.0012708143485281624 | 0.5153671988966484 | 90.37499998192499 | 0.06982625666408267 |  |
| tofu_Llama-3.1-8B-Instruct_forget01_GradDiff | Llama-3.1-8B-Instruct | GradDiff | forget01 |  |  | 0.0005039436209702519 | 0.5411554646389389 | 90.24999998194998 | 0.07513289712620343 |  |
| tofu_Llama-3.2-1B-Instruct_forget01_DPO | Llama-3.2-1B-Instruct | DPO | forget01 |  |  | 0.01430154804770646 | 0.36768894408757047 | -80.04722548664988 | 0.14453427696414184 |  |
| tofu_Llama-3.2-1B-Instruct_forget01_GradAscent | Llama-3.2-1B-Instruct | GradAscent | forget01 |  |  | 0.054141077480362725 | 0.28140554616115865 | 78.15820541616844 | 0.035729959904470324 |  |
| tofu_Llama-3.2-1B-Instruct_forget01_GradDiff | Llama-3.2-1B-Instruct | GradDiff | forget01 |  |  | 0.7659314523482239 | 0.4161552492004518 | 75.20661155604124 | 0.04387899836600878 |  |
| tofu_Llama-3.2-1B-Instruct_forget01_NPO | Llama-3.2-1B-Instruct | NPO | forget01 |  |  | 0.7659314523482239 | 0.5490800025951961 | 45.92680046357934 | 0.06367313215876183 |  |
| tofu_Llama-3.2-1B-Instruct_forget01_RMU | Llama-3.2-1B-Instruct | RMU | forget01 |  |  | 0.16497269950224194 | 0.5464455898304593 | 85.7142856980941 | 0.02905940823391865 |  |
| tofu_Llama-3.2-1B-Instruct_forget05_DPO | Llama-3.2-1B-Instruct | DPO | forget05 |  |  | 1.3921047931216453e-06 | 0.26721066835398427 | -87.19024465382839 | 0.15542494992328515 |  |
| tofu_Llama-3.2-1B-Instruct_forget05_GradAscent | Llama-3.2-1B-Instruct | GradAscent | forget05 |  |  | 1.9426434495222354e-119 | 0.0 | 20.541875780968976 | 0.03265219234440614 |  |
| tofu_Llama-3.2-1B-Instruct_forget05_GradDiff | Llama-3.2-1B-Instruct | GradDiff | forget05 |  |  | 1.9426434495222354e-119 | 0.5146490962975375 | 56.83814302747208 | 0.03265219234440614 |  |
| tofu_Llama-3.2-1B-Instruct_forget05_NPO | Llama-3.2-1B-Instruct | NPO | forget05 |  |  | 0.1420746514551761 | 0.43818079751782746 | 17.428638642184975 | 0.06878069712869794 |  |
| tofu_Llama-3.2-1B-Instruct_forget05_RMU | Llama-3.2-1B-Instruct | RMU | forget05 |  |  | 1.2127544312107394e-10 | 0.5814533702453573 | 44.87923462282321 | 0.03265219234440614 |  |
| tofu_Llama-3.2-1B-Instruct_forget10_DPO | Llama-3.2-1B-Instruct | DPO | forget10 |  |  | 1.02724452984523e-14 | 0.36041510028449364 | -96.04482596488512 | 0.2208837617672043 |  |
| tofu_Llama-3.2-1B-Instruct_forget10_GradAscent | Llama-3.2-1B-Instruct | GradAscent | forget10 |  |  | 1.0635896769518578e-239 | 0.0 | -10.674009433188386 | 0.03250892997513522 |  |
| tofu_Llama-3.2-1B-Instruct_forget10_GradDiff | Llama-3.2-1B-Instruct | GradDiff | forget10 |  |  | 3.760175603932929e-219 | 0.4803424553179038 | 61.97282904788762 | 0.03250892997513522 |  |
| tofu_Llama-3.2-1B-Instruct_forget10_NPO | Llama-3.2-1B-Instruct | NPO | forget10 |  |  | 0.006260402267679663 | 0.5765062052109308 | 14.242473322792822 | 0.0757674859382988 |  |
| tofu_Llama-3.2-1B-Instruct_forget10_RMU | Llama-3.2-1B-Instruct | RMU | forget10 |  |  | 4.353260441808186e-19 | 0.5873371458351506 | 56.96888096098983 | 0.03250892997513522 |  |
| tofu_Llama-3.2-3B-Instruct_forget01_DPO | Llama-3.2-3B-Instruct | DPO | forget01 |  |  | 0.16497269950224194 | 0.5980418316210064 | -91.52542370812982 | 0.219327660862281 |  |
| tofu_Llama-3.2-3B-Instruct_forget01_GradAscent | Llama-3.2-3B-Instruct | GradAscent | forget01 |  |  | 0.01430154804770646 | 0.0893401181585673 | 97.45762709661975 | 0.033920519345029765 |  |
| tofu_Llama-3.2-3B-Instruct_forget01_GradDiff | Llama-3.2-3B-Instruct | GradDiff | forget01 |  |  | 0.5786001416508443 | 0.3843460838255215 | 89.8305084542756 | 0.043012887455255014 |  |
| tofu_Llama-3.2-3B-Instruct_forget01_NPO | Llama-3.2-3B-Instruct | NPO | forget01 |  |  | 0.9900193288833089 | 0.6359327036899843 | 68.07909602981263 | 0.07506677214955502 |  |
| tofu_Llama-3.2-3B-Instruct_forget01_RMU | Llama-3.2-3B-Instruct | RMU | forget01 |  |  | 0.2656871402817289 | 0.6499259228917077 | 82.90960450103736 | 0.029753852678363096 |  |
| tofu_Llama-3.2-3B-Instruct_forget05_DPO | Llama-3.2-3B-Instruct | DPO | forget05 |  |  | 6.734842895808746e-06 | 0.42463425916440534 | -94.07879307526322 | 0.20707627134397896 |  |
| tofu_Llama-3.2-3B-Instruct_forget05_GradAscent | Llama-3.2-3B-Instruct | GradAscent | forget05 |  |  | 1.9426434495222354e-119 | 0.0 | -23.745407641093493 | 0.03265219234440614 |  |
| tofu_Llama-3.2-3B-Instruct_forget05_GradDiff | Llama-3.2-3B-Instruct | GradDiff | forget05 |  |  | 1.0642884497984988e-106 | 0.6080764841992972 | 56.31595402855998 | 0.03265219234440614 |  |
| tofu_Llama-3.2-3B-Instruct_forget05_NPO | Llama-3.2-3B-Instruct | NPO | forget05 |  |  | 0.006094418258803505 | 0.48625672195631797 | 26.768545294025074 | 0.05726857542289255 |  |
| tofu_Llama-3.2-3B-Instruct_forget05_RMU | Llama-3.2-3B-Instruct | RMU | forget05 |  |  | 1.4275699621532978e-12 | 0.6690693248336371 | 52.696787297319325 | 0.03265219234440614 |  |
| tofu_Llama-3.2-3B-Instruct_forget10_GradAscent | Llama-3.2-3B-Instruct | GradAscent | forget10 |  |  | 1.0635896769518578e-239 | 0.0 | -15.379299672569037 | 0.03250892997513522 |  |
| tofu_Llama-3.2-3B-Instruct_forget10_GradDiff | Llama-3.2-3B-Instruct | GradDiff | forget10 |  |  | 1.404477461400576e-213 | 0.2981338396513453 | 65.02501159259039 | 0.03250892997513522 |  |
