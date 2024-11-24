# Improving Robustness of Quantitative Susceptibility Mapping (QSM)

### *Codes and results of improving noise robustness of QSM generation using a cascade model, comprising QSMnet and Unet-based denoisers.*

#### Flow of the study
![image](https://github.com/user-attachments/assets/d6fd97ae-f891-4f32-af19-c4c31e38805c)


#### Denoiser candidates
![image](https://github.com/user-attachments/assets/76039e61-4f6d-4356-8612-fd9c2dad9662)


#### Results
- The **cascade model generated higher-quality QSM** compared to QSMnet-only model **when the input local field maps are noisy.**
- Among various Unet-based denoisers, **3D Unet with L1+SSIM loss term** showed the best performance.
- However, **with noiseless inputs**, **QSMnet-only model performed the best.** 
- These findings underscore the effectiveness of 3D Unet for MR image denoising and highlight the importance of incorporating denoising methods when dealing with potentially high-noise images.
