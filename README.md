# Text to Image Generation with Stable Diffusion and CLIP

IE7615 Generative AI course project. We generate images from natural language captions using Stable Diffusion v1.5 and encode captions with CLIP, using a cleaned subset of the Flickr30k dataset.

## Team

| Members |
| --- |
| Vaishnavi Gajarla 
| Pranathi Bombay  
| Varaalakshime Vigneswara Pandiarajan  
| Nikhil Bharadwaj Yellapragada  

## Dataset

* **Source:** Flickr30k from Hugging Face (`AminDehnavi/flickr30k`)
* **Subset:** 2,000 cleaned image caption pairs
* **Cleaning:** lowercase captions, trim and collapse spaces, remove captions under 4 words, one caption per image, remove duplicates
* **Images:** converted to RGB and resized to 512 x 512 with padding
* **Validation:** 2,000 rows, 0 missing captions, 0 duplicate captions

## Models

| Component | Model |
| --- | --- |
| Image generation | Stable Diffusion v1.5 (`runwayml/stable-diffusion-v1-5`) |
| Text embeddings | CLIP ViT B/32 (`openai/clip-vit-base-patch32`) |
| Text conditioning (Milestone 2) | CLIP ViT-L/14 text encoder built into Stable Diffusion v1.5 |
| Image to text scoring (Milestone 2) | CLIP ViT B/32 (`openai/clip-vit-base-patch32`) |

**Environment:** PyTorch, Diffusers, Transformers and Accelerate on Google Colab with a Tesla T4 GPU.

## Milestone 1 Results

* Cleaned 2,000 image caption pairs and saved them to `captions_clean.csv`
* Encoded all 2,000 captions into CLIP embeddings with shape (2000, 512)
* Tested similarity search: the query "a dog playing outside" returned 3 relevant dog captions
* Generated 5 sample images from text prompts (30 inference steps, guidance scale 7.5)

| No. | Prompt | Image |
| --- | --- | --- |
| 1 | a dog running through a green field | [sample_1.png](outputs/samples/sample_1.png) |
| 2 | a red car parked beside a building | [sample_2.png](outputs/samples/sample_2.png) |
| 3 | a child playing with a colorful ball | [sample_3.png](outputs/samples/sample_3.png) |
| 4 | a cat sleeping on a sofa | [sample_4.png](outputs/samples/sample_4.png) |
| 5 | a beautiful sunset over the ocean | [sample_5.png](outputs/samples/sample_5.png) |

## Milestone 2 Results

Notebook: `notebooks/IE7615_Milestone2.ipynb`. Outputs: `outputs/milestone2/`.

* Connected the CLIP ViT-L/14 text encoder to Stable Diffusion v1.5 by encoding prompts ourselves and passing the embeddings (1 x 77 x 768) to the UNet through cross attention. The embeddings match the pipeline's own encoding of the same text.
* Ran baseline conditional generation and tuned classifier free guidance, the sampler, the noise level spacing (Karras sigmas) and the number of steps
* Tested every setting on 5 Flickr30k captions with 3 seeds each (42, 7, 123), 210 unique images in total
* Scored every image with a CLIP score (cosine similarity between image and prompt x 100, using CLIP ViT B/32)
* Logged every run (settings, time and CLIP score) to [experiment_log.csv](outputs/milestone2/experiment_log.csv)

| Experiment | Settings tested | Mean CLIP score |
| --- | --- | --- |
| Baseline (PNDM, 30 steps, CFG 7.5) | 5 captions x 3 seeds | 31.32 to 32.99 per caption |
| Guidance scale | 1.0, 3.0, 5.0, 7.5, 10.0, 15.0 | 26.31 to 33.11, best at 10.0 |
| Sampler (30 steps) | PNDM, DDIM, Euler, Euler Karras, DPM++, DPM++ Karras | 32.30 to 33.30 |
| Steps (DPM++) | 10, 20, 30, 50 | 32.94 to 33.36, at 2.1 to 9.4 seconds per image |

Main findings:

* **Guidance has the largest effect.** Without guidance (CFG 1.0) the images barely follow the text. Scores peak at CFG 10, and above that colors become harsher.
* **The sampler matters little at 30 steps.** All six are within 1 point of each other, which is smaller than the variation between seeds. Karras sigmas made no measurable difference.
* **20 steps is the best trade-off.** It looks as good as 30 or 50 steps at less than half the time of 50.
* **The images get the scene right but often miss details** in the longer captions, such as clothing colors, actions and small objects.
* **Best setting:** DPM++, 20 steps, CFG 7.5 to 10.

| Output | Contents |
| --- | --- |
| [early_samples/](outputs/milestone2/early_samples) | The 10 early samples, the 2 best images by CLIP score for each caption |
| [early_samples_grid.png](outputs/milestone2/early_samples_grid.png) | The 10 early samples in one grid |
| [grid_baseline.png](outputs/milestone2/grid_baseline.png), [grid_cfg.png](outputs/milestone2/grid_cfg.png), [grid_scheduler.png](outputs/milestone2/grid_scheduler.png), [grid_steps.png](outputs/milestone2/grid_steps.png) | Image grids for each experiment |
| [metrics.png](outputs/milestone2/metrics.png) | CLIP score vs guidance, by sampler and by number of steps |
| [experiment_log.csv](outputs/milestone2/experiment_log.csv) | One row per generated image with its settings, time and CLIP score |

## Repository Structure

```
text-to-image-project/
├── notebooks/
│   └── IE7615_Text_to_Image_Project.ipynb   # full Milestone 1 notebook
├── outputs/
│   └── samples/                             # 5 generated images + sample_prompts.csv
├── data/                                    # processed data (images not uploaded, too large)
├── requirements.txt
├── proposal.pdf
└── README.md
```

Added in Milestone 2:

```
text-to-image-project/
├── notebooks/
│   └── IE7615_Milestone2.ipynb              # Milestone 2 experiments
└── outputs/
    └── milestone2/
        ├── early_samples/                   # 10 early samples
        ├── early_samples_grid.png
        ├── grid_baseline.png, grid_cfg.png, grid_scheduler.png, grid_steps.png
        ├── metrics.png
        └── experiment_log.csv
```

## How to Run

1. Open `notebooks/IE7615_Text_to_Image_Project.ipynb` in Google Colab.
2. Go to Runtime, then Change runtime type, and select a T4 GPU.
3. Run all cells. The notebook mounts Google Drive, downloads the dataset, cleans it, builds the embeddings and generates the sample images.

**Milestone 2**

1. Make sure `captions_clean.csv` from Milestone 1 is in `My Drive/IE7615_Project/`.
2. Open `notebooks/IE7615_Milestone2.ipynb` in Google Colab and select a T4 GPU.
3. Run all cells. The full run generates about 210 images in roughly 20 to 25 minutes and saves everything to `My Drive/IE7615_Project/milestone2_outputs/`.

## Next Steps

* Compare generated images with real Flickr30k images
* Explore fine tuning on our 2,000 pair subset
* Compute FID and Inception Score for the generated images (Milestone 3)
* Test the best settings as one combination (DPM++, 20 steps, CFG 7.5 to 10), since guidance was only tested with PNDM
