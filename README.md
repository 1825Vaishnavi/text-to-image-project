# Text to Image Generation with Stable Diffusion and CLIP

IE7615 Generative AI course project. We generate images from natural language captions using Stable Diffusion v1.5 and encode captions with CLIP, using a cleaned subset of the Flickr30k dataset.

## Team

| Member | Role |
| --- | --- |
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

## How to Run

1. Open `notebooks/IE7615_Text_to_Image_Project.ipynb` in Google Colab.
2. Go to Runtime, then Change runtime type, and select a T4 GPU.
3. Run all cells. The notebook mounts Google Drive, downloads the dataset, cleans it, builds the embeddings and generates the sample images.

## Next Steps

* Compute CLIP scores between generated images and their prompts
* Compare generated images with real Flickr30k images
* Explore fine tuning on our 2,000 pair subset
