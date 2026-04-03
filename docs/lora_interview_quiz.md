# LoRA Scaling & Architecture Quiz

Here is a Staff-level quiz designed to test your understanding of the low-level mechanics of LoRA and how it applies to this project's architecture.

## Question 1: The Matrix Fundamentals
In a standard LoRA implementation for a single linear projection (e.g., the `query` projection in attention), how many new weight matrices are introduced? If the original base weight matrix $W_0$ has dimensions `d x d`, and the LoRA rank is `r`, what are the dimensions of these new matrices?

## Question 2: The Initialization Trick
When training a model with LoRA from scratch, one of the LoRA matrices is initialized with a Gaussian distribution (random noise), while the other is initialized with zeros. 
* Which one is initialized with zeros (the "shrink" matrix $A$ or the "expand" matrix $B$)? 
* *Why* is it mathematically critical that it initializes to zero?

## Question 3: The Forward Pass (Standard vs. Repo Implementation)
Mathematically, the standard forward pass of a projection with an injected LoRA is: 
$h = W_0x + \Delta Wx$
* **Part A:** How is $\Delta W$ calculated using your LoRA matrices?
* **Part B:** In a typical single-tenant deployment, we often just do $W_{new} = W_0 + \Delta W$ before inference. Why does your `lora_scaling` repo explicitly avoid doing this, instead computing the $\Delta Wx$ path completely separately?

## Question 4: VRAM Implications
During fine-tuning, why does LoRA massively reduce the VRAM required compared to full fine-tuning? (Hint: It's not *just* because there are fewer parameter weights stored in memory).

## Question 5: Batched Inference (The Hard Part)
In your `lora_serving/ops/lora.py` file, you execute the LoRA forward pass using `torch.bmm` (Batch Matrix Multiplication). If you have a batch size of `B`, sequence length `S`, hidden dimension `H`, and rank `R`:
* What are the shapes of the two tensors being multiplied in your `shrink` step? 
* Why do you have to use `torch.bmm` instead of standard `torch.matmul` (`@`) when serving multiple tenants in the same batch?

---

*Write down your answers below!*
