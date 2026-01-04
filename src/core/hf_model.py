import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from .model import Model
import re
import string
import logging

logger = logging.getLogger(__name__)

def create_model(config_path, device=None):
    """
    Factory method to create a LLM instance
    """
    # config_path is actually config dict in our context mostly
    config = config_path 
    provider = config.get("model_info", {}).get("provider", "llama").lower()
    
    if provider == 'llama':
        model = Llama(config, device)
    else:
        # Fallback to Llama for this specific defense usage if undefined
        model = Llama(config, device)
    return model

class Llama(Model):
    def __init__(self, config, device):
        super().__init__(config, device)
        # Handle config differences
        params = config.get("params", {})
        self.max_output_tokens = int(params.get("max_output_tokens", 512))
        
        # Use simple name if provider/info struct not present
        if self.name == "unknown":
            self.name = config.get("model_path", "meta-llama/Llama-3.1-8B-Instruct")

        # Assume auth is handled by system unless specific token provided
        # api_pos = int(config["api_key_info"]["api_key_use"])
        # hf_token = config["api_key_info"]["api_keys"][api_pos]
        # login(token=hf_token)
        
        logger.info(f"Loading HF Model: {self.name} on {device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.name, 
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if device=="cuda" else None,
                attn_implementation="eager"
            )
            if device != "cuda" and device is not None:
                self.model.to(device)
            elif device is None and not torch.cuda.is_available():
                self.model.to("cpu")

            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.requires_grad_(False) # Freeze model weights
        except Exception as e:
            logger.error(f"Failed to load model {self.name}: {e}")
            raise e
        
    
    def query(self, msg, top_tokens=100000):
        # Ensure input is on correct device
        inputs = self.tokenizer(msg, padding=True, return_tensors="pt").to(self.model.device)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        with torch.no_grad():
            # When do_sample=False, we don't need temperature or top_p
            # This avoids warnings about incompatible generation flags
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=self.max_output_tokens,
                output_attentions=True,
                return_dict_in_generate=True,
                use_cache=True,
                do_sample=False, # Deterministic greedy decoding
            )

        out = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        # Be careful with slicing: ensure we only strip the input prompt
        # msg might be slightly different after tokenization/detokenization
        # Simple approach:
        result = out[len(self.tokenizer.decode(input_ids[0], skip_special_tokens=True)):]

        generated_ids = outputs.sequences[:, input_ids.shape[1]:]
        generated_tokens = self.tokenizer.convert_ids_to_tokens(generated_ids[0])
        punctuation_set = set(string.punctuation)
        total_attention_values = None
        
        # Calculate attention scores
        # outputs.attentions is a tuple (one per generated token)
        # Each element is a tuple (one per layer)
        # Shape: (batch_size, num_heads, sequence_length, sequence_length)
        
        # Iterating over generated tokens
        for i in range(len(generated_tokens) - 1):
            if i >= len(outputs.attentions):
                break
                
            # Stop at newline if needed (logic from snippet)
            if i != 0 and total_attention_values is not None and generated_tokens[i] == '<0x0A>':
                break
            
            clean_token = generated_tokens[i]
            # Clean token check
            if not clean_token or clean_token in punctuation_set or clean_token == self.tokenizer.eos_token or clean_token == '<0x0A>':
                continue

            # Stack attention scores from all layers for this token step `i`
            # outputs.attentions[i] is tuple of layers
            # We want the attention from the *last* token generated (the one at `i`) to all previous tokens
            # But the snippet does: attention_scores = torch.stack(outputs.attentions[i + 1]).mean(dim=0)
            # Providing standard implementation based on snippet logic:
            
            try:
                # Use i-th step attentions. 
                # Note: outputs.attentions[i] corresponds to the generation step for generated_ids[i]
                # The attention matrix size grows.
                
                # Snippet used `outputs.attentions[i+1]`. 0-th generation step usually has attentions too.
                # Let's trust snippet logic if possible, but safely.
                if i + 1 >= len(outputs.attentions):
                    continue
                    
                layer_attentions = outputs.attentions[i] 
                # Tuple of (batch, head, query_len, key_len). query_len is 1 for generation steps.
                
                # Average over layers
                # stack: (num_layers, batch, head, 1, seq_len)
                attention_tensor = torch.stack(layer_attentions).squeeze(3) # Remove query dim
                # formatting: (layers, batch, head, seq_len)
                
                attention_scores = attention_tensor.mean(dim=0) # Average over layers -> (batch, head, seq_len)
                avg_attention = attention_scores.mean(dim=1) # Average over heads -> (batch, seq_len)
                
                # Extract first (and only) batch element
                if avg_attention.dim() > 1:
                    avg_attention = avg_attention[0]
                
                # We care about attention to INPUT tokens
                token_attention = avg_attention[:input_ids.shape[1]]
                
                input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
                
                # Map back to input tokens
                current_values = []
                for j in range(len(input_tokens)):
                    if j < len(token_attention):
                         # Check if the value is a tensor and convert to scalar
                         att_val = token_attention[j]
                         if isinstance(att_val, torch.Tensor):
                             val = att_val.item()
                         else:
                             val = float(att_val)
                         # Zero out punctuation attention
                         if input_tokens[j] in punctuation_set:
                             val = 0
                         current_values.append(val)
                    else:
                        current_values.append(0)

                if total_attention_values is None:
                    total_attention_values = current_values
                else:
                    # element-wise sum
                    total_attention_values = [sum(x) for x in zip(total_attention_values, current_values)]

            except Exception as e:
                logger.warning(f"Error calculating attention for token {i}: {e}")
                continue
        
        # Map tokens back to passages using regex
        # Assuming msg has [1] ... format
        pattern = r'\[\d+\]\s+(.*?)(?=\n\[\d+\]|\n-{5,}|\Z)'
        matches = list(re.finditer(pattern, msg, re.DOTALL)) # Make list to be safe

        passage_indices = []
        for match in matches:
            passage_indices.append((match.start(), match.end()))
        
        if not passage_indices:
             # Fallback if no passages found
             return result, [], 0

        token_passage_indices = []
        # Re-tokenize to find token boundaries
        # This is expensive but necessary to map char indices to token indices
        # Optimization: Tokenize once and map.
        
        # Simplified mapping:
        # We need the token start/end index corresponding to char start/end
        # tokenizer(msg, return_offsets_mapping=True) is better but keeping snippet style with partial tokenization
        
        current_token_idx = 0
        
        # Note: The snippet re-tokenized prefixes. 
        # msg_tokens = self.tokenizer(msg[:passage_idx[1]], ...).input_ids.shape[1] is the End Token Index
        
        last_end_token = 0
        try:
             # Find start of first passage
             prefix = msg[:passage_indices[0][0]]
             prefix_ids = self.tokenizer(prefix, add_special_tokens=True).input_ids
             start = len(prefix_ids)
             
             for passage_idx in passage_indices:
                 # Passages include [1] header? Regex groups usually exclude it if using (.*?) but include in match.start()
                 # Snippet used match.start() and match.end() of the MATCH object.
                 # match.start() is start of "[1] content"
                 
                 substring_end = passage_idx[1]
                 sub_text = msg[:substring_end]
                 sub_ids = self.tokenizer(sub_text, add_special_tokens=True).input_ids
                 end = len(sub_ids)
                 
                 token_passage_indices.append((start, end))
                 start = end # Next passage starts after this one?
                 # Actually next passage matches start at match[i+1].start()
                 # So we need to calculate start for next loop
                 
                 # Wait, snippet logic:
                 # for passage_idx in passage_indices:
                 #   msg_tokens = ...
                 #   end = ...
                 #   append((start, end))
                 #   start = end 
                 # This implies passages are contiguous? They might not be.
                 # Correct logic:
                 # Start of passage P is determined by len(tokenize(msg[:P.start]))
                 # End of passage P is len(tokenize(msg[:P.end]))
                 
             # Re-doing loop correctly
             token_passage_indices = []
             for p_idx in passage_indices:
                 s_char, e_char = p_idx
                 s_ids = self.tokenizer(msg[:s_char], add_special_tokens=True).input_ids
                 e_ids = self.tokenizer(msg[:e_char], add_special_tokens=True).input_ids
                 token_passage_indices.append((len(s_ids), len(e_ids)))

        except Exception as e:
            logger.error(f"Error mapping tokens: {e}")
            pass

        passage_scores = []

        for token_idx in token_passage_indices:
            if total_attention_values is not None:
                #Bounds check
                start_t, end_t = token_idx
                if start_t < len(total_attention_values) and end_t <= len(total_attention_values):
                    attention_slice = total_attention_values[start_t: end_t]
                    if attention_slice:
                        top_alpha = sorted(attention_slice, reverse=True)[:min(top_tokens, len(attention_slice))] 
                        score = sum(top_alpha)
                    else:
                        score = 0
                else:
                    score = 0
            else:
                score = 1
            passage_scores.append(score)
            
        total_attention_score = sum(total_attention_values) if total_attention_values is not None else 10
        
        return result, passage_scores, total_attention_score
