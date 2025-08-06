"""
Aether Stub adapter for lm-evaluation-harness
"""

import json
import requests
import time
from typing import List, Optional, Union
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model


@register_model("aether_stub")
class AetherStubLM(LM):
    """
    Aether Stub Language Model adapter for lm-evaluation-harness.
    
    This adapter allows the evaluation harness to test the Aether runtime
    by converting evaluation requests into Aether DAG format and executing
    them through the Aether runtime API.
    """
    
    def __init__(
        self,
        aether_url: str = "http://localhost:3000",
        model_name: str = "gpt-4o",
        **kwargs
    ):
        super().__init__()
        self.aether_url = aether_url
        self.model_name = model_name
        self._rank = 0
        self._world_size = 1
        
    @property
    def eot_token_id(self):
        return 0
        
    @property
    def max_length(self):
        return 4096
        
    @property
    def max_gen_toks(self):
        return 1024
        
    @property
    def batch_size(self):
        return 1
        
    @property
    def device(self):
        return "cpu"
        
    def tok_encode(self, string: str, **kwargs) -> List[int]:
        """Dummy tokenization - just return character codes for simplicity"""
        return [ord(c) for c in string[:100]]  # Limit to 100 chars
        
    def tok_decode(self, tokens: List[int], **kwargs) -> str:
        """Dummy detokenization"""
        try:
            return ''.join(chr(t) for t in tokens if 0 <= t <= 127)
        except:
            return "TODO"
            
    def _model_call(self, inps: List[List[int]], **kwargs) -> List[List[float]]:
        """Not implemented for this stub"""
        raise NotImplementedError("Use generate_until instead")
        
    def generate_until(
        self, 
        requests: List[dict], 
        disable_tqdm: bool = False,
        **kwargs
    ) -> List[str]:
        """
        Generate responses using the Aether runtime.
        
        Args:
            requests: List of generation requests with 'context' and 'until' keys
            
        Returns:
            List of generated strings
        """
        results = []
        
        for request in requests:
            context = request.get('context', '')
            until = request.get('until', [])
            
            # Convert the request to an Aether DAG
            dag = self._create_dag_from_request(context, until)
            
            try:
                # Execute the DAG through Aether runtime
                response = requests.post(
                    f"{self.aether_url}/execute",
                    json=dag,
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )
                
                if response.status_code == 200:
                    result_data = response.json()
                    # Extract the output from the first LLM node result
                    llm_results = [r for r in result_data.get('results', []) 
                                 if r.get('node_id', '').startswith('llm_')]
                    if llm_results:
                        output = llm_results[0].get('output', 'TODO')
                    else:
                        output = 'TODO'
                else:
                    output = 'TODO'
                    
            except Exception as e:
                print(f"Error calling Aether runtime: {e}")
                output = 'TODO'
                
            results.append(output)
            
        return results
        
    def loglikelihood(
        self, 
        requests: List[tuple], 
        disable_tqdm: bool = False,
        **kwargs
    ) -> List[tuple]:
        """
        Calculate log-likelihood for the given requests.
        For this stub, we return dummy values.
        """
        results = []
        for context, continuation in requests:
            # Return dummy log-likelihood and is_greedy flag
            results.append((-1.0, False))
        return results
        
    def loglikelihood_rolling(
        self, 
        requests: List[tuple], 
        disable_tqdm: bool = False,
        **kwargs
    ) -> List[float]:
        """
        Calculate rolling log-likelihood.
        For this stub, we return dummy values.
        """
        return [-1.0] * len(requests)
        
    def _create_dag_from_request(self, context: str, until: List[str]) -> dict:
        """
        Convert an evaluation request into an Aether DAG format.
        """
        # Create a simple DAG with one LLM node
        prompt = f"Complete the following text: {context}"
        if until:
            prompt += f" Stop when you encounter: {', '.join(until)}"
            
        dag = {
            "nodes": [
                {
                    "id": "llm_generation",
                    "node_type": "llm_fn",
                    "prompt": prompt,
                    "model": self.model_name,
                    "dependencies": []
                }
            ]
        }
        
        return dag

