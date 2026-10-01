import re
from samplers._types import Eval, EvalResult, MessageList, SamplerBase, SingleEvalResult
from typing import Any, Dict, Iterable, List, Optional, Tuple
from pathlib import Path


class MedValEval(Eval):
    def evaluate(self, generation_data: List[Dict]) -> List[SingleEvalResult]: 
        results = []
        for gen_data in generation_data:
            response_text = gen_data["response_text"]

            sections = [(None, [])]
            field_header_pattern = re.compile(r"\[\[ ## (.*?) ## \]\]")
            for line in response_text.splitlines():
                match = field_header_pattern.match(line.strip())
                if match:
                    sections.append((match.group(1), []))
                else:
                    sections[-1][1].append(line)
            
            pred = {}
            for field, val in sections:
                if field is None:
                    continue
                pred[field] = "\n".join(val).strip()
            
            is_correct = gen_data["row"]["physician_risk_grade"] == int(pred["risk_level"].strip())

            results.append(
                SingleEvalResult(
                    id=gen_data["id"],
                    score=float(is_correct),
                    metrics = {},  # TODO: have one for error type, one for risk level
                    gt_answer = gen_data["row"]["physician_risk_grade"],
                    pred_answer = response_text,
                    example_level_metadata={
                        "physician_risk_grade": gen_data["row"]["physician_risk_grade"],
                        "physician_error_assessment": gen_data["row"]["physician_error_assessment"],
                        "pred_risk_grade": pred["risk_level"],
                        "pred_error_assessment": pred["errors"]
                    }
                )
            )
        
        return results


        

    
    def __call__(self, sampler: SamplerBase) -> EvalResult:
        pass