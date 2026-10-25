"""
Self-Builder Module: BabyAGI-style Function Synthesis

Implements:
- Task analysis and decomposition
- Function synthesis based on requirements
- Automatic testing and validation
- Iterative improvement based on execution feedback
"""

from __future__ import annotations

import ast
import copy
import json
import logging
import re
import textwrap
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable
import asyncio

from .function_db import (
    FunctionDB,
    GeneratedFunction,
    FunctionResult,
    FunctionStatus,
    Priority,
)

logger = logging.getLogger(__name__)

_SANITIZE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')

_MAX_SYNTHESIS_HISTORY = 1000
_MAX_TASK_DESCRIPTION = 10000
_MAX_FEEDBACK = 5000


class TaskType(str, Enum):
    DATA_PROCESSING = "data_processing"
    FILE_OPERATION = "file_operation"
    CALCULATION = "calculation"
    TRANSFORMATION = "transformation"
    VALIDATION = "validation"
    COMMUNICATION = "communication"
    ANALYSIS = "analysis"
    UNKNOWN = "unknown"


class ComplexityLevel(str, Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


@dataclass
class TaskAnalysis:
    description: str
    task_type: TaskType
    complexity: ComplexityLevel
    required_inputs: list[str]
    expected_outputs: list[str]
    dependencies: list[str]
    constraints: list[str]
    keywords: list[str]
    suggested_patterns: list[str]
    confidence: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> dict:
        return {
            "description": self.description[:200],
            "task_type": self.task_type.value,
            "complexity": self.complexity.value,
            "required_inputs": list(self.required_inputs),
            "expected_outputs": list(self.expected_outputs),
            "dependencies": list(self.dependencies),
            "constraints": list(self.constraints),
            "keywords": list(self.keywords),
            "suggested_patterns": list(self.suggested_patterns),
            "confidence": round(self.confidence, 3),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class SynthesisRequest:
    task_analysis: TaskAnalysis
    existing_functions: list[str] = field(default_factory=list)
    template_hints: dict[str, Any] = field(default_factory=dict)
    priority: Priority = Priority.MEDIUM
    max_retries: int = 3
    timeout_seconds: float = 30.0


@dataclass
class SynthesisResult:
    success: bool
    function: GeneratedFunction | None = None
    error: str | None = None
    iterations: int = 0
    tests_passed: bool = False
    synthesis_time: float = 0.0


class SelfBuilder:
    """
    BabyAGI-style self-building function generator.
    
    Workflow:
    1. Analyze task to determine requirements
    2. Search for existing functions that match
    3. If not found, synthesize new function
    4. Test synthesis against task requirements
    5. Iterate and improve if needed
    """
    
    TASK_PATTERNS = {
        TaskType.DATA_PROCESSING: [
            r"(?:process|transform|parse|convert).*data",
            r"(?:filter|sort|group|aggregate)",
            r"(?:extract|map).*(?:from|to)",
            r"(?:clean|normalize|validate)",
        ],
        TaskType.FILE_OPERATION: [
            r"(?:read|write|create|delete).*file",
            r"(?:open|close|save|load)",
            r"(?:copy|move|rename)",
            r"(?:search|find).*(?:file|directory)",
        ],
        TaskType.CALCULATION: [
            r"(?:calculate|compute|evaluate)",
            r"(?:sum|subtract|multiply|divide)",
            r"(?:average|mean|median|mode)",
            r"(?:count|total|percentage)",
        ],
        TaskType.TRANSFORMATION: [
            r"(?:convert|transform)",
            r"(?:encode|decode)",
            r"(?:serialize|deserialize)",
            r"(?:format|resize)",
        ],
        TaskType.VALIDATION: [
            r"(?:validate|verify|check)",
            r"(?:assert|ensure)",
            r"(?:test|confirm)",
            r"(?:sanity|completeness)",
        ],
        TaskType.COMMUNICATION: [
            r"(?:send|receive|transmit)",
            r"(?:request|response)",
            r"(?:api|http|websocket)",
            r"(?:notify|alert)",
        ],
        TaskType.ANALYSIS: [
            r"(?:analyze|examine|investigate)",
            r"(?:detect|identify|recognize)",
            r"(?:pattern|trend|anomaly)",
            r"(?:report|summary|insight)",
        ],
    }
    
    COMPLEXITY_INDICATORS = {
        ComplexityLevel.SIMPLE: [
            r"(?:simple|basic|trivial)",
            r"^calculate \w+$",
            r"^get \w+$",
            r"^is \w+$",
        ],
        ComplexityLevel.MODERATE: [
            r"(?:with|including|having)",
            r"(?:multiple|several|many)",
            r"(?:combine|merge|join)",
            r"(?:while|until|before)",
        ],
        ComplexityLevel.COMPLEX: [
            r"(?:recursive|nested|hierarchical)",
            r"(?:asynchronous|parallel|concurrent)",
            r"(?:distributed|scalable)",
            r"(?:optimize|minimize|maximize)",
        ],
    }
    
    FUNCTION_TEMPLATES = {
        TaskType.CALCULATION: textwrap.dedent('''
            def {func_name}({params}) -> {return_type}:
                """
                {description}
                
                Args:
                    {param_docs}
                
                Returns:
                    {return_doc}
                """
                {body}
                return result
        '''),
        TaskType.DATA_PROCESSING: textwrap.dedent('''
            def {func_name}({params}) -> {return_type}:
                """
                {description}
                
                Args:
                    {param_docs}
                
                Returns:
                    {return_doc}
                """
                result = []
                for item in data:
                    {body}
                return result
        '''),
        TaskType.FILE_OPERATION: textwrap.dedent('''
            def {func_name}({params}) -> {return_type}:
                """
                {description}
                
                Args:
                    {param_docs}
                
                Returns:
                    {return_doc}
                """
                {imports}
                {body}
                return result
        '''),
        TaskType.VALIDATION: textwrap.dedent('''
            def {func_name}({params}) -> bool:
                """
                {description}
                
                Args:
                    {param_docs}
                
                Returns:
                    True if valid, False otherwise
                """
                {body}
                return result
        '''),
    }
    
    PATTERNS_BODY = {
        "sum": "result = sum(values)",
        "count": "result = len(items)",
        "average": "result = sum(values) / len(values) if values else 0",
        "max": "result = max(values) if values else None",
        "min": "result = min(values) if values else None",
        "filter": "result = [item for item in data if condition(item)]",
        "map": "result = [transform(item) for item in data]",
        "sort": "result = sorted(data, key=lambda x: x)",
    }
    
    def __init__(self, db: FunctionDB):
        self.db = db
        self._synthesis_history: list[SynthesisResult] = []
        self._template_cache: dict[str, str] = {}
        self._lock = threading.Lock()
    
    def analyze_task(self, task_description: str) -> TaskAnalysis:
        task_description = _SANITIZE_RE.sub('', str(task_description))[:_MAX_TASK_DESCRIPTION]
        task_lower = task_description.lower()
        
        task_type = self._detect_task_type(task_lower)
        complexity = self._detect_complexity(task_lower)
        
        inputs = self._extract_inputs(task_description)
        outputs = self._extract_outputs(task_description)
        dependencies = self._extract_dependencies(task_description)
        constraints = self._extract_constraints(task_description)
        keywords = self._extract_keywords(task_description)
        patterns = self._suggest_patterns(task_type, keywords)
        confidence = self._calculate_confidence(
            task_type, complexity, patterns
        )
        
        return TaskAnalysis(
            description=task_description,
            task_type=task_type,
            complexity=complexity,
            required_inputs=inputs,
            expected_outputs=outputs,
            dependencies=dependencies,
            constraints=constraints,
            keywords=keywords,
            suggested_patterns=patterns,
            confidence=confidence,
        )
    
    def _detect_task_type(self, task_lower: str) -> TaskType:
        scores = {}
        for task_type, patterns in self.TASK_PATTERNS.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, task_lower):
                    score += 1
            if score > 0:
                scores[task_type] = score
        
        if not scores:
            return TaskType.UNKNOWN
        
        return max(scores.keys(), key=lambda t: scores[t])
    
    def _detect_complexity(self, task_lower: str) -> ComplexityLevel:
        for level, patterns in self.COMPLEXITY_INDICATORS.items():
            for pattern in patterns:
                if re.search(pattern, task_lower):
                    return level
        return ComplexityLevel.MODERATE
    
    def _extract_inputs(self, description: str) -> list[str]:
        inputs = []
        input_patterns = [
            r"(?:input|param|arg|value)s?[:\s]+([\w,\s]+)",
            r"(?:given|provided|from)\s+([\w\s]+)",
            r"(?:take|accept|receive)s?\s+(?:a\s+)?([\w\s]+)",
        ]
        
        for pattern in input_patterns:
            matches = re.findall(pattern, description, re.IGNORECASE)
            for match in matches:
                items = [i.strip() for i in match.split(",")]
                inputs.extend(items)
        
        return list(set(inputs))
    
    def _extract_outputs(self, description: str) -> list[str]:
        outputs = []
        output_patterns = [
            r"(?:output|result|return|response)s?[:\s]+([\w,\s]+)",
            r"(?:yield|produce|generate)s?\s+(?:a\s+)?([\w\s]+)",
            r"(?:return|give)s?\s+(?:the\s+)?([\w\s]+)",
        ]
        
        for pattern in output_patterns:
            matches = re.findall(pattern, description, re.IGNORECASE)
            for match in matches:
                items = [i.strip() for i in match.split(",")]
                outputs.extend(items)
        
        return list(set(outputs))
    
    def _extract_dependencies(self, description: str) -> list[str]:
        deps = []
        import re as local_re
        import_patterns = [
            r"(?:import|use|require|need)s?\s+(\w+)",
            r"using\s+(\w+)",
            r"from\s+(\w+)\s+import",
        ]
        
        for pattern in import_patterns:
            matches = local_re.findall(pattern, description, local_re.IGNORECASE)
            deps.extend(matches)
        
        standard_libs = {
            "os", "sys", "json", "re", "datetime", "time", 
            "pathlib", "collections", "itertools", "functools",
            "typing", "dataclasses", "enum", "logging", "asyncio",
        }
        deps = [d for d in deps if d in standard_libs]
        
        return list(set(deps))
    
    def _extract_constraints(self, description: str) -> list[str]:
        constraints = []
        constraint_patterns = [
            r"(?:constraint|limit|requirement)[:\s]+([\w\s]+)",
            r"(?:must|should|need to)\s+(?:be\s+)?([\w\s]+)",
            r"(?:cannot|must not|should not)\s+(?:be\s+)?([\w\s]+)",
        ]
        
        for pattern in constraint_patterns:
            matches = re.findall(pattern, description, re.IGNORECASE)
            constraints.extend(m.strip() for m in matches)
        
        return constraints
    
    def _extract_keywords(self, description: str) -> list[str]:
        stop_words = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after",
            "above", "below", "between", "under", "over", "again",
            "further", "then", "once", "here", "there", "when", "where",
            "why", "how", "all", "each", "few", "more", "most", "other",
            "some", "such", "no", "nor", "not", "only", "own", "same",
            "so", "than", "too", "very", "can", "and", "or", "but",
        }
        words = re.findall(r"\b[a-zA-Z]{3,}\b", description.lower())
        return [w for w in words if w not in stop_words]
    
    def _suggest_patterns(self, task_type: TaskType, keywords: list[str]) -> list[str]:
        patterns = []
        
        for keyword in keywords:
            if keyword in self.PATTERNS_BODY:
                patterns.append(keyword)
        
        if not patterns:
            if task_type == TaskType.CALCULATION:
                patterns = ["calculate"]
            elif task_type == TaskType.DATA_PROCESSING:
                patterns = ["transform"]
            elif task_type == TaskType.VALIDATION:
                patterns = ["check"]
        
        return patterns
    
    def _calculate_confidence(
        self,
        task_type: TaskType,
        complexity: ComplexityLevel,
        patterns: list[str]
    ) -> float:
        if task_type == TaskType.UNKNOWN:
            return 0.3
        
        base_confidence = 0.5
        
        complexity_modifiers = {
            ComplexityLevel.SIMPLE: 0.3,
            ComplexityLevel.MODERATE: 0.1,
            ComplexityLevel.COMPLEX: -0.1,
        }
        
        confidence = (
            base_confidence + 
            complexity_modifiers.get(complexity, 0) +
            min(0.2, len(patterns) * 0.05)
        )
        
        return min(1.0, max(0.1, confidence))
    
    def find_existing_function(
        self,
        analysis: TaskAnalysis
    ) -> GeneratedFunction | None:
        tags = analysis.keywords + [analysis.task_type.value]
        
        existing = self.db.search_by_tags(tags, match_all=False)
        for func in existing:
            if func.status in (FunctionStatus.APPROVED, FunctionStatus.TESTING):
                if self._function_matches_analysis(func, analysis):
                    return func
        
        return None
    
    def _function_matches_analysis(
        self,
        func: GeneratedFunction,
        analysis: TaskAnalysis
    ) -> bool:
        if analysis.complexity == ComplexityLevel.SIMPLE:
            if func.avg_execution_time > 0.1:
                return False
        
        if analysis.task_type.value in func.tags:
            return True
        
        for keyword in analysis.keywords[:3]:
            if keyword in func.name.lower() or keyword in func.description.lower():
                return True
        
        return False
    
    def synthesize_function(
        self,
        request: SynthesisRequest
    ) -> SynthesisResult:
        start_time = time.perf_counter()
        
        analysis = request.task_analysis
        
        existing = self.find_existing_function(analysis)
        if existing:
            self.db.register(existing)
            return SynthesisResult(
                success=True,
                function=existing,
                iterations=0,
                tests_passed=True,
                synthesis_time=time.perf_counter() - start_time,
            )
        
        for iteration in range(request.max_retries):
            try:
                func = self._generate_function(analysis, iteration)
                
                if func is None:
                    continue
                
                tests_passed = self._validate_function(func, analysis)
                
                if tests_passed:
                    func.status = FunctionStatus.APPROVED
                    self.db.register(func)
                    
                    result = SynthesisResult(
                        success=True,
                        function=func,
                        iterations=iteration + 1,
                        tests_passed=True,
                        synthesis_time=time.perf_counter() - start_time,
                    )
                    with self._lock:
                        if len(self._synthesis_history) >= _MAX_SYNTHESIS_HISTORY:
                            self._synthesis_history.pop(0)
                        self._synthesis_history.append(result)
                    return result
                else:
                    func.status = FunctionStatus.FAILED
                    self.db.register(func)
                    
            except Exception as e:
                logger.warning(f"Synthesis iteration {iteration} failed: {e}")
                continue
        
        result = SynthesisResult(
            success=False,
            error=f"Failed after {request.max_retries} iterations",
            iterations=request.max_retries,
            synthesis_time=time.perf_counter() - start_time,
        )
        with self._lock:
            if len(self._synthesis_history) >= _MAX_SYNTHESIS_HISTORY:
                self._synthesis_history.pop(0)
            self._synthesis_history.append(result)
        return result
    
    def _generate_function(
        self,
        analysis: TaskAnalysis,
        iteration: int
    ) -> GeneratedFunction | None:
        func_name = self._generate_name(analysis, iteration)
        params = self._generate_params(analysis)
        return_type = self._infer_return_type(analysis)
        body = self._generate_body(analysis, iteration)
        
        if not body:
            return None
        
        description = analysis.description
        imports = "\n".join(f"import {dep}" for dep in analysis.dependencies)
        if imports:
            body = imports + "\n\n" + body
        
        return GeneratedFunction(
            name=func_name,
            description=description,
            code=f"def {func_name}({params}) -> {return_type}:\n{textwrap.indent(body, '    ')}",
            dependencies=analysis.dependencies,
            parameters={p.split(":")[0]: p.split(":")[-1] if ":" in p else "Any" for p in params.split(", ") if p},
            returns=return_type,
            status=FunctionStatus.TESTING,
            tags=analysis.keywords + [analysis.task_type.value],
        )
    
    def _generate_name(self, analysis: TaskAnalysis, iteration: int) -> str:
        base_words = analysis.keywords[:3]
        base_name = "_".join(base_words)
        
        name = f"{analysis.task_type.value}_{base_name}"
        if iteration > 0:
            name = f"{name}_v{iteration}"
        
        name = name.replace("__", "_").rstrip("_")
        name = re.sub(r"[^a-z0-9_]", "", name)
        name = re.sub(r"_+", "_", name)
        
        return name or f"generated_func_{iteration}"
    
    def _generate_params(self, analysis: TaskAnalysis) -> str:
        if not analysis.required_inputs:
            return ""
        
        params = []
        for inp in analysis.required_inputs[:5]:
            param_name = re.sub(r"[^a-z0-9_]", "_", inp.lower()).replace("__", "_").strip("_")
            params.append(f"{param_name}: Any")
        
        return ", ".join(params)
    
    def _infer_return_type(self, analysis: TaskAnalysis) -> str:
        if analysis.task_type == TaskType.VALIDATION:
            return "bool"
        elif analysis.task_type in (TaskType.CALCULATION,):
            return "float | int"
        elif analysis.task_type == TaskType.DATA_PROCESSING:
            return "list[Any]"
        elif analysis.task_type == TaskType.FILE_OPERATION:
            return "str | bytes | None"
        else:
            return "Any"
    
    def _generate_body(
        self,
        analysis: TaskAnalysis,
        iteration: int
    ) -> str:
        for pattern in analysis.suggested_patterns:
            if pattern in self.PATTERNS_BODY:
                return self.PATTERNS_BODY[pattern]
        
        if iteration == 0:
            return "result = None  # TODO: Implement"
        elif iteration == 1:
            return 'raise NotImplementedError("Function not yet implemented")'
        else:
            return "pass"
    
    def _validate_function(
        self,
        func: GeneratedFunction,
        analysis: TaskAnalysis
    ) -> bool:
        try:
            ast.parse(func.code)
        except SyntaxError:
            return False
        
        callable_func = func.get_callable()
        if callable_func is None:
            return False
        
        try:
            if analysis.task_type == TaskType.VALIDATION:
                test_inputs = [True, False, None, "", 0, 1]
                for inp in test_inputs:
                    try:
                        result = callable_func(inp)
                        if not isinstance(result, bool):
                            return False
                    except TypeError:
                        pass
            elif analysis.task_type == TaskType.CALCULATION:
                test_inputs = [[1, 2, 3], [10, 20, 30], []]
                for inp in test_inputs:
                    try:
                        result = callable_func(inp)
                        if not isinstance(result, (int, float)):
                            return False
                    except TypeError:
                        pass
            
            return True
            
        except Exception as e:
            logger.debug(f"Function validation failed: {e}")
            return False
    
    async def synthesize_async(
        self,
        request: SynthesisRequest
    ) -> SynthesisResult:
        return await asyncio.get_running_loop().run_in_executor(
            None, self.synthesize_function, request
        )
    
    def improve_function(
        self,
        function_id: str,
        feedback: str
    ) -> GeneratedFunction | None:
        function_id = _SANITIZE_STRICT_RE.sub('', str(function_id))[:100]
        feedback = _SANITIZE_RE.sub('', str(feedback))[:_MAX_FEEDBACK]
        original = self.db.get(function_id)
        if original is None:
            return None
        
        analysis = self.analyze_task(f"Improve: {original.description}. Feedback: {feedback}")
        
        request = SynthesisRequest(
            task_analysis=analysis,
            existing_functions=[function_id],
            priority=Priority.HIGH,
        )
        
        result = self.synthesize_function(request)
        
        if result.success and result.function:
            result.function.parent_id = function_id
            result.function.name = original.name
            result.function.version = original.version + 1
            self.db.register(result.function)
            return result.function
        
        return None
    
    def get_synthesis_history(self) -> list[SynthesisResult]:
        with self._lock:
            return copy.deepcopy(self._synthesis_history)
    
    def get_statistics(self) -> dict:
        with self._lock:
            history = list(self._synthesis_history)

        total = len(history)
        successful = sum(1 for r in history if r.success)
        total_iterations = sum(r.iterations for r in history)
        total_time = sum(r.synthesis_time for r in history)

        return {
            "total_syntheses": total,
            "successful": successful,
            "success_rate": successful / total if total > 0 else 0.0,
            "total_iterations": total_iterations,
            "avg_iterations": total_iterations / total if total > 0 else 0.0,
            "total_time": total_time,
            "avg_time": total_time / total if total > 0 else 0.0,
        }