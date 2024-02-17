import asyncio
import logging
from typing import List, Optional, Sequence, cast

from llama_index.bridge.pydantic import BaseModel, Field
from llama_index.async_utils import run_async_tasks
from llama_index.callbacks.base import CallbackManager
from llama_index.callbacks.schema import CBEventType, EventPayload
from llama_index.core import BaseQueryEngine
from llama_index.tools.query_engine import QueryEngineTool
from llama_index.response_synthesizers import get_response_synthesizer

from rag.core.service_context import ServiceContext
from rag.entity.node.base_node import NodeWithScore, TextNode
from rag.entity.retriever import QueryBundle
from rag.entity.synthesizer.types import RESPONSE_TYPE
from rag.entity.synthesizer.base_synthesizer import BaseSynthesizer
from rag.entity.prompt.mixin import PromptMixinType
from rag.components.sub_quetion import BaseQuestionGenerator, SubQuestion
from rag.components.sub_quetion.question_generator import LLMQuestionGenerator
from rag.rag_utils.utils import get_color_mapping, print_text

logger = logging.getLogger(__name__)


class SubQAPairData(BaseModel):
    """
    Pair of the sub question and optionally its answer (if its been answered yet).
    """

    sub_q: SubQuestion
    answer: Optional[str] = None
    sources: List[NodeWithScore] = Field(default_factory=list)


class SubQuestionQueryEngine(BaseQueryEngine):
    """Sub question query engine.

    A query engine that breaks down a complex query (e.g. compare and contrast) into
        many sub questions and their target query engine for execution.
        After executing all sub questions, all responses are gathered and sent to
        response synthesizer to produce the final response.

    Args:
        question_gen (BaseQuestionGenerator): A module for generating sub questions
            given a complex question and tools.
        response_synthesizer (BaseSynthesizer): A response synthesizer for
            generating the final response
        query_engine_tools (Sequence[QueryEngineTool]): Tools to answer the
            sub questions.
        verbose (bool): whether to print intermediate questions and answers.
            Defaults to True
        use_async (bool): whether to execute the sub questions with asyncio.
            Defaults to True
    """

    def __init__(
        self,
        question_gen: BaseQuestionGenerator,
        response_synthesizer: BaseSynthesizer,
        query_engine_tools: Sequence[QueryEngineTool],
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = True,
        use_async: bool = False,
    ) -> None:
        self._question_gen = question_gen
        self._response_synthesizer = response_synthesizer
        self._metadatas = [x.metadata for x in query_engine_tools]
        self._query_engines = {
            tool.metadata.name: tool.query_engine for tool in query_engine_tools
        }
        self._verbose = verbose
        self._use_async = use_async
        super().__init__(callback_manager)

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt sub-modules."""
        return {
            "question_gen": self._question_gen,
            "response_synthesizer": self._response_synthesizer,
        }

    @classmethod
    def from_defaults(
        cls,
        query_engine_tools: Sequence[QueryEngineTool],
        question_gen: Optional[BaseQuestionGenerator] = None,
        response_synthesizer: Optional[BaseSynthesizer] = None,
        service_context: Optional[ServiceContext] = None,
        verbose: bool = True,
        use_async: bool = True,
    ) -> "SubQuestionQueryEngine":
        callback_manager = None
        if service_context is not None:
            callback_manager = service_context.callback_manager
        elif len(query_engine_tools) > 0:
            callback_manager = query_engine_tools[0].query_engine.callback_manager

        # TODO: gen question
        question_gen = question_gen or LLMQuestionGenerator.from_defaults(
            service_context=service_context
        )

        synth = response_synthesizer or get_response_synthesizer(
            callback_manager=callback_manager,
            service_context=service_context,
            use_async=use_async,
        )

        return cls(
            question_gen,
            synth,
            query_engine_tools,
            callback_manager=callback_manager,
            verbose=verbose,
            use_async=use_async,
        )

    # main function
    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        with self.callback_manager.event(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as query_event:
            # gen sub_question and map color
            sub_questions = self._question_gen.generate(self._metadatas, query_bundle)
            colors = get_color_mapping([str(i) for i in range(len(sub_questions))])

            if self._verbose:
                print_text(f"Generated {len(sub_questions)} sub questions.\n")

            if self._use_async:
                tasks = [
                    self._aquery_subq(sub_q, color=colors[str(ind)])
                    for ind, sub_q in enumerate(sub_questions)
                ]

                qa_pairs_all = run_async_tasks(tasks)
                qa_pairs_all = cast(List[Optional[SubQAPairData]], qa_pairs_all)
            else:
                qa_pairs_all = [
                    self._query_subq(sub_q, color=colors[str(ind)])
                    for ind, sub_q in enumerate(sub_questions)
                ]

            # filter out sub questions that failed
            qa_pairs: List[SubQAPairData] = list(filter(None, qa_pairs_all))

            nodes = [self._construct_node(pair) for pair in qa_pairs]
            source_nodes = [node for qa_pair in qa_pairs for node in qa_pair.sources]

            response = self._response_synthesizer.synthesize(
                query=query_bundle,
                nodes=nodes,
                additional_source_nodes=source_nodes,
            )

            query_event.on_end(payload={EventPayload.RESPONSE: response})

        return response

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        with self.callback_manager.event(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as query_event:
            sub_questions = await self._question_gen.agenerate(
                self._metadatas, query_bundle
            )

            colors = get_color_mapping([str(i) for i in range(len(sub_questions))])

            if self._verbose:
                print_text(f"Generated {len(sub_questions)} sub questions.\n")

            tasks = [
                self._aquery_subq(sub_q, color=colors[str(ind)])
                for ind, sub_q in enumerate(sub_questions)
            ]

            qa_pairs_all = await asyncio.gather(*tasks)
            qa_pairs_all = cast(List[Optional[SubQAPairData]], qa_pairs_all)

            # filter out sub questions that failed
            qa_pairs: List[SubQAPairData] = list(filter(None, qa_pairs_all))

            nodes = [self._construct_node(pair) for pair in qa_pairs]
            source_nodes = [node for qa_pair in qa_pairs for node in qa_pair.sources]

            response = await self._response_synthesizer.asynthesize(
                query=query_bundle,
                nodes=nodes,
                additional_source_nodes=source_nodes,
            )

            query_event.on_end(payload={EventPayload.RESPONSE: response})

        return response

    def _construct_node(self, qa_pair: SubQAPairData) -> NodeWithScore:
        node_text = (
            f"Sub question: {qa_pair.sub_q.sub_question}\nResponse: {qa_pair.answer}"
        )
        return NodeWithScore(node=TextNode(text=node_text))

    async def _aquery_subq(
        self, sub_q: SubQuestion, color: Optional[str] = None
    ) -> Optional[SubQAPairData]:
        try:
            with self.callback_manager.event(
                CBEventType.SUB_QUESTION,
                payload={EventPayload.SUB_QUESTION: SubQAPairData(sub_q=sub_q)},
            ) as event:
                question = sub_q.sub_question
                query_engine = self._query_engines[sub_q.tool_name]

                if self._verbose:
                    print_text(f"[{sub_q.tool_name}] Q: {question}\n", color=color)

                response = await query_engine.aquery(question)
                response_text = str(response)

                if self._verbose:
                    print_text(f"[{sub_q.tool_name}] A: {response_text}\n", color=color)

                qa_pair = SubQAPairData(
                    sub_q=sub_q, answer=response_text, sources=response.source_nodes
                )

                event.on_end(payload={EventPayload.SUB_QUESTION: qa_pair})

            return qa_pair
        except ValueError:
            logger.warning(f"[{sub_q.tool_name}] Failed to run {question}")
            return None

    def _query_subq(
        self, sub_q: SubQuestion, color: Optional[str] = None
    ) -> Optional[SubQAPairData]:
        try:
            with self.callback_manager.event(
                CBEventType.SUB_QUESTION,
                payload={EventPayload.SUB_QUESTION: SubQAPairData(sub_q=sub_q)},
            ) as event:
                question = sub_q.sub_question
                query_engine = self._query_engines[sub_q.tool_name]

                if self._verbose:
                    print_text(f"[{sub_q.tool_name}] Q: {question}\n", color=color)

                response = query_engine.query(question)
                response_text = str(response)

                if self._verbose:
                    print_text(f"[{sub_q.tool_name}] A: {response_text}\n", color=color)

                qa_pair = SubQAPairData(
                    sub_q=sub_q, answer=response_text, sources=response.source_nodes
                )

                event.on_end(payload={EventPayload.SUB_QUESTION: qa_pair})

            return qa_pair
        except ValueError:
            logger.warning(f"[{sub_q.tool_name}] Failed to run {question}")
            return None