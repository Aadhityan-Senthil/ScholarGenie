"""Base crew configuration for ScholarGenie CrewAI integration."""

import os
from typing import List, Dict, Any, Optional
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic


class BaseScholarGenieCrew:
    """
    Base class for ScholarGenie crews.

    Provides common configuration and utilities for all crews.
    """

    def __init__(
        self,
        use_anthropic: bool = True,
        model: str = None,
        temperature: float = 0.7,
        verbose: bool = True
    ):
        """
        Initialize base crew.

        Args:
            use_anthropic: Use Anthropic Claude vs OpenAI
            model: Specific model to use
            temperature: LLM temperature
            verbose: Enable verbose logging
        """
        self.use_anthropic = use_anthropic
        self.temperature = temperature
        self.verbose = verbose

        # Initialize LLM
        if use_anthropic:
            self.llm = ChatAnthropic(
                model=model or "claude-3-5-sonnet-20241022",
                temperature=temperature,
                anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
            )
        else:
            self.llm = ChatOpenAI(
                model=model or "gpt-4-turbo-preview",
                temperature=temperature,
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )

    def create_agent(
        self,
        role: str,
        goal: str,
        backstory: str,
        tools: List[Any] = None,
        allow_delegation: bool = True,
        memory: bool = True
    ) -> Agent:
        """
        Create a CrewAI agent with standard configuration.

        Args:
            role: Agent's role
            goal: Agent's goal
            backstory: Agent's backstory
            tools: List of tools available to agent
            allow_delegation: Whether agent can delegate
            memory: Enable agent memory

        Returns:
            Configured Agent instance
        """
        return Agent(
            role=role,
            goal=goal,
            backstory=backstory,
            tools=tools or [],
            llm=self.llm,
            allow_delegation=allow_delegation,
            verbose=self.verbose,
            memory=memory
        )

    def create_task(
        self,
        description: str,
        agent: Agent,
        expected_output: str,
        context: Optional[List[Task]] = None,
        output_file: Optional[str] = None
    ) -> Task:
        """
        Create a CrewAI task.

        Args:
            description: Task description
            agent: Agent to perform task
            expected_output: Description of expected output
            context: Previous tasks this task depends on
            output_file: Optional file to save output

        Returns:
            Configured Task instance
        """
        return Task(
            description=description,
            agent=agent,
            expected_output=expected_output,
            context=context or [],
            output_file=output_file
        )

    def create_crew(
        self,
        agents: List[Agent],
        tasks: List[Task],
        process: Process = Process.sequential,
        manager_llm: Any = None
    ) -> Crew:
        """
        Create a CrewAI crew.

        Args:
            agents: List of agents in crew
            tasks: List of tasks for crew
            process: Process type (sequential or hierarchical)
            manager_llm: LLM for manager (hierarchical only)

        Returns:
            Configured Crew instance
        """
        crew_config = {
            "agents": agents,
            "tasks": tasks,
            "process": process,
            "verbose": self.verbose
        }

        if process == Process.hierarchical and manager_llm:
            crew_config["manager_llm"] = manager_llm

        return Crew(**crew_config)

    def kickoff(self, crew: Crew, inputs: Dict[str, Any] = None) -> str:
        """
        Execute crew with inputs.

        Args:
            crew: Crew to execute
            inputs: Input variables for tasks

        Returns:
            Crew execution result
        """
        if inputs:
            return crew.kickoff(inputs=inputs)
        return crew.kickoff()
