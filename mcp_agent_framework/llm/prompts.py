"""
Prompt Management for MCP Agent Framework.

This module handles the creation and formatting of prompts for
various agent tasks and interactions with language models.
"""

from typing import Dict, Any, List, Optional, Union

def format_prompt(template: str, **kwargs: Any) -> str:
    """
    Format a prompt template with provided values.
    
    This function replaces placeholders in a template with
    their corresponding values.
    
    Args:
        template: Prompt template with {placeholders}
        **kwargs: Values to fill into the template
    
    Returns:
        str: Formatted prompt
    """
    return template.format(**kwargs)

def format_role_prompt(role: str, task: str) -> str:
    """
    Format a prompt with role and task information.
    
    This function creates a standard prompt format that includes
    the agent's role and the task to be performed.
    
    Args:
        role: Role of the agent (e.g., "developer", "researcher")
        task: Description of the task to perform
    
    Returns:
        str: Formatted prompt
    """
    return f"""<role>{role}</role>
<task>{task}</task>
<context>
As an expert in {role}, analyze and solve this task.
Provide a detailed and comprehensive solution.
</context>
"""

def format_task_decomposition_prompt(role: str, task: str) -> str:
    """
    Format a prompt for task decomposition.
    
    This function creates a prompt that instructs the LLM to
    decompose a complex task into smaller subtasks.
    
    Args:
        role: Role of the agent (e.g., "CEO", "project manager")
        task: Description of the task to decompose
    
    Returns:
        str: Formatted prompt for task decomposition
    """
    return f"""<role>{role}</role>
<task>Decompose the following task into smaller subtasks</task>
<task_description>{task}</task_description>

Break down this task into 2-5 sequential subtasks. For each subtask, provide:
1. A brief description
2. The specialized role best suited to handle it (e.g., developer, researcher, writer)
3. Any dependencies on other subtasks (by subtask number)

Format your response as a structured list of subtasks like this:
<subtasks>
  <subtask id="1">
    <description>Research existing solutions</description>
    <role>researcher</role>
    <dependencies></dependencies>
  </subtask>
  <subtask id="2">
    <description>Develop prototype</description>
    <role>developer</role>
    <dependencies>1</dependencies>
  </subtask>
</subtasks>
"""

def format_agent_selection_prompt(role: str, task: str, available_roles: List[str]) -> str:
    """
    Format a prompt for selecting an appropriate agent role.
    
    This function creates a prompt that helps determine which
    specialized agent role is best suited for a particular task.
    
    Args:
        role: Role of the current agent
        task: Description of the task to assign
        available_roles: List of available specialized roles
    
    Returns:
        str: Formatted prompt for agent selection
    """
    roles_list = "\n".join([f"- {r}" for r in available_roles])
    
    return f"""<role>{role}</role>
<task>Select the most appropriate role for handling this task</task>
<task_description>{task}</task_description>
<available_roles>
{roles_list}
</available_roles>

Based on the task description, select the most appropriate role from the available roles.
Consider the expertise required for the task and the specialization of each role.

Respond with just the selected role from the list above, without any additional text.
"""

def format_result_integration_prompt(role: str, main_task: str, subtask_results: Dict[str, str]) -> str:
    """
    Format a prompt for integrating subtask results.
    
    This function creates a prompt that instructs the LLM to
    combine the results of multiple subtasks into a coherent
    solution for the main task.
    
    Args:
        role: Role of the agent (e.g., "CEO", "project manager")
        main_task: Description of the main task
        subtask_results: Dictionary mapping subtask descriptions to their results
    
    Returns:
        str: Formatted prompt for result integration
    """
    # Format subtask results
    results_text = ""
    for i, (subtask, result) in enumerate(subtask_results.items(), 1):
        results_text += f"""<subtask_result id="{i}">
  <subtask>{subtask}</subtask>
  <result>{result}</result>
</subtask_result>
"""
    
    return f"""<role>{role}</role>
<task>Integrate the results of subtasks into a coherent solution</task>
<main_task>{main_task}</main_task>
<subtask_results>
{results_text}
</subtask_results>

You have received the results of several subtasks that were part of solving the main task.
Your job is to integrate these results into a coherent, comprehensive solution for the main task.

Analyze the relationships between the subtask results, resolve any conflicts or inconsistencies,
and synthesize a complete solution that addresses all aspects of the main task.

Format your response as a structured solution:
<solution>
  <summary>
    [Brief summary of the integrated solution]
  </summary>
  <details>
    [Detailed explanation of the solution, referencing the subtask results]
  </details>
  <recommendations>
    [Any recommendations or next steps]
  </recommendations>
</solution>
"""

def format_error_handling_prompt(role: str, task: str, error: str) -> str:
    """
    Format a prompt for handling task errors.
    
    This function creates a prompt that helps analyze and
    resolve errors that occurred during task execution.
    
    Args:
        role: Role of the agent
        task: Description of the task that failed
        error: Error message or description
    
    Returns:
        str: Formatted prompt for error handling
    """
    return f"""<role>{role}</role>
<task>Analyze and resolve an error that occurred during task execution</task>
<original_task>{task}</original_task>
<error>{error}</error>

An error occurred while executing the task described above.
Analyze the error message, identify the likely cause, and suggest solutions.

Format your response as:
<error_analysis>
  <cause>
    [Your analysis of what caused the error]
  </cause>
  <solutions>
    [Potential solutions to resolve the error]
  </solutions>
  <recommended_approach>
    [The best approach to resolve the error]
  </recommended_approach>
</error_analysis>
"""

def format_xml_content(tag: str, content: str) -> str:
    """
    Format content with XML-style tags.
    
    This function wraps content in the specified XML tag.
    
    Args:
        tag: XML tag name
        content: Content to wrap
    
    Returns:
        str: Content wrapped in XML tags
    """
    return f"<{tag}>{content}</{tag}>"

def format_system_prompt(role: str) -> str:
    """
    Format a system prompt for the LLM.
    
    This function creates a system prompt that sets the context
    for the LLM, defining its role and capabilities.
    
    Args:
        role: Role of the agent
    
    Returns:
        str: Formatted system prompt
    """
    return f"""You are an expert AI agent with specialized knowledge in {role}.
You communicate using a structured XML format for clarity and precision.
Always format your responses using appropriate XML tags.

When given a task, analyze it carefully and provide a detailed, well-reasoned solution.
If the task requires specialized knowledge outside your expertise, acknowledge this
and suggest how to obtain the needed information or expertise.

For complex problems, break them down into smaller, manageable components.
Think step by step, showing your reasoning process clearly."""

class PromptTemplate:
    """
    Template class for creating structured prompts.
    
    This class provides a reusable template for prompts,
    with placeholders that can be filled in with specific values.
    
    Attributes:
        template: The prompt template string
        required_fields: List of field names that must be provided
        optional_fields: Dictionary of field names to default values
    """
    
    def __init__(
        self, 
        template: str, 
        required_fields: List[str] = None, 
        optional_fields: Dict[str, str] = None
    ):
        """
        Initialize a new PromptTemplate instance.
        
        Args:
            template: The prompt template string with {placeholders}
            required_fields: List of field names that must be provided
            optional_fields: Dictionary of field names to default values
        """
        self.template = template
        self.required_fields = required_fields or []
        self.optional_fields = optional_fields or {}
    
    def format(self, **kwargs: Any) -> str:
        """
        Format the template with provided values.
        
        This method checks that all required fields are provided,
        applies default values for optional fields, and formats
        the template with the resulting values.
        
        Args:
            **kwargs: Values to fill into the template
        
        Returns:
            str: Formatted prompt
        
        Raises:
            ValueError: If any required fields are missing
        """
        # Check for missing required fields
        missing_fields = [field for field in self.required_fields if field not in kwargs]
        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")
        
        # Apply default values for optional fields
        values = self.optional_fields.copy()
        values.update(kwargs)
        
        # Format the template
        return self.template.format(**values)

# Common prompt templates
TASK_PROMPT = PromptTemplate(
    template="""<role>{role}</role>
<task>{task}</task>
<context>
As an expert in {role}, analyze and solve this task.
Provide a detailed and comprehensive solution.
</context>
""",
    required_fields=["role", "task"]
)

TASK_DECOMPOSITION_PROMPT = PromptTemplate(
    template="""<role>{role}</role>
<task>Decompose the following task into smaller subtasks</task>
<task_description>{task}</task_description>

Break down this task into {num_subtasks} sequential subtasks. For each subtask, provide:
1. A brief description
2. The specialized role best suited to handle it (e.g., developer, researcher, writer)
3. Any dependencies on other subtasks (by subtask number)

Format your response as a structured list of subtasks like this:
<subtasks>
  <subtask id="1">
    <description>Research existing solutions</description>
    <role>researcher</role>
    <dependencies></dependencies>
  </subtask>
  <subtask id="2">
    <description>Develop prototype</description>
    <role>developer</role>
    <dependencies>1</dependencies>
  </subtask>
</subtasks>
""",
    required_fields=["role", "task"],
    optional_fields={"num_subtasks": "2-5"}
)

AGENT_CREATION_PROMPT = PromptTemplate(
    template="""<role>{role}</role>
<task>Determine what specialized agents should be created for this task</task>
<task_description>{task}</task_description>

Based on the task description, determine what specialized agents should be created to handle this task.
For each agent, specify:
1. A short, descriptive name
2. The specialized role/expertise
3. Why this agent is needed for the task

Format your response as a structured list of agents like this:
<agents>
  <agent>
    <name>Research Agent</name>
    <role>researcher</role>
    <justification>Needed to gather information about existing solutions</justification>
  </agent>
  <agent>
    <name>Developer Agent</name>
    <role>developer</role>
    <justification>Needed to implement the technical solution</justification>
  </agent>
</agents>
""",
    required_fields=["role", "task"]
)
