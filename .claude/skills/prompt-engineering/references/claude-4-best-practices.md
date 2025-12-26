# Claude 4.x Best Practices

Detailed prompt engineering techniques specific to Claude 4.x models (Sonnet 4.5, Haiku 4.5, Opus 4.5).

## General Principles

### Be Explicit with Instructions

Claude 4.x models respond well to clear, explicit instructions. Being specific about desired output enhances results. Customers wanting "above and beyond" behavior from previous models should explicitly request these behaviors.

### Add Context to Improve Performance

Providing context or motivation behind instructions helps Claude 4.x understand your goals:

```
I need this analysis for a board presentation, so please focus on executive-level insights and keep technical jargon minimal.
```

### Be Vigilant with Examples

Claude 4.x pays close attention to details and examples. Ensure examples align with behaviors you want and minimize unintended patterns.

## Long-Horizon Reasoning

Claude 4.5 excels at tasks with exceptional state tracking. It maintains orientation across extended sessions by focusing on incremental progress—making steady advances on a few things at a time.

### Context Awareness

Claude 4.5 tracks its remaining context window ("token budget") throughout conversations. For agent harnesses with context compaction:

```xml
<context_management>
Your context window will be automatically compacted as it approaches its limit, allowing you to continue working indefinitely. Do not stop tasks early due to token budget concerns. Save progress and state before context refreshes. Always be persistent and autonomous—complete tasks fully even near budget limits.
</context_management>
```

## Multi-Context Window Workflows

### First Context Window Setup
- Set up framework (write tests, create setup scripts)
- Use subsequent windows to iterate on todo-list

### Structured Test Format
```
Have Claude create tests before starting work in structured format (e.g., tests.json).
Remind Claude: "It is unacceptable to remove or edit tests because this could lead to missing or buggy functionality."
```

### Setup Scripts
Encourage Claude to create setup scripts (e.g., `init.sh`) to gracefully start servers, run test suites, and linters.

### Starting Fresh vs Compacting
Consider starting with a brand new context rather than compaction. Claude 4.5 effectively discovers state from the local filesystem. Be prescriptive:
- "Call pwd; you can only read and write files in this directory."
- "Review progress.txt, tests.json, and the git logs."
- "Run integration tests before implementing new features."

### State Management

**Structured formats**: Use JSON for test results, task status
**Unstructured text**: Freeform progress notes work well for general progress
**Git**: Provides checkpoints that can be restored across sessions

## Communication Style

Claude 4.5 has a more concise, natural style:
- **Direct and grounded**: Fact-based progress reports rather than self-celebratory updates
- **Conversational**: More fluent and colloquial, less machine-like
- **Less verbose**: May skip detailed summaries unless prompted

### Balancing Verbosity

If you want updates as Claude works:
```
After completing a task that involves tool use, provide a quick summary of the work you've done.
```

## Tool Usage Patterns

Claude 4.5 follows instructions precisely. "Can you suggest some changes" may yield suggestions rather than implementations.

### For Proactive Action
```xml
<default_to_action>
By default, implement changes rather than only suggesting them. If the user's intent is unclear, infer the most useful likely action and proceed, using tools to discover any missing details instead of guessing.
</default_to_action>
```

### For Cautious Behavior
```xml
<do_not_act_before_instructions>
Do not jump into implementation or change files unless clearly instructed. When the user's intent is ambiguous, default to providing information, doing research, and providing recommendations rather than taking action.
</do_not_act_before_instructions>
```

### Tool Triggering

Claude Opus 4.5 is more responsive to system prompts. If prompts were designed to reduce undertriggering, Claude may now overtrigger. Dial back aggressive language:
- Instead of: "CRITICAL: You MUST use this tool when..."
- Try: "Use this tool when..."

## Output Format Control

### Tell Claude What to Do (Not What to Avoid)
- Instead of: "Do not use markdown in your response"
- Try: "Your response should be composed of smoothly flowing prose paragraphs."

### Use XML Format Indicators
```xml
<smoothly_flowing_prose_paragraphs>
Write the prose sections of your response here.
</smoothly_flowing_prose_paragraphs>
```

### Match Prompt Style to Output Style
The formatting style in your prompt influences Claude's response. Removing markdown from prompts can reduce markdown in outputs.

### Detailed Formatting Guidance
```xml
<avoid_excessive_markdown_and_bullet_points>
When writing reports, documents, technical explanations, analyses, or long-form content, write in clear, flowing prose using complete paragraphs and sentences. Use standard paragraph breaks. Reserve markdown primarily for `inline code`, code blocks, and simple headings.

Avoid **bold** and *italics*. DO NOT use ordered lists (1. ...) or unordered lists (*) unless presenting truly discrete items or the user explicitly requests a list.

Instead of listing items with bullets or numbers, incorporate them naturally into sentences. NEVER output a series of overly short bullet points.
</avoid_excessive_markdown_and_bullet_points>
```

## Research and Information Gathering

Claude 4.5 demonstrates exceptional agentic search capabilities.

```xml
<structured_research>
Search for this information in a structured way. As you gather data, develop several competing hypotheses. Track your confidence levels in progress notes to improve calibration. Regularly self-critique your approach and plan. Update a hypothesis tree or research notes file to persist information and provide transparency.
</structured_research>
```

## Subagent Orchestration

Claude 4.5 recognizes when tasks benefit from delegating to specialized subagents and does so proactively.

- Ensure well-defined subagent tools in tool definitions
- Let Claude orchestrate naturally
- Adjust conservativeness if needed:
```
Only delegate to subagents when the task clearly benefits from a separate agent with a new context window.
```

## Model Self-Knowledge

For correct self-identification:
```
The assistant is Claude, created by Anthropic. The current model is Claude Sonnet 4.5.
```

For LLM-powered apps:
```
When an LLM is needed, default to Claude Sonnet 4.5 unless the user requests otherwise. The exact model string is claude-sonnet-4-5-20250929.
```

## Thinking Sensitivity

When extended thinking is disabled, Claude Opus 4.5 is sensitive to "think" and variants. Replace with:
- "consider"
- "believe"
- "evaluate"

### Leveraging Thinking Capabilities
```
After receiving tool results, carefully reflect on their quality and determine optimal next steps before proceeding. Use your thinking to plan and iterate based on this new information, then take the best next action.
```

## Parallel Tool Calling

Claude 4.x excels at parallel execution, particularly Sonnet 4.5.

### Boost Parallelism
```xml
<use_parallel_tool_calls>
If you intend to call multiple tools and there are no dependencies between calls, make all independent calls in parallel. Prioritize simultaneous tool calls for actions that can be done in parallel. For example, when reading 3 files, run 3 tool calls in parallel. Never use placeholders or guess missing parameters.
</use_parallel_tool_calls>
```

### For Sequential Execution
```
Execute operations sequentially with brief pauses between each step to ensure stability.
```

## Reducing File Creation in Agentic Coding

Claude 4.x may create temporary files for testing/iteration. To minimize:
```
If you create any temporary new files, scripts, or helper files for iteration, clean up these files by removing them at the end of the task.
```

## Preventing Over-Engineering

Claude Opus 4.5 tends to overengineer:

```xml
<keep_solutions_minimal>
Avoid over-engineering. Only make changes directly requested or clearly necessary. Keep solutions simple and focused.

Don't add features, refactor code, or make "improvements" beyond what was asked. A bug fix doesn't need surrounding code cleaned up. A simple feature doesn't need extra configurability.

Don't add error handling, fallbacks, or validation for scenarios that can't happen. Trust internal code and framework guarantees. Only validate at system boundaries (user input, external APIs).

Don't create helpers, utilities, or abstractions for one-time operations. Don't design for hypothetical future requirements. The right amount of complexity is the minimum needed for the current task.
</keep_solutions_minimal>
```

## Frontend Design

Claude 4.x, particularly Opus 4.5, excels at building complex web applications. Without guidance, it defaults to generic patterns.

```xml
<frontend_aesthetics>
You tend to converge toward generic, "on distribution" outputs. In frontend design, this creates the "AI slop" aesthetic. Avoid this: make creative, distinctive frontends that surprise and delight.

Focus on:
- Typography: Choose beautiful, unique fonts. Avoid generic fonts like Arial and Inter.
- Color & Theme: Commit to a cohesive aesthetic. Use CSS variables. Dominant colors with sharp accents outperform timid palettes.
- Motion: Use animations for effects and micro-interactions. Focus on high-impact moments: one well-orchestrated page load with staggered reveals creates more delight than scattered micro-interactions.
- Backgrounds: Create atmosphere and depth rather than defaulting to solid colors.

Avoid:
- Overused font families (Inter, Roboto, Arial, system fonts)
- Clichéd color schemes (particularly purple gradients on white)
- Predictable layouts and component patterns
- Cookie-cutter design lacking context-specific character

Interpret creatively and make unexpected choices. Vary between light/dark themes, different fonts, different aesthetics.
</frontend_aesthetics>
```

## Avoiding Hard-Coding and Test-Focused Solutions

```xml
<general_solutions>
Write a high-quality, general-purpose solution using standard tools. Do not create helper scripts or workarounds. Implement a solution that works correctly for all valid inputs, not just test cases. Do not hard-code values.

Focus on understanding problem requirements and implementing the correct algorithm. Tests verify correctness—they don't define the solution. If the task is unreasonable or tests are incorrect, inform me rather than working around them.
</general_solutions>
```

## Encouraging Code Exploration

```xml
<explore_code_first>
ALWAYS read and understand relevant files before proposing code edits. Do not speculate about code you have not inspected. If the user references a specific file/path, you MUST open and inspect it before explaining or proposing fixes. Be rigorous and persistent in searching code for key facts. Thoroughly review the style, conventions, and abstractions before implementing new features.
</explore_code_first>
```

## Minimizing Hallucinations

```xml
<investigate_before_answering>
Never speculate about code you have not opened. If the user references a specific file, you MUST read the file before answering. Make sure to investigate and read relevant files BEFORE answering questions about the codebase. Never make claims about code before investigating unless you are certain—give grounded, hallucination-free answers.
</investigate_before_answering>
```

## Vision Capabilities

Claude Opus 4.5 has improved vision capabilities, particularly with multiple images. Effective technique: give Claude a crop tool to "zoom" in on relevant image regions.

## Document Creation

Claude 4.5 excels at creating presentations, animations, and visual documents with impressive creative flair and stronger instruction following.

```
Create a professional presentation on [topic]. Include thoughtful design elements, visual hierarchy, and engaging animations where appropriate.
```

## Migration Considerations

When migrating to Claude 4.5:

1. **Be specific about desired behavior**: Describe exactly what you'd like in the output
2. **Use quality modifiers**: "Create an analytics dashboard. Include as many relevant features and interactions as possible. Go beyond the basics to create a fully-featured implementation."
3. **Request features explicitly**: Animations and interactive elements should be requested when desired
