# Extended Thinking Tips

Advanced strategies for getting the most out of Claude's extended thinking features.

## Technical Considerations

- **Minimum budget**: 1024 tokens. Start with minimum and increase based on task complexity.
- **For budgets above 32K**: Use batch processing to avoid networking issues (long-running requests may hit system timeouts).
- **Language**: Performs best in English; outputs can be any supported language.
- **Below minimum budget**: Use standard mode with traditional CoT prompting (`<thinking>` tags).

## Prompting Techniques

### Start with General Instructions

Claude often performs better with high-level instructions rather than step-by-step guidance. The model's creativity may exceed human ability to prescribe optimal thinking.

**Instead of:**
```
Think through this math problem step by step:
1. First, identify the variables
2. Then, set up the equation
3. Next, solve for x
```

**Try:**
```
Please think about this math problem thoroughly and in great detail.
Consider multiple approaches and show your complete reasoning.
Try different methods if your first approach doesn't work.
```

Claude can still follow complex structured steps when needed. Start with generalized instructions, read Claude's thinking output, then iterate with more specific steering as needed.

## Multishot Prompting with Extended Thinking

Multishot prompting works well. When you provide examples of how to think through problems, Claude follows similar reasoning patterns.

Include few-shot examples using XML tags like `<thinking>` or `<scratchpad>`:

```
I'm going to show you how to solve a math problem, then I want you to solve a similar one.

Problem 1: What is 15% of 80?
<thinking>
To find 15% of 80:
1. Convert 15% to a decimal: 15% = 0.15
2. Multiply: 0.15 × 80 = 12
</thinking>
The answer is 12.

Now solve this one:
Problem 2: What is 35% of 240?
```

Claude generalizes patterns to the formal extended thinking process. You may get better results giving Claude free rein to think as it deems best.

## Maximizing Instruction Following

Claude shows significantly improved instruction following with extended thinking:
- It reasons about instructions in the extended thinking block
- Then executes those instructions in the response

To maximize:
- Be clear and specific about what you want
- For complex instructions, break into numbered steps
- Allow enough budget to process instructions fully

## Debugging and Steering

Use Claude's thinking output to debug logic (not always perfectly reliable).

**Important**:
- Don't pass extended thinking back in user text blocks—doesn't improve performance and may degrade results
- Prefilling extended thinking is explicitly not allowed
- Manually changing output text after thinking blocks degrades results due to model confusion
- Standard assistant response prefill is still allowed when extended thinking is off

### Clean Responses

If Claude repeats extended thinking in output text:
```
Do not repeat your extended thinking. Only output the final answer.
```

## Long Outputs and Longform Thinking

### For Dataset Generation
```
Please create an extremely detailed table of...
```

### For Detailed Content Generation
- Increase both maximum extended thinking length AND explicitly request longer outputs
- For very long outputs (20,000+ words): Request detailed outline with word counts down to paragraph level, then have Claude index paragraphs to the outline

Don't push Claude to output more tokens for tokens' sake. Start small and increase as needed.

## Improved Consistency and Error Handling

Use natural language prompting:
- Ask Claude to verify work with simple tests before declaring complete
- Instruct the model to analyze whether previous steps achieved expected results
- For coding tasks, ask Claude to run through test cases in extended thinking

**Example:**
```
Write a function to calculate the factorial of a number.
Before you finish, please verify your solution with test cases for:
- n=0
- n=1
- n=5
- n=10
And fix any issues you find.
```

## Budget Recommendations

| Task Complexity | Starting Budget | Notes |
|-----------------|-----------------|-------|
| Simple reasoning | 1024 (minimum) | Basic math, straightforward logic |
| Medium complexity | 4K-8K | Multi-step analysis, coding tasks |
| Complex problems | 16K-32K | Deep research, complex debugging |
| Very complex | 32K+ | Use batch processing |

## Best Practices Summary

1. Start with minimum budget and increase as needed
2. Use high-level instructions first, then add specificity
3. Leverage multishot prompting with `<thinking>` examples
4. Ask for verification and self-checking
5. Don't pass thinking back to user turns
6. For clean outputs, explicitly request no repetition of thinking
7. Use batch processing for budgets above 32K
