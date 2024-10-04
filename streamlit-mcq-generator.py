import streamlit as st
import PyPDF2
import json
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
import re
import time
import openai

# Set up OpenAI API key
#openai_api_key = st.secrets["OPEN_AI_KEY"]

chat_model = ChatOpenAI(model="gpt-4o", temperature=0.2)

def extract_text_from_pdf(pdf_file):
    """Extracts and returns the full text from a PDF file."""
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() + "\n"
        return full_text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def make_api_call(messages, max_tokens, is_final_answer=False):
    for attempt in range(3):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.5
            )
            return json.loads(response.choices[0].message['content'])
        except json.JSONDecodeError:
            return {"title": "Parsing Error", "content": response.choices[0].message['content'], "next_action": "final_answer"}
        except Exception as e:
            if attempt == 2:
                if is_final_answer:
                    return {"title": "Error",
                            "content": f"Failed to generate final answer after 3 attempts. Error: {str(e)}"}
                else:
                    return {"title": "Error", "content": f"Failed to generate step after 3 attempts. Error: {str(e)}",
                            "next_action": "final_answer"}
            time.sleep(1)  # Wait for 1 second before retrying
def process_exam_with_openai(full_text, chat_model):
    """Processes the extracted text to identify and format MCQs using OpenAI."""
    # Define the system prompt
    system_prompt = """
    You are an advanced AI assistant that extracts and processes multiple-choice questions (MCQs) from text.
    Please extract the following for each identified question:

    1. The question stem.
    2. Four answer options (A, B, C, D).
    3. Metadata including:
       - Topic
       - Subtopic
       - Cognitive Level
       - Expected Solution Time

    Use the following JSON format for each question:
    {
        "question": "",
        "options": ["", "", "", ""],
        "metadata": {
            "topic": "",
            "subtopic": "",
            "grade_level": "9",
            "cognitive_level": "",
            "expected_solution_time": ""
        }
    }
    return only the JSON output with nothing else and ensure that the whole response is a proper json file that can be interpreted.
    """

    # Prepare the message with the extracted text
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Extract and format the MCQs from this text:\n{full_text}")
    ]

    # Get response from OpenAI
    response = chat_model(messages)
    response_content = response.content.strip()
    response_content = re.sub(r'```json\n', '', response_content)
    response_content = re.sub(r'\n```', '', response_content)
    #response_content = ['[',response_content,']']

    # Parse and display the formatted questions
    try:
        formatted_questions = json.loads(response_content)
        return formatted_questions
    except json.JSONDecodeError:
        print("Error: Unable to parse the JSON output from OpenAI.")
        print(f"Response Content: {response_content}")
        return []
    

def process_outline_with_openai(full_text, chat_model):
    """Processes the extracted text to identify and format MCQs using OpenAI."""
    # Define the system prompt
    system_prompt = """
    You are an advanced AI assistant that extracts and processes Course Outline from text.
    Please extract the following for each identified question:

    1. Main topics
    2. suptopics of the main topics
    3. Learning outcomes and brief content

    **output format**:
     topic:
        -subtopic:
            -learning outcomes:
    """

    # Prepare the message with the extracted text
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Extract and format the MCQs from this text:\n{full_text}")
    ]

    # Get response from OpenAI
    response = chat_model(messages)
    response_content = response.content.strip()
    response_content = re.sub(r'```json\n', '', response_content)
    response_content = re.sub(r'\n```', '', response_content)
    #response_content = ['[',response_content,']']

    return response_content



def generate_response(prompt):
    messages = [
        {"role": "system", "content": """You are an expert AI assistant with advanced reasoning capabilities. Your task is to provide detailed, step-by-step explanations of your thought process. For each step:

1. Provide a clear, concise title describing the current reasoning phase.
2. Elaborate on your thought process in the content section.
3. Decide whether to continue reasoning or provide a final answer.

Response Format:
Use JSON with keys: 'title', 'content', 'next_action' (values: 'continue' or 'final_answer')

Key Instructions:
- Employ at least 5 distinct reasoning steps.
- Acknowledge your limitations as an AI and explicitly state what you can and cannot do.
- Actively explore and evaluate alternative answers or approaches.
- Critically assess your own reasoning; identify potential flaws or biases.
- When re-examining, employ a fundamentally different approach or perspective.
- Utilize at least 3 diverse methods to derive or verify your answer.
- Incorporate relevant domain knowledge and best practices in your reasoning.
- Quantify certainty levels for each step and the final conclusion when applicable.
- Consider potential edge cases or exceptions to your reasoning.
- Provide clear justifications for eliminating alternative hypotheses.
"""},
        {"role": "user", "content": prompt},
        {"role": "assistant",
         "content": "Understood. I will now think through this step-by-step, following the given instructions and starting by decomposing the problem."}
    ]

    steps = []
    step_count = 1
    total_thinking_time = 0

    while True:
        start_time = time.time()
        step_data = make_api_call(messages, 1000)
        end_time = time.time()
        thinking_time = end_time - start_time
        total_thinking_time += thinking_time

        steps.append((f"Step {step_count}: {step_data['title']}", step_data['content'], thinking_time))

        messages.append({"role": "assistant", "content": json.dumps(step_data)})

        if step_data['next_action'] == 'final_answer':
            break

        step_count += 1

    # Generate final answer
    messages.append({"role": "user", "content": "Please provide the final answer based on your reasoning above."})

    start_time = time.time()
    final_data = make_api_call(messages, 200, is_final_answer=True)
    end_time = time.time()
    thinking_time = end_time - start_time
    total_thinking_time += thinking_time

    steps.append(("Final Answer", final_data['content'], thinking_time))

    return steps[-1][1]  # Return only the final answer content

def generate_question_stems(course_outline, example_exam=None, num_questions=5):
    prompt = f"""
Create challenging and high-quality {num_questions} question stems based on the following course outline topics and subtopics:
{course_outline}.

- Ensure questions are clear, concise, and target critical concepts in each subtopic.
- Use the example exam as a style reference to match the difficulty and question type, if available.
- Include a variety of question types (definition, calculation, application, and reasoning) to cover different cognitive levels.

### Output format:
  output should be regular text without any formats
  -Question <question number >: <question stem>
"""

    if example_exam:
        prompt += f"\nExample exam for reference:\n{json.dumps(example_exam, indent=4)}"

    response = generate_response(prompt)
    return response

def generate_answer_options(question_stems, example_exam=None):
    prompt = f"""
Create four answer options (A, B, C, D) for the following questions stems:
{question_stems}

**Requirements:**
- One correct answer and three high-quality distractors.
- Distractors should be plausible and designed to reveal common misconceptions or errors.
- Vary the difficulty of the options: at least one should be very close to the correct answer, and others should include common mistakes.
- Use the example exam as a reference for the complexity and style of options, if provided.

### Output format:
output should be regular text without any formats
  -Question <question number >: <question stem>
  -Option A: <option A>
  -Option B: <option B>
  -Option C: <option C>
  -Option D: <option D>
  -Correct Answer: <correct answer>
"""
    question_stems = generate_response(prompt)
    return question_stems

def review_and_improve(questions):
    prompt = f"""
Review the following questions for clarity, quality, and difficulty. Suggest improvements or refinements for each question:

{questions}

**Refinement Criteria:**
- Ensure that the question stem and options are free from ambiguity.
- Balance the difficulty: identify if any options are too obvious or too confusing.
- Improve the distractors if they do not adequately challenge the understanding of the topic.
- Adjust the language and format to ensure they align with a high-quality exam standard.

### Output format:
output should be regular text without any formats
  -Question <question number >: <question stem>
  -Option A: <option A>
  -Option B: <option B>
  -Option C: <option C>
  -Option D: <option D>
  -Correct Answer: <correct answer>
"""

    response = generate_response(prompt)
    return response

def add_metadata(questions, course_outline, grade_level="Grade 9"):
    prompt = f"""
Analyze the following multiple-choice questions and extract detailed metadata for each one. Ensure the metadata captures the cognitive complexity, educational objectives, topic relevance, and expected solution time.

**Instructions:**
1. **Cognitive Level**: Determine the cognitive level of the question using Bloom's Taxonomy (e.g., Knowledge, Comprehension, Application, Analysis, Synthesis, Evaluation). Base this on the type of thinking required to solve the question.
2. **Difficulty Level**: Classify the difficulty as one of the following:
   - **Easy**: Basic recall or simple calculations.
   - **Medium**: Requires reasoning or understanding of the concept.
   - **Hard**: Involves multi-step problem solving, advanced reasoning, or complex analysis.
3. **Topic and Subtopic**: Identify the specific topic and subtopic being tested (e.g., Topic: Algebra, Subtopic: Solving Quadratic Equations).
4. **Grade Level**: Specify the appropriate grade level for this question (e.g., Grade 7, Grade 9).
5. **Expected Solution Time**: Estimate the time required for a typical student to solve this question (e.g., "2 minutes").

**Input Questions:**
{questions}

**Output Format:**
output should be regular text without any formats (no json no html don't create any formats)
you must output with the following format:
-Question <question number >: <question stem>
  -Option A: <option A>
  -Option B: <option B>
  -Option C: <option C>
  -Option D: <option D>
  -Correct Answer: <correct answer>
  -Metadata:
    -Cognitive Level: <cognitive level>
    -Difficulty Level: <difficulty level>
    -Topic: <topic>
    -Subtopic: <subtopic>
    -Grade Level: <grade level>
    -Expected Solution Time: <expected solution time>
"""
    questions = generate_response(prompt)
    return questions

def main():
    st.title("Multiple Choice Question Generator")

    st.header("Upload Files")
    course_outline_file = st.file_uploader("Upload Course Outline (PDF)", type="pdf")
    example_exam_file = st.file_uploader("Upload Example Exam (PDF, optional)", type="pdf")

    if course_outline_file is not None:
        course_outline = extract_text_from_pdf(course_outline_file)
        course_outline = process_outline_with_openai(course_outline, chat_model)
        st.success("Course outline uploaded successfully!")

        example_exam = None
        if example_exam_file is not None:
            example_exam = extract_text_from_pdf(example_exam_file)
            example_exam = process_exam_with_openai(example_exam, chat_model)
            st.success("Example exam uploaded successfully!")
        st.write("number of questions to generate")
        num_questions = st.number_input("Number of Questions to Generate", min_value=1, max_value=40, value=5)    
        if st.button("Generate Questions"):
            with st.spinner("Generating questions..."):
                question_stems = generate_question_stems(course_outline, example_exam,num_questions=num_questions)
                st.subheader("Generated Question Stems:")
                st.text(question_stems)

                questions_with_options = generate_answer_options(question_stems, example_exam)
                st.subheader("Questions with Answer Options:")
                st.text(questions_with_options)

                refined_questions = review_and_improve(questions_with_options)
                st.subheader("Refined Questions:")
                st.text(refined_questions)

                questions_with_metadata = add_metadata(refined_questions, course_outline)
                st.subheader("Final Questions with Metadata:")
                st.text(questions_with_metadata)

if __name__ == "__main__":
    main()
