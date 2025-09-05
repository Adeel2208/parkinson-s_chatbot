from langchain.schema import HumanMessage
import re

def main():
    while True:
        question = input()
        if not question or question.lower() in ["exit", "quit", "bye"]:
            break
        inputs = {"messages": [HumanMessage(content=question)]}
        result = app.invoke(inputs)
        final_answer = result["messages"][-1].content
        match = re.search(r"Final Answer:(.*?)(?=Assistant:|$)", final_answer, re.DOTALL)
        if match:
            print(match.group(1).strip())
        else:
            print(final_answer)

if __name__ == "__main__":
    main()