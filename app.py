from crewai import Crew, Agent, Task
from langchain_groq import ChatGroq
import os
from crewai_tools import ScrapeWebsiteTool
from textwrap import dedent

llm = ChatGroq(temperature=0.1, groq_api_key=os.environ["GROQ_API_KEY"], model_name="llama3-8b-8192")

Qiscraptool = ScrapeWebsiteTool(website_url="https://qi.iq/en/home")

question =  input(
    dedent("""
      Please provide your query?
   """))

Qi_web_agent = Agent(
    role='Website Agent',
    goal='Answer the questions based on the website content',
    backstory='Given a website url, you can professionally understand and answer the questions based on the website content',
    llm=llm,
    tools=[Qiscraptool],
    verbose=True
)

web_scrape_task = Task(
    description=dedent(f"""
        You have to reply to the user given the user has asked: {question}. The answer should 
        clear and understandable based on the website content. Explain with steps if needed.
          If you don't know the answer from provided content , politely say "I don't know" and provide contact details from web content  
        """),
    expected_output = dedent("""
                                 The output should be in markdown format.
                                 Explain the answer in simple language 
                                 Be clear and consise

                                 
        """),
    name="web_scrape",
    agent=Qi_web_agent
)



crew = Crew(
            agents=[Qi_web_agent],
            tasks=[web_scrape_task],
            full_output=False,
            verbose=True,
        )

result = crew.kickoff()

print(result)