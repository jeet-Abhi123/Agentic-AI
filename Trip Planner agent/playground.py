import openai
from phi.agent import Agent
import phi.api
from phi.model.openai import OpenAIChat
from dotenv import load_dotenv
from phi.tools.calculator import Calculator
from phi.tools.exa import ExaTools
from phi.model.groq import Groq

import os
import phi
from phi.playground import Playground, serve_playground_app
# Load environment variables from .env file
load_dotenv()

phi.api=os.getenv("PHI_API_KEY")

# Calculator agent
Cal_agent = Agent(
    name = "Calculator",
    model = Groq(id="llama-3.1-70b-versatile"),
    tools = [
        Calculator(
            add = True,
            subtract = True,
            multiply = True,
            divide = True,
            exponentiate = True,
            factorial = True,
            is_prime = True,
            square_root = True
        )
    ],
    show_tool_calls = True,
    markdown = True
)

#trip planner
trip_agent = Agent(
    name = "Trip Planner Agent",
    description = "You help the user plan their trips",
    model = Groq(id="llama-3.1-70b-versatile"),
    instructions = [
        "You are a trip planning assistant that helps users create a personalized trip itinerary especially in India.",
        "Always mention the timeframe, location, and year provided by the user (e.g., '16–17 December 2023 in Bangalore').",
        "Recommendations should align with the specified dates.",
        "Provide responses in these sections: Events, Activities, Dining Options.",
        "- **Events**: Include name, date, time, location, a brief description, and booking links from platforms like BookMyShow or Insider.in.",
        "- **Activities**: Suggest engaging options with estimated time required, location, and additional tips (e.g., best time to visit).",
        "- **Dining Options**: Recommend restaurants with cuisine highlights and links to platforms like Zomato or Google Maps.",
        "Ensure all recommendations are for the current or future dates relevant to the query.",
        "If no specific data is available for the dates, suggest general activities or evergreen attractions in the city.",
    ],
    tools = [ExaTools()]
)

app=Playground(agents=[Cal_agent,trip_agent]).get_app()

if __name__=="__main__":
    serve_playground_app("playground:app",reload=True)

