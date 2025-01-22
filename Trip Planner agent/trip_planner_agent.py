from phi.agent import Agent
from phi.tools.calculator import Calculator
from phi.tools.exa import ExaTools
from phi.model.groq import Groq
import openai

import os
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Calculator agent
Cal_agent = Agent(
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
    description = "You help the user plan their trips",
    name = "TimeOut",
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

# multi_ai_agent
multi_ai_agent = Agent(
    team = [Cal_agent, trip_agent],
    model = Groq(id="llama-3.1-70b-versatile"),
    instructions = [
        "Always include sources",
        "Use table to display the data"
    ],
    show_tool_calls = True,
    markdown = True
)

multi_ai_agent.print_response("I want to plan my coming weekend filled with fun activities and romantic themed activities in Bangalore for 21 and 22 Jan 2025.", stream=True)

