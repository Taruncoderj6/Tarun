import os
import uvicorn
import requests
import json

from fastapi import FastAPI
from langserve import add_routes
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda


# ============================================================
# 1. DEFINE TOOLS
# ============================================================

@tool
def search_movies(genre: str) -> str:
    """Search for Indian movies by genre."""

    movies = {
        "sci-fi": "Cargo, 2.0, Mr. India",
        "comedy": "3 Idiots, Hera Pheri, Munna Bhai M.B.B.S.",
        "action": "RRR, Vikram, Baahubali"
    }

    return movies.get(
        genre.lower(),
        "No Indian movies found for that genre"
    )


@tool
def change_to_f(temp_c: float) -> float:
    """Convert Celsius temperature to Fahrenheit."""

    return temp_c * 1.8 + 32


@tool
def get_weather(city: str) -> str:
    """Get current weather for an Indian city."""

    # --------------------------------------------------------
    # List of allowed Indian cities
    # --------------------------------------------------------

    indian_cities = [
        "hyderabad",
        "delhi",
        "mumbai",
        "bangalore",
        "bengaluru",
        "chennai",
        "kolkata",
        "pune",
        "ahmedabad",
        "jaipur",
        "lucknow",
        "kanpur",
        "nagpur",
        "indore",
        "bhopal",
        "visakhapatnam",
        "vizag",
        "vijayawada",
        "warangal",
        "tirupati",
        "goa",
        "surat",
        "patna",
        "ranchi",
        "kochi",
        "thiruvananthapuram",
        "mysore",
        "mysuru"
    ]

    city_lower = city.lower().strip()

    if city_lower not in indian_cities:
        return "Invalid input"

    # --------------------------------------------------------
    # Geocoding
    # --------------------------------------------------------

    geo_url = "https://geocoding-api.open-meteo.com/v1/search"

    geo_params = {
        "name": city,
        "count": 1,
        "language": "en",
        "format": "json"
    }

    try:

        geo_response = requests.get(
            geo_url,
            params=geo_params,
            timeout=10
        ).json()

        if "results" not in geo_response:
            return "Invalid input"

        location = geo_response["results"][0]

        latitude = location["latitude"]
        longitude = location["longitude"]

        # ----------------------------------------------------
        # Weather API
        # ----------------------------------------------------

        weather_url = "https://api.open-meteo.com/v1/forecast"

        weather_params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,weather_code",
            "temperature_unit": "celsius"
        }

        weather_response = requests.get(
            weather_url,
            params=weather_params,
            timeout=10
        ).json()

        current = weather_response["current"]

        result = {
            "resolved_city": location["name"],
            "temperature_celsius": current["temperature_2m"],
            "weather_code": current["weather_code"]
        }

        return json.dumps(result)

    except Exception:
        return "Invalid input"


# ============================================================
# 2. TOOLS LIST
# ============================================================

tools = [
    get_weather,
    search_movies,
    change_to_f
]


# ============================================================
# 3. GEMINI API KEY
# ============================================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY environment variable is not set."
    )


# ============================================================
# 4. INITIALIZE MODEL
# ============================================================

llm_flash = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    api_key=GEMINI_API_KEY,
    temperature=0
)


# ============================================================
# 5. CREATE AGENT
# ============================================================

agent = create_agent(
    model=llm_flash,
    tools=tools,

    system_prompt=(
        "You are a specialized AI agent. "

        "You are ONLY allowed to answer questions related to "
        "Indian weather and Indian movies/cinema. "

        "You can also perform Celsius to Fahrenheit conversion "
        "when the user asks for temperature conversion. "

        "Do not answer questions about programming, mathematics, "
        "science, technology, general knowledge, politics, "
        "sports, celebrities outside Indian cinema, or any other "
        "topic outside your defined capabilities. "

        "If the user asks something outside your capabilities, "
        "respond exactly with: Invalid input"
    )
)


# ============================================================
# 6. INPUT MODEL
# ============================================================

class AgentInput(BaseModel):

    input: str = Field(
        description="Your message to the agent"
    )


# ============================================================
# 7. INPUT VALIDATOR
# ============================================================

def is_valid_input(user_input: str) -> bool:

    text = user_input.lower().strip()

    # --------------------------------------------------------
    # Weather keywords
    # --------------------------------------------------------

    weather_keywords = [
        "weather",
        "temperature",
        "rain",
        "rainfall",
        "forecast",
        "climate",
        "humidity",
        "wind",
        "hot",
        "cold",
        "sunny",
        "cloudy",
        "storm"
    ]

    # --------------------------------------------------------
    # Indian cinema keywords
    # --------------------------------------------------------

    movie_keywords = [
        "movie",
        "movies",
        "film",
        "films",
        "cinema",
        "bollywood",
        "tollywood",
        "kollywood",
        "actor",
        "actress",
        "director",
        "producer",
        "movie genre",
        "indian movie",
        "indian cinema",

        # Some known movies
        "rrr",
        "bahubali",
        "baahubali",
        "vikram",
        "3 idiots",
        "hera pheri",
        "munna bhai",
        "cargo",
        "mr india"
    ]

    # --------------------------------------------------------
    # Temperature conversion keywords
    # --------------------------------------------------------

    conversion_keywords = [
        "celsius",
        "fahrenheit",
        "convert temperature",
        "temperature conversion"
    ]

    # --------------------------------------------------------
    # Check weather
    # --------------------------------------------------------

    for keyword in weather_keywords:

        if keyword in text:

            # Make sure it refers to India/Indian city
            indian_locations = [
                "india",
                "hyderabad",
                "delhi",
                "mumbai",
                "bangalore",
                "bengaluru",
                "chennai",
                "kolkata",
                "pune",
                "ahmedabad",
                "jaipur",
                "lucknow",
                "kanpur",
                "nagpur",
                "indore",
                "bhopal",
                "visakhapatnam",
                "vizag",
                "vijayawada",
                "warangal",
                "tirupati",
                "goa",
                "surat",
                "patna",
                "ranchi",
                "kochi",
                "mysore",
                "mysuru"
            ]

            for location in indian_locations:

                if location in text:
                    return True

    # --------------------------------------------------------
    # Check Indian cinema
    # --------------------------------------------------------

    for keyword in movie_keywords:

        if keyword in text:
            return True

    # --------------------------------------------------------
    # Check temperature conversion
    # --------------------------------------------------------

    for keyword in conversion_keywords:

        if keyword in text:
            return True

    # --------------------------------------------------------
    # Anything else = INVALID
    # --------------------------------------------------------

    return False


# ============================================================
# 8. FORMAT INPUT FOR AGENT
# ============================================================

def format_for_agent(x):

    user_input = (
        x["input"]
        if isinstance(x, dict)
        else x.input
    )

    # --------------------------------------------------------
    # Validate before sending to AI
    # --------------------------------------------------------

    if not is_valid_input(user_input):

        return {
            "messages": [
                (
                    "user",
                    "INVALID_INPUT"
                )
            ]
        }

    # --------------------------------------------------------
    # Valid input
    # --------------------------------------------------------

    return {
        "messages": [
            (
                "user",
                user_input
            )
        ]
    }


# ============================================================
# 9. EXTRACT AGENT RESPONSE
# ============================================================

def extract_text_response(agent_output):

    if not isinstance(agent_output, dict):

        return str(agent_output)

    messages = agent_output.get("messages")

    # --------------------------------------------------------
    # Check nested messages
    # --------------------------------------------------------

    if messages is None:

        for value in agent_output.values():

            if (
                isinstance(value, dict)
                and "messages" in value
            ):

                messages = value["messages"]

                break

    # --------------------------------------------------------
    # Extract final message
    # --------------------------------------------------------

    if messages:

        last = messages[-1]

        content = getattr(
            last,
            "content",
            str(last)
        )

        # ----------------------------------------------------
        # Invalid input
        # ----------------------------------------------------

        if content == "INVALID_INPUT":

            return "Invalid input"

        return content

    return "Invalid input"


# ============================================================
# 10. CREATE AGENT CHAIN
# ============================================================

formatted_agent_chain = (

    RunnableLambda(format_for_agent)

    | agent

    | RunnableLambda(extract_text_response)

).with_types(
    input_type=AgentInput,
    output_type=str
)


# ============================================================
# 11. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Indian Weather and Cinema Agent",
    description=(
        "AI agent restricted to Indian weather, "
        "Indian cinema and temperature conversion."
    ),
    version="1.0.0"
)


# ============================================================
# 12. LANGSERVE ROUTE
# ============================================================

add_routes(
    app,
    formatted_agent_chain,
    path="/agent",
    playground_type="default"
)


# ============================================================
# 13. RUN SERVER
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            8000
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
