from google import genai
import os
from dotenv import load_dotenv
from google.genai import types
from collections import defaultdict
import json
import pandas as pd 

load_dotenv()
nb_generation = 200

client = genai.Client(api_key=os.getenv("API_KEY"))

x_data = []
y_data = []
prompt_data = []

for (x,y) in [(0,9), (9,0)]:

    prompt = f"""- the G position is (row={x},column={y})

    Your task is to generate a clear, concise textual instruction that guides an agent to a specific target location on a grid. 
    The grid is 10x10, with both columns and rows numbered starting from 0.

    Requirements for the Instruction:
        - Describe the Target's Position: Explain where the target is located using relative terms or columns/rows number.
        - Avoid Direct References: Do not refer to the starting position (A) or explicitly mention the symbol for the target (G).
        - Use Grid Coordinates Implicitly: Since rows and columns are numbered from 0, your description should naturally guide the agent using directional cues without directly stating coordinates.

    Example for Inspiration: For a grid where the target is in the bottom right corner, one might say:
    “Go to the bottom right corner.” or “Move to the rightmost case in the bottom row” for (9,9) G position
    """
    for _ in range(nb_generation):
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction="reverify yourself to be sure about the answer, and give me just the concise textual instruction",
                top_k=1000,
                temperature=2.0
            )
        )
        x_data.append(x)
        y_data.append(y)
        prompt_data.append(response.text)


# prompt = f"""- the G position is (row=9,column=9)

#     Your task is to generate a clear, concise textual instruction that guides an agent to a specific target location on a grid. 
#     The grid is 10x10, with both columns and rows numbered starting from 0.

#     Requirements for the Instruction:
#         - Describe the Target's Position: Explain where the target is located using relative terms or columns/rows number.
#         - Avoid Direct References: Do not refer to the starting position (A) or explicitly mention the symbol for the target (G).
#         - Use Grid Coordinates Implicitly: Since rows and columns are numbered from 0, your description should naturally guide the agent using directional cues without directly stating coordinates.

#     Example for Inspiration: For a grid where the target is in the top right corner, one might say:
#     “Go to the top right corner.” or “Move to the rightmost case in the top row” for (0,9) G position
#     """
# for _ in range(nb_generation):
#     response = client.models.generate_content(
#         model="gemini-2.0-flash",
#         contents=prompt,
#         config=types.GenerateContentConfig(
#             system_instruction="reverify yourself to be sure about the answer, and give me just the concise textual instruction, be innovative and use synonyms",
#             top_k=1000,
#             temperature=2.0
#         )
#     )
#     x_data.append(9)
#     y_data.append(9)
#     prompt_data.append(response.text)

dataset = pd.DataFrame({
    "row": x_data,
    "column": y_data,
    "prompt":prompt_data
})

dataset_2 = pd.read_pickle("./dataset/data.pickle")

pd.concat([dataset, dataset_2], ignore_index=True).to_pickle("./dataset/data.pickle")
