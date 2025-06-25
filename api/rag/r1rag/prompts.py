
        
day_prompt="""

You will be given a JSON dataset containing multi-day event summaries. A small excerpt might look like:

{
  "generated_text": "…detailed summary text…",
  "date": 1,
  "start_time": 11090000,
  "end_time": 22100000
},
{
  "generated_text": "…detailed summary text…",
  "date": 2,
  "start_time": 10440000,
  "end_time": 23000000
},
…

You will also receive a list of keywords. Note that these keywords might not appear exactly as-is in the summaries. 
You may need to interpret whether certain entries are contextually relevant (e.g., use of synonyms or related topics).

Your Task:
1. Examine all generated_text entries in the dataset.

2. Determine if each day's generated_text is directly or contextually relevant to keywords.

3. Among all relevant entries, select only the single most relevant one based on:
   - How closely it matches the keywords
   - How significant and meaningful the content is
   - The clarity and specificity of the information

4. For the selected most relevant entry, return:

5. The day label in the format DAYX (where X is the date value).

6. The corresponding generated_text excerpt.

7. The corresponding generated_text excerpt should be short and concise.


Important Requirements:
1. Apply logical reasoning to see if a summary might be relevant, even if the exact keywords are missing.

2. Return ONLY THE SINGLE MOST RELEVANT day entry.

3. Base your answer solely on the dataset's content; do not add extra commentary or explanations.


Required Output Format (Strictly Follow):

Return a valid JSON array containing exactly one object with two keys:

1. "day_label": The string label in the format DAYX.

2. "relevant_content": The corresponding excerpt from generated_text.


Example:

[
  {
    "day_label": "DAY1",
    "relevant_content": "<Relevant content from generated_text>"
  }
]

No additional fields or formatting beyond what is shown above.
"""


hour_prompt="""
You will be given a JSON dataset that contains multiple time intervals within a single day. Each interval has the following format:
{
  "generated_text": "…content description for this interval…",
  "date": 1,
  "start_time": 11090000,
  "end_time": 12100000
},
{
  "generated_text": "…content description for this interval…",
  "date": 1,
  "start_time": 12100000,
  "end_time": 13100000
},
…

You will also receive a set of keywords (where synonyms or related concepts might apply).

Your Task:

1. Examine each record’s generated_text within the provided JSON dataset to determine whether it directly or semantically relates to the keywords.

2. If a record’s generated_text is relevant to any of the keywords, that portion is considered “relevant content.”

3. For each relevant record, extract:

    The date (return it exactly as in the dataset)

    The start_time (return it exactly as in the dataset)

    The generated_text (the excerpt relevant to the keywords)

4. Return Return the single most relevant record in the following format:

  {
    "date": same as in the source data,
    
    "start_time": same as in the source data,
    
    "relevant_content": the corresponding text from generated_text
  }
  
  
Required Output Format (Strictly Follow):

[
  {
    "date": 1,
    "start_time": 11090000,
    "relevant_content": "Excerpt of generated_text that relates to keywords"
  }
]

Only include the three fields: date, start_time, and relevant_content (with those exact names).

Do not add extra explanations, summaries, or fields.
"""


min_prompt="""
You will be given a JSON dataset that contains multiple 10-minute time intervals within a single day. An example entry might look like this:

{
  "generated_text": "Description of activities within this 10-minute interval...",
  "date": 1,
  "start_time": 11090000,
  "end_time": 11190000
},
{
  "generated_text": "Description of activities within this 10-minute interval...",
  "date": 1,
  "start_time": 12000000,
  "end_time": 12100000
}
…

You will also receive a list of keywords (where synonyms or related concepts might apply).


Your Task:
1. Review each entry in the JSON dataset, examining the generated_text field to see whether it directly or semantically relates to the provided keywords ({keywords}).

2. If the entry’s generated_text is relevant, treat that portion as “relevant content.”

3. For each relevant entry, extract:

    The date (return exactly as in the dataset)

    The start_time (return exactly as in the dataset)

    The generated_text (the portion that pertains to the keywords)

4. Return ONE SINGLE RECORD as a JSON array. The array must include these three keys:

    "date"

    "start_time"

    "relevant_content" (the portion of generated_text that relates to the keywords)


Required Output Format (Strictly Follow):

[
  {
    "date": 1,
    "start_time": 11090000,
    "relevant_content": "Excerpt from generated_text that is relevant to the keywords"
  }
]


Only include the three fields: date, start_time, and relevant_content (with those exact names).

Do not add any extra commentary, explanation, or fields.

"""

level1_prompt_for_videomme_long="""
You will be given a JSON dataset that contains multiple time intervals within a long video. Each interval has the 
{
  "generated_text": "…content description for this interval…",
  "date": 1,
  "start_time": 00100000,
  "end_time": 00200000
},
{
  "generated_text": "…content description for this interval…",
  "date": 1,
  "start_time": 00200000,
  "end_time": 00300000
},
…

You will also receive a set of keywords (where synonyms or related concepts might apply).

Your Task:

1. Examine each record’s generated_text within the provided JSON dataset to determine whether it directly or semantically relates to the keywords.

2. If a record’s generated_text is relevant to any of the keywords, that portion is considered “relevant content.”

3. For each relevant record, extract:

    The date (return it exactly as in the dataset)

    The start_time (return it exactly as in the dataset)

    The generated_text (the excerpt relevant to the keywords)

4. Return Return the single most relevant record in the following format:

  {
    "date": same as in the source data,
    
    "start_time": same as in the source data,
    
    "relevant_content": the corresponding text from generated_text
  }
  
  
Required Output Format (Strictly Follow):

[
  {
    "date": 1,
    "start_time": 00100000,
    "relevant_content": "Excerpt of generated_text that relates to keywords"
  }
]

Only include the three fields: date, start_time, and relevant_content (with those exact names).

Do not add extra explanations, summaries, or fields.

"""
level2_prompt_for_videomme_long="""
You will be given a JSON dataset that contains multiple 30 seconds descriptions within a video. Each interval has the following format:
{
  "generated_text": "…content description for this interval…",
  "date": 1,
  "start_time": 00000000,
  "end_time": 00003000
},
{
  "generated_text": "…content description for this interval…",
  "date": 1,
  "start_time": 00003000,
  "end_time": 00010000
},
…

You will also receive a set of keywords (where synonyms or related concepts might apply).

Your Task:

1. Examine each record’s generated_text within the provided JSON dataset to determine whether it directly or semantically relates to the keywords.

2. If a record’s generated_text is relevant to any of the keywords, that portion is considered “relevant content.”

3. For each relevant record, extract:

    The date (return it exactly as in the dataset)

    The start_time (return it exactly as in the dataset)

    The generated_text (the excerpt relevant to the keywords)

4. Return Return the single most relevant record in the following format:

  {
    "date": same as in the source data,
    
    "start_time": same as in the source data,
    
    "relevant_content": the excerpt relevant to the keywords
  }
  
  
Required Output Format (Strictly Follow):

[
  {
    "date": 1,
    "start_time": 00003000,
    "relevant_content": the excerpt relevant to the keywords
  }
]

Only include the three fields: date, start_time, and relevant_content (with those exact names).

Do not add extra explanations, summaries, or fields.

You must return one result.
"""



prompt_for_egoschema="""  
You are an AI assistant. Your primary goal is to identify the single most relevant segment from a JSON dataset based on a given set of keywords and return it in a specific format.

**Input Data Structure:**

You will be provided with:
1.  A JSON dataset, which is a list of records. Each record represents a 30-second interval description from a first-person video and has the following structure:
    ```json
    {
      "generated_text": [
        "<time_in_seconds>:#C <description_camera_wearer>",
        "<time_in_seconds>:#O <description_other_person>",
        // ... more descriptions
      ],
      "date": <integer_date_identifier>,
      "start_time": "<HHMMSSMS_string>", // Example: "00003000"
      "end_time": "<HHMMSSMS_string>"
    }
    ```
    * **Regarding `generated_text`**:
        * It is a list of strings. Each string starts with a timestamp in seconds, followed by `#`.
        * `C` denotes the camera wearer.
        * `O` denotes another person.
        * If `generated_text` is an empty list (`[]`), it signifies no descriptive text for that interval.
2.  A set of keywords. When matching, you should consider synonyms and semantically related concepts for these keywords.

**Your Task:**

1.  **Iterate and Evaluate**: Carefully examine each record within the provided JSON dataset. For each record, analyze its `generated_text` field.
2.  **Determine Relevance**: For each record, assess whether the content of its `generated_text` (i.e., the list of description strings) is directly or semantically related to any of the provided keywords. A record is considered relevant if **any part** of its `generated_text` relates to the keywords.
3.  **Identify the Single Most Relevant Record**: From all records, you must identify the **single most relevant record** whose `generated_text` has the strongest connection to the keywords. If multiple records seem relevant, choose the one you determine to be the absolute best match.
4.  **Extract Content from the Chosen Record**:
    * Once you have identified the single most relevant record, its **entire `generated_text` field** (which is a list of strings as it appears in the source data) will be the value for `relevant_content`.

**Required Output Format (Strictly Follow):**

You must return a JSON array containing **exactly one** JSON object. This object represents the single most relevant record you found.
The object **must** contain only the following three fields, using these exact names:

* `"date"`: The `date` value from the source record (return it exactly as in the dataset).
* `"start_time"`: The `start_time` value from the source record (return it exactly as in the dataset).
* `"relevant_content"`: The **entire `generated_text` list of strings** from the chosen most relevant record (return it exactly as in the dataset for that record).

**Example of the required output format:**

Let's assume the following record was identified as the single most relevant one, and keywords matched something within its `generated_text`:
Original record:
```json
 {
   "generated_text": [
     "49:#C C takes scissors",
     "51:#C C cuts thread with a scissor",
     "52:#C C puts down a scissor",
     "53:#C C takes cloth",
     "55:#C C drinks tea",
     "58:#C C puts down a cup of tea"
   ],
   "start_time": "00003000",
   "end_time": "00010000",
   "date": 1
 }
 
Output would be:

[
  {
    "date": 1,
    "start_time": "00003000",
    "relevant_content": [
      "49:#C C takes scissors",
      "51:#C C cuts thread with a scissor",
      "52:#C C puts down a scissor",
      "53:#C C takes cloth",
      "55:#C C drinks tea",
      "58:#C C puts down a cup of tea"
    ]
  }
]

Key Constraints:

You must return one result in the format specified.
Only include the three specified fields (date, start_time, relevant_content) in the output object, with those exact names.
Do not add any extra explanations, summaries, or any fields not explicitly requested.
The output must be a valid JSON array containing a single object.
"""