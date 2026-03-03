import json
import google.generativeai as genai
from agent_tools import DetectiveTools

# 1. Configure Gemini
# Ensure GOOGLE_API_KEY is defined in the environment or previous cell
if "GOOGLE_API_KEY" not in globals():
    # Fallback or placeholder if running standalone without Cell 1
    # You might want to use os.environ.get("GOOGLE_API_KEY") here
    pass

try:
    genai.configure(api_key=GOOGLE_API_KEY)
except NameError:
    print("Warning: GOOGLE_API_KEY is not defined. Please define it before running this script.")

# Use a model that supports JSON mode for reliability
# Generation config ensures the model returns valid JSON
generation_config = {
    "temperature": 0.7,  # Reduced for more focused reasoning
    "top_p": 0.9,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "application/json",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

# 2. Define the OPTIMIZED System Prompt with Investigation Patterns
SYSTEM_PROMPT = """
You are an elite detective AI Agent. Your goal is to solve crimes EFFICIENTLY in 2-4 steps maximum.

**Available Tools:**
- interview_witness(witness_name: str) - Get witness statement
- review_traffic_cctv(location: str, timeframe: str) - Check CCTV (format: 'HH:MM-HH:MM')
- check_vehicle_registration(vehicle_number: str) - Lookup vehicle owner
- analyze_fingerprints(sample_id: str) - Match fingerprints
- trace_mobile_number(mobile_number: str) - Find phone owner
- review_access_logs(facility_or_room: str, timeframe: str) - Check room access
- review_wifi_logs(area: str, timeframe: str) - Check WiFi connections
- check_upi_transactions(party_name: str, timeframe: str) - Check transactions
- interrogate_suspect(suspect_name: str) - Question suspect (use when you have strong evidence)

**INVESTIGATION PATTERNS - Follow These:**

Pattern A: Person Sighting → Direct Investigation
1. Initial clue mentions witness who saw someone → interview_witness(name)
2. Witness reveals suspect info → interrogate_suspect(suspect_name)

Pattern B: Vehicle/Object Clue → Owner Trace
1. Initial clue has vehicle → check_vehicle_registration(vehicle_number)
2. Get owner → interrogate_suspect(owner_name)

Pattern C: Location/Time Clue → Surveillance Check
1. Initial clue mentions location/time → review_traffic_cctv or review_access_logs
2. Identify suspect → interrogate_suspect(suspect_name)

Pattern D: Digital Evidence
1. Tech theft → review_wifi_logs or review_access_logs
2. If suspect identified → interrogate_suspect(name)

**FEW-SHOT EXAMPLES:**

Example 1 (2 steps):
Clue: "Guard saw person in red hoodie running to Parking B at 20:15"
Step 1: interview_witness("Guard") → "Saw student Rahul in red hoodie"
Step 2: interrogate_suspect("Rahul") → Confesses
Result: Culprit = Rahul

Example 2 (2 steps):
Clue: "White Maruti Swift MH-01-AB-1234 speeding away"
Step 1: check_vehicle_registration("MH-01-AB-1234") → Owner is "Vijay Kumar"
Step 2: interrogate_suspect("Vijay Kumar") → Confesses
Result: Culprit = Vijay Kumar

Example 3 (3 steps):
Clue: "Door swipe at Lab 203 at 22:12"
Step 1: review_access_logs("Lab 203", "22:00-22:30") → "Prof. Sharma accessed at 22:12"
Step 2: interrogate_suspect("Prof. Sharma") → Denies, mentions seeing student
Step 3: interview_witness("Prof. Sharma") → Points to "Amit"
Result: Culprit = Amit

**CRITICAL RULES:**
1. ANALYZE the initial clue to pick the RIGHT pattern immediately
2. Extract ALL names, vehicles, locations from observations
3. Move to interrogate_suspect as SOON as you have a strong lead
4. You have MAX 4 STEPS - be strategic
5. Each observation gives you multiple clues - use them ALL

**OUTPUT FORMAT (STRICT JSON):**

For action:
{
  "analysis": "Brief pattern identification: Using Pattern X because...",
  "thought": "Specific reasoning for this action",
  "action": "tool_name",
  "args": {"param": "value"}
}

For solution:
{
  "analysis": "Evidence chain summary",
  "thought": "Why this person is guilty",
  "culprit": "Full Name"
}

**CONSTRAINTS:**
- NO markdown code blocks (```json)
- Just raw JSON
- Use EXACT tool names and argument names
- Always use full names from observations
"""

# 3. Load Data
with open("reported_cases.json", "r") as f:
    data = json.load(f)

cases_to_solve = data["cases"]["easy"]
all_predictions = {}

print(f"Starting investigation for {len(cases_to_solve)} cases...")

# 4. Main Loop with REDUCED max steps
for case in cases_to_solve:
    case_id = case["case_id"]
    description = case["description"]
    initial_clue = case["initial_clue"]
    
    print(f"\n--- Solving Case: {case_id} ---")
    
    # Initialize tools for this specific case
    tools = DetectiveTools(case_id=case_id)
    
    # Initialize History
    history = [
        f"Case Description: {description}",
        f"Initial Clue: {initial_clue}"
    ]
    
    # Track steps for the final JSON export (action & args only)
    steps_taken_for_export = []
    
    max_steps = 4  # REDUCED from 6 to force efficiency
    solved = False
    
    for step_i in range(max_steps):
        # Construct the prompt
        context = "\n".join(history)
        full_user_prompt = f"{SYSTEM_PROMPT}\n\n[HISTORY]\n{context}\n\n[YOUR NEXT MOVE]"
        
        try:
            # Call Gemini
            response = model.generate_content(full_user_prompt)
            
            # Clean response if necessary
            response_text = response.text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            # Parse JSON
            action_data = json.loads(response_text)
            
            # Check if we are done
            if "culprit" in action_data:
                culprit = action_data["culprit"]
                analysis = action_data.get("analysis", "")
                explanation = action_data.get("thought", "")
                print(f"  [Solved] Culprit: {culprit}")
                print(f"    Analysis: {analysis}")
                print(f"    Reasoning: {explanation}")
                
                all_predictions[case_id] = {
                    "culprit": culprit,
                    "steps": steps_taken_for_export
                }
                solved = True
                break
                
            elif "action" in action_data:
                action_name = action_data["action"]
                args = action_data.get("args", {})
                analysis = action_data.get("analysis", "")
                thought = action_data.get("thought", "")
                
                print(f"  [Step {step_i+1}]")
                print(f"    Analysis: {analysis}")
                print(f"    Thought: {thought}")
                print(f"    Action: {action_name}({args})")
                
                # Execute Tool
                if hasattr(tools, action_name):
                    tool_func = getattr(tools, action_name)
                    try:
                        observation = tool_func(**args)
                    except Exception as e:
                        observation = f"Error executing tool: {e}"
                else:
                    observation = f"Error: Tool '{action_name}' not found."
                
                print(f"    Result: {observation}")
                    
                # Store the result for the agent's history (Context)
                history.append(f"Action: {action_name} | Args: {args}")
                history.append(f"Observation: {observation}")
                
                # Store the step for final export (REQUIRED FORMAT: action, args)
                steps_taken_for_export.append({
                    "action": action_name,
                    "args": args
                })
                
            else:
                print("  [Error] Invalid JSON format received.")
                history.append("System: Invalid JSON. Use correct format with 'action' or 'culprit'.")
                
        except Exception as e:
            print(f"  [Error] Unexpected error: {e}")
            history.append(f"System: Error: {e}")
            
    if not solved:
        print(f"  [Failed] Could not solve case {case_id} within {max_steps} steps.")
        all_predictions[case_id] = {
            "culprit": "Unknown",
            "steps": steps_taken_for_export
        }

print("\nAll cases processed.")
