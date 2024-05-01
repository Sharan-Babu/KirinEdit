import streamlit as st
#from streamlit_ace import st_ace
from streamlit_monaco import st_monaco
import anthropic
import json
import ast
#from code_editor import code_editor

st.set_page_config(page_title="KirinEdit",page_icon="⚡️")
st.title("KirinEdit⚡️✏️")
st.caption("An experiment")
st.write("[YouTube Demo](https://www.youtube.com/watch?v=szU2lKXkYEI)")

explanation_text = """
**KirinEdit** is a Large Language Model (LLM) that can **edit** given 
'text input' by *identifying and replacing text sequences in it*, 
without rewriting the entire text. 

How? Input text is processed and sent to the model. Model
has been taught to suggest changes to the processed text. These changes
are applied and the final text is returned back to the user.

This opens up a lot of use-cases which require quick response 
(feedback) times like Content Writing (*Long Document Editing*), Coding, creating User Interfaces,
animations and powerpoints, etc. thanks to the decreased latency when 
interacting with the LLM. 

Costs also decrease significantly as you would pay 
only for `"tokens for identification of text sequences to be replaced"` + 
`"new (replacement) content"` instead of rewriting the whole text with the required 
changes.

Though KirinEdit by itself is a capable LLM, performant systems can be built
by making it available as a tool for more powerful/larger LLMs. Ex: Pairing 
it up with GPT-4 or Claude Opus. This way, you get the `"new content"`
from the larger model (better reasoning, larger context window and other features) 
while the rewriting itself is being done by the cheap KirinEdit Model.

This demo uses Claude-3-Haiku; input limits capped. Using GPT-4
or Claude Sonnet leads to even higher accuracy thanks to those
models being able to better articulate changes to make and form corresponding
queries. If that implementation, or API access of interest to you,
please DM!


[Follow me on Twitter](https://twitter.com/sharanbabu2001) for updates and upcoming exciting work in the LLM space!
"""

with st.expander("What is it ⚙️"):
	st.write(explanation_text)

quick_templates1 = """
Test KirinEdit immediately by using these copyable templates.

**1. Editing a Research Doc**

*Content*
"""

ldoc = r"""
\documentclass{article}

\usepackage[utf8]{inputenc}
\usepackage{amsmath}
\usepackage{amssymb}

\title{Title of Your Research Paper}
\author{Your Name}
\date{\today}

\begin{document}

\maketitle

\begin{abstract}
This is where you write a brief abstract summarizing the main points of your research paper.
\end{abstract}

\section{Introduction}
Start with an introduction section that provides background information, motivation, and the main objectives of your research.

\section{Literature Review}
Include a literature review section that discusses relevant existing work and how your research fits into the broader context.

\section{Methodology}
Describe the methodology you used for your research, including any experimental setup, data collection, or analytical techniques.

\section{Results}
Present the key results and findings from your research in this section, using figures, tables, and equations as needed.

\begin{equation}
    E = mc^2
\end{equation}

\section{Discussion}
Discuss the implications of your results, their significance, and how they relate to existing work.

\section{Conclusion}
Summarize the main conclusions of your research and suggest potential future work.

\bibliographystyle{plain}
\bibliography{references.bib}

\end{document}
"""

quick_templates2 = """
**2. Integrating a library (wandb) with Existing Neural Networks code**

*Content*
"""

cdoc = r"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten

# Define the input shape
input_shape = (64, 64, 3)  # Example: 64x64 RGB images

# Create the model
model = Sequential()

# Add convolutional layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Flatten the output for fully connected layers
model.add(Flatten())

# Add fully connected layers
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))  # Example: 10 output classes

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val))

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test loss: {loss:.4f}')
print(f'Test accuracy: {accuracy:.4f}')
"""

cdoc2 = r"""
Quickstart
Install W&B and start tracking your machine learning experiments in minutes.

1. Create an account and install W&B
Before you get started, make sure you create an account and install W&B:

Sign up for a free account at https://wandb.ai/site and then log in to your wandb account.
Install the wandb library on your machine in a Python 3 environment using pip.
The following code snippets demonstrate how to install and log into W&B using the W&B CLI and Python Library:

Notebook
Command Line
Install the CLI and Python library for interacting with the Weights and Biases API:

!pip install wandb

2. Log in to W&B
Notebook
Command Line
Next, import the W&B Python SDK and log in:

wandb.login()

Provide your API key when prompted.

3. Start a run and track hyperparameters
Initialize a W&B Run object in your Python script or notebook with wandb.init() and pass a dictionary to the config parameter with key-value pairs of hyperparameter names and values:

run = wandb.init(
    # Set the project where this run will be logged
    project="my-awesome-project",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": 0.01,
        "epochs": 10,
    },
)

A run is the basic building block of W&B. You will use them often to track metrics, create logs, create jobs, and more.

Putting it all together
Putting it all together, your training script might look similar to the following code example. The highlighted code shows W&B-specific code. Note that we added code that mimics machine learning training.

# train.py
import wandb
import random  # for demo script

wandb.login()

epochs = 10
lr = 0.01

run = wandb.init(
    # Set the project where this run will be logged
    project="my-awesome-project",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": lr,
        "epochs": epochs,
    },
)

offset = random.random() / 5
print(f"lr: {lr}")

# simulating a training run
for epoch in range(2, epochs):
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset
    print(f"epoch={epoch}, accuracy={acc}, loss={loss}")
    wandb.log({"accuracy": acc, "loss": loss})

# run.log_code()

That's it! Navigate to the W&B App at https://wandb.ai/home to view how the metrics we logged with W&B (accuracy and loss) improved during each training step.

Shows the loss and accuracy that was tracked from each time we ran the script above. 

The image above (click to expand) shows the loss and accuracy that was tracked from each time we ran the script above. Each run object that was created is show within the Runs column. Each run name is randomly generated.

What's next?
Explore the rest of the W&B ecosystem.

Check out W&B Integrations to learn how to integrate W&B with your ML framework such as PyTorch, ML library such as Hugging Face, or ML service such as SageMaker.
Organize runs, embed and automate visualizations, describe your findings, and share updates with collaborators with W&B Reports.
Create W&B Artifacts to track datasets, models, dependencies, and results through each step of your machine learning pipeline.
Automate hyperparameter search and explore the space of possible models with W&B Sweeps.
Understand your datasets, visualize model predictions, and share insights in a central dashboard.


Common Questions
Where do I find my API key? Once you've signed in to www.wandb.ai, the API key will be on the Authorize page.

How do I use W&B in an automated environment? If you are training models in an automated environment where it's inconvenient to run shell commands, such as Google's CloudML, you should look at our guide to configuration with Environment Variables.

Do you offer local, on-prem installs? Yes, you can privately host W&B locally on your own machines or in a private cloud, try this quick tutorial notebook to see how.

How do I turn off wandb logging temporarily? If are testing code and want to disable wandb syncing, set the environment variable WANDB_MODE=offline.
"""

quick_templates3 = """
**3. Editing Text Documents**

*Content*
"""

qdoc = """
Artificial Intelligence (AI) has made remarkable strides in recent years,\nwith applications ranging from virtual assistants to self-driving cars.\nHowever, what we have witnessed so far is just the tip of the iceberg.\nThe future of AI holds immense promise, with the potential to revolutionize\nvirtually every aspect of our lives. As AI systems become more sophisticate\nand capable, they will play an increasingly pivotal role in fields such as\nhealthcare, education, scientific research, and environmental conservation.

One of the most exciting prospects of AI is its ability to augment\nhuman intelligence and capabilities. By processing vast amounts of data\nand identifying patterns that would be imperceptible to humans, AI can\nassist us in making more informed decisions, generating innovative solutions\nand tackling complex challenges that have long eluded us.\nFurthermore, AI-powered automation could alleviate the burden of tedious\nand repetitive tasks, freeing up human resources to focus on more creative\nand intellectually stimulating endeavors.

Despite the numerous benefits, the development of AI also raises ethical\nand societal concerns. As AI systems become more autonomous and capable of\nmaking decisions that affect human lives, it is crucial to ensure that\nthey are designed with robust safeguards and adhere to principles of\ntransparency, accountability, and respect for human rights. Additionally\nthe potential impact of AI on employment and the workforce necessitates\nproactive measures to mitigate potential disruptions and ensure a smooth\ntransition. Navigating these challenges will require a collaborative effort\nfrom policymakers, researchers, ethicists, and the broader public to shape\nthe responsible development and deployment of AI technologies.
"""

with st.expander("Quick Templates"):
	st.write(quick_templates1)
	st.code(ldoc, line_numbers=True, language="textile")
	st.write("*Context*")
	st.code("Help with this research doc", language="textile")
	st.write("*Command*")
	st.code("add a results table", language="textile")
	st.write("\n---")

	st.write(quick_templates2)
	st.code(cdoc, line_numbers=True)
	st.write("*Context*")
	st.code(cdoc2, language="textile")
	st.write("*Command*")
	st.code("Integrate Wandb with current Neural Networks Code", language="textile")
	st.write("\n---")

	st.write(quick_templates3)
	st.code(qdoc, line_numbers=True, language="textile")
	st.write("*Context*")
	st.code("You are a helpful assistant", language="textile")
	st.write("*Command*")
	st.code("Let us modify the opening and also remove the very last line", language="textile")
	st.write("\n---")


if "editor_val" not in st.session_state:
	ev = "---\nMy Text\n---"
	st.session_state["editor_val"] = ev

else:
	ev = st.session_state["editor_val"]	

def anthropic_call(numbered_content_string, context, command):
	
	client = anthropic.Anthropic(
				    # defaults to os.environ.get("ANTHROPIC_API_KEY")
				    api_key=st.secrets["api_key"]
				)


	message = client.messages.create(
			    model="claude-3-haiku-20240307",
			    #model="claude-3-sonnet-20240229",
			    max_tokens=2000,
			    temperature=0.1,
			    system=st.secrets["s1"],
			    messages=[
			        {
			            "role": "user",
			            "content": [
			                {
			                    "type": "text",
			                    "text": st.secrets["s2"]
			                }
			            ]
			        },
			        {
			            "role": "assistant",
			            "content": [
			                {
			                    "type": "text",
			                    "text": st.secrets["s3"]
			                }
			            ]
			        },
			        {
			            "role": "user",
			            "content": [
			                {
			                    "type": "text",
			                    "text": st.secrets["s4"]
			                }
			            ]
			        },
			        {
			            "role": "assistant",
			            "content": [
			                {
			                    "type": "text",
			                    "text": st.secrets["s5"]
			                }
			            ]
			        },
			        {
			            "role": "user",
			            "content": [
			                {
			                    "type": "text",
			                    "text": st.secrets["s6"]
			                }
			            ]
			        },
			        {
			            "role": "assistant",
			            "content": [
			                {
			                    "type": "text",
			                    "text": st.secrets["s7"]
			                }
			            ]
			        },
			        {
			            "role": "user",
			            "content": [
			                {
			                    "type": "text",
			                    "text": st.secrets["s8"]
			                }
			            ]
			        },
			        {
			            "role": "assistant",
			            "content": [
			                {
			                    "type": "text",
			                    "text": st.secrets["s9"]
			                }
			            ]
			        },
			        {
			            "role": "user",
			            "content": [
			                {
			                    "type": "text",
			                    "text": st.secrets["s10"]
			                }
			            ]
			        },
			        {
			            "role": "assistant",
			            "content": [
			                {
			                    "type": "text",
			                    "text": st.secrets["s11"]
			                }
			            ]
			        },
			        {
			            "role": "user",
			            "content": [
			                {
			                    "type": "text",
			                    "text": st.secrets["s12"]
			                }
			            ]
			        },
			        {
			            "role": "assistant",
			            "content": [
			                {
			                    "type": "text",
			                    "text": st.secrets["s13"]
			                }
			            ]
			        },
			        {
			            "role": "user",
			            "content": [
			                {
			                    "type": "text",
			                    "text": st.secrets["s14"]
			                }
			            ]
			        },
			        {
			            "role": "assistant",
			            "content": [
			                {
			                    "type": "text",
			                    "text": st.secrets["s15"]
			                }
			            ]
			        },
			        {
			            "role": "user",
			            "content": [
			                {
			                    "type": "text",
			                    "text": st.secrets["s16"]
			                }
			            ]
			        },
			        {
			            "role": "assistant",
			            "content": [
			                {
			                    "type": "text",
			                    "text": st.secrets["s17"]
			                }
			            ]
			        },
			        {
			            "role": "user",
			            "content": [
			                {
			                    "type": "text",
			                    "text": f"Do not speak about previous examples as new user session begins.\n\nExample 3:\n----------\n<Helpful Context>\n{context}\n</Helpful Context>\n\n<Content>\n{numbered_content_string}\n</Content>\n\nRemember that following:\n1. For deleting or removing, simply set new_content_w/o_line_nums to empty character \"\" after selecting appropriate lines.\n2. When adding/replacing content, new_content should be able to transition smoothly 'without overwriting'.\n3. In new_content_w/o_line_nums, you have to use \\n for new line character. Ex: \"new_content_w/o_line_nums\" : \"First item \\n Second item \\n Third\"\n\n<User Query>\n{command}\n</User Query>"
			                }
			            ]
			        }
			    ]
			)

	
	message_content = message.content[0].text
	#print(f"\n\nNLOG:\n{message_content}")
	return message_content


# my replace function
def replace_lines(initial_content_list, tuples, new_content):
	new_list = initial_content_list[:]
	i = 0
	for pair in tuples:
		val1 = pair[0]
		val2 = pair[1]
		val1 = min(val1, len(new_list) - 1)
		val2 = min(val2, len(new_list) - 1)
		new_list[val1] = new_content[i]
		for x in range(val1 + 1, val2 + 1):
			new_list[x] = "%^%KIRIN-DELETE"

		i += 1

	return_list = []

	for element in new_list:
		if element != "%^%KIRIN-DELETE":
			return_list.append(element)

	return return_list	

# prompt injection check
def pic(json_obj):
	if "new_content_w/o_line_nums" in json_obj["response"]:
		return True

	for x in json_obj["replacements"]:
		if "new_content_w/o_line_nums" in x.get("new_content_w/o_line_nums"):
			return True

		if "|" not in x.get("start_line_full"):
			return True

		if "|" not in x.get("end_line_full"):
			return True		

	return False		

def refresh():
	if "temp_val" in st.session_state:
		st.session_state["editor_val"] = st.session_state["temp_val"]
		
with st.form("edit_form"):
	st.subheader("Content")
	st.caption("Max characters: 50000")
	#content = st_ace(height=400, value="---\nMy Text\n---", auto_update=True)
	content = st_monaco(value=ev, height="400px", language="text", theme="vs-dark")
	context = st.text_area("Context", placeholder="(Optional)", value="You are a helpful assistant",height=100, max_chars=15000, help="Additional Context or Reference")
	command = st.text_input("Command", placeholder="Desc. change in content", max_chars=5000, help="Specific Instruction")

	button = st.form_submit_button("Make Changes ✏️")
	if button:
		# process
		content_list = content.splitlines()
		numbered_content_list = [f"{i+1}|{line}" for i, line in enumerate(content_list)]
		numbered_content_string = "\n".join(numbered_content_list)[:50000]

		#print(str(numbered_content_string))

		llm_response = anthropic_call(numbered_content_string, context, command)

		# convert llm response to json
		try:
			#json_response = json.loads(llm_response, strict = False)
			json_response = ast.literal_eval(llm_response)
		except Exception as e:
			#e
			json_response = "error"

		#json_response
		prompt_injection_message = "I am a helpful AI assistant and my purpose is to help you create and edit content."

		if json_response == "error":
			with st.chat_message("ai"):
					st.write(prompt_injection_message)

		# json load successful
		else:
			if pic(json_response) == True:
				#st.write(prompt_injection_message)
				with st.chat_message("ai"):
					st.write(prompt_injection_message)

			else:	
				st.info("Changing content..")
				replacement_tuples = []
				replacement_content_list = []
				assistant_response = json_response["response"]

				for replacement in json_response["replacements"]:
					# start line number
					start_line = ""
					for char in replacement["start_line_full"]:
						if char != "|":
							start_line += str(char)
						else:
							break	

					start_line = int(start_line)
					start_index = start_line - 1

					# end line number
					end_line = ""
					for char in replacement["end_line_full"]:
						if char != "|":
							end_line += str(char)
						else:
							break	

					end_line = int(end_line)
					end_index = end_line - 1

					replacement_tuples.append((start_index, end_index))
					replacement_content_list.append(replacement["new_content_w/o_line_nums"])

				# make replacements
				final_edited_content = "\n".join(replace_lines(content_list, replacement_tuples, replacement_content_list)).replace("\\n","\n")
				with st.chat_message("ai"):
					st.write(assistant_response)
				final_content_editor = st_monaco(value=final_edited_content, height="410px", language="text", theme="vs-dark")
				st.session_state["temp_val"] = final_edited_content
				st.success("Please click 'Approve Changes' button below and continue editing")

col1, col2, col3 = st.columns([1.5,1,1.5])
if col2.button("Approve Changes", on_click = refresh):
	pass
