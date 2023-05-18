
<h2 class="code-line" data-line-start=1 data-line-end=2 ><a id="Description_of_chatbot_architecture_1"></a>Description of chatbot architecture:</h2>
<p class="has-line-data" data-line-start="3" data-line-end="4">In this project, a prototype of a chatbot is proposed, and its final implementation structure is designed based on the study of literature on chatbot implementations used in the business process. The architecture of the AI chatbot is presented in Figure below and consists of several components. It includes:</p>
<ol>
<li class="has-line-data" data-line-start="4" data-line-end="5">Dialogue management module: This module is responsible for controlling the flow of dialogue between the chatbot and the user. It manages interactions, stores the context of the current dialogue, and makes decisions regarding generating responses.</li>
<li class="has-line-data" data-line-start="5" data-line-end="6">Database: The chatbot utilizes a database to store information about ongoing dialogues with users. This allows it to remember previous conversations and utilize previous contexts for better understanding and responding to questions.</li>
<li class="has-line-data" data-line-start="6" data-line-end="7">Natural Language Understanding (NLU) module: This module processes the text input provided by the user and understands their intentions and extracts information from it. It employs various natural language processing techniques such as tokenization, lemmatization, syntactic analysis, etc., to process the input text into data structures understood by the chatbot.</li>
<li class="has-line-data" data-line-start="7" data-line-end="8">Business logic and scripts for performing additional actions: This module contains the business logic of the chatbot, including rules and procedures needed for proper information processing and decision-making. It may also include scripts and functions that enable the chatbot to access external data sources such as databases, APIs, or other sources of information.</li>
<li class="has-line-data" data-line-start="8" data-line-end="10">Natural Language Generation (NLG) module: This module is responsible for generating textual responses of the chatbot. It employs various text generation techniques such as templates, language modeling, or neural networks to generate a response based on the analysis of the context and business logic.</li>
</ol>
<p class="has-line-data" data-line-start="10" data-line-end="11">All these components are interconnected and collaborate to form a coherent chatbot architecture. This enables the chatbot to understand and respond to user queries in a natural and contextually appropriate manner within the defined business context.</p>

![image](https://github.com/Tsyhankova/Master-s-thesis/assets/52218796/41e93fe3-805f-492a-a97c-d74ab230a408)

<p class="has-line-data" data-line-start="12" data-line-end="14">Let’s also consider the sequence of component interactions in the final implementation of the AI chatbot, as reflected in the Figure.<br>
When the system receives a user request in the form of a text message, it is passed to the dialogue management module (1), which loads the dialogue context from the database (2). This reflects the current state of correspondence with the user based on previous events in the dialogue window. The dialogue management module forwards the message and context to the NLU module (3), where the input text is converted. The transformed input data is then passed to the neural network for classifying the user’s intent.</p>
<p class="has-line-data" data-line-start="15" data-line-end="16">The NLU module is also responsible for extracting relevant query parameters present in the user’s utterance, which are necessary for further action execution (4). Next, the dialogue management module determines the next most appropriate dialogue state (5), and the business logic (6) is executed, for example, if it is necessary to redirect the user’s inquiry to a live operator.</p>
<p class="has-line-data" data-line-start="17" data-line-end="18">The NLG module (7) generates the AI chatbot’s response for the user (8). In the simplest case, pre-defined templates labeled for each query and macro substitutions can be used to generate the response. In more complex scenarios, a neural network mechanism can be employed to generate the user’s response. These approaches can be combined since, for some questions, the response will always be generic, while for others, it may take a more flexible form.</p>

<p class="has-line-data" data-line-start="1" data-line-end="2">Let’s consider the structure of the implemented prototype of the AI chatbot, as shown in Figure below. The main component of the structure is the NLP module, which consists of the NLU and NLG modules. The NLU module converts the input data into a format suitable for feeding into the neural network and determines the user’s intent. The dialogue management module provides interaction with the user on the messenger platform (e.g., Telegram) while accessing the NLP module by passing the untranslated text of the user’s message as input and attempting to understand their intent. The NLG module focuses on generating text in English based on a given dataset.</p>

![image](https://github.com/Tsyhankova/Master-s-thesis/assets/52218796/942f6f8d-ba48-4d36-b56c-c029a0d64620)

<p class="has-line-data" data-line-start="0" data-line-end="1">Let’s take a closer look at the process happening in the NLP module. As discussed in the paper, we will use a Seq2Seq model to implement the chatbot prototype. The Seq2Seq model consists of two main units: an encoder and a decoder.</p>
<p class="has-line-data" data-line-start="2" data-line-end="3">There are different Sequence-to-Sequence models available. In this work, we will use the standard RNN-based model with an added attention layer. For comparison, we will consider the Transformer model, which is also a Sequence-to-Sequence model but does not utilize recurrent neural networks.</p>
<p class="has-line-data" data-line-start="4" data-line-end="5">Now, let’s examine the architecture of both models used in this work.</p>
<p class="has-line-data" data-line-start="6" data-line-end="7">Figure bellow illustrates the standard Seq2Seq architecture with RNN. The original sentence, such as “How are you?”, is fed into the recurrent neural network cells of the encoder. The encoder processes the sentence and produces an encoded sequence, denoted as ‘z’, at its output. The decoder, in addition to the information from the encoder’s output, receives the reference response that it learns from, for example, “I am fine”. During the training process, the decoder adjusts its weights in such a way that, given the original question as input, it aims to produce the reference phrase as accurately as possible in the output. During training, the phrase is enclosed with tags. In this case, “&lt;BOS&gt;” represents the beginning-of-sequence tag, and “&lt;EOS&gt;” represents the end-of-sequence tag.</p>

![image](https://github.com/Tsyhankova/Master-s-thesis/assets/52218796/8cbf3596-518a-40b5-b29d-5564eaece2de)

[source](https://docs.chainer.org/en/stable/examples/seq2seq.html)

<p class="has-line-data" data-line-start="1" data-line-end="6">The first model was built with an Encoder-Decoder architecture with attention (Bahdanau’s attention model).<br>
•   The encoder consists of an embedding layer followed by a BiLSTM layer.<br>
•   The decoder also includes an embedding layer, followed by LSTM and Dense layers.<br>
•   The attention layer takes the encoder’s output and the decoder’s output as input.<br>
The Encoder-Decoder model with attention focuses on specific parts of the input text. In the case of Seq2Seq neural networks, using attention allows the creation of a context vector at any given time, considering the current hidden state of the decoder and a subset of the hidden states from the encoder. The model’s architecture is illustrated in Figure bellow.</p>

![image](https://github.com/Tsyhankova/Master-s-thesis/assets/52218796/add7b33f-0e10-4b16-84dd-d3b612eeee00)

<p class="has-line-data" data-line-start="1" data-line-end="2">The Transformer used in this study is directly built from the Transformer described by François Chollet in “English-to-Spanish translation with a sequence-to-sequence Transformer,” with modified output for the chatbot.</p>
<p class="has-line-data" data-line-start="3" data-line-end="4">Let’s delve into the processes occurring in the encoder and decoder of the Transformer model, which is implemented in this study.</p>
<p class="has-line-data" data-line-start="5" data-line-end="6">The encoder consists of 6 identical layers, each composed of two sub-layers. The first sub-layer implements the multi-head self-attention mechanism, which receives linear projections of queries, keys, and values. Each projected version generates outputs in parallel to produce the final result of this sub-layer. The second sub-layer is defined by a fully connected feed-forward network. The network includes two linear transformation layers with ReLU activation between them. Residual connections are applied around each sub-layer and a normalization layer. Due to the structure of the Transformer, we cannot obtain information about the relative positions of words in the sequence. To address this, positional encoding is computed using sine and cosine functions with different frequencies. This positional encoding is simply added to the input embeddings. Figure bellow illustrates the encoder model.</p>

![image](https://github.com/Tsyhankova/Master-s-thesis/assets/52218796/3194451c-68bb-43e1-8a17-4ba35dc94179)

<p class="has-line-data" data-line-start="1" data-line-end="2">The decoder is very similar to the encoder. It also consists of 6 identical layers. However, this time each layer comprises 3 sub-layers instead of 2. The first sub-layer receives the decoder’s output from the previous time step, enriched with positional encoding. Once again, we employ the multi-head self-attention mechanism. However, this time we focus only on the preceding words rather than all words in the sequence simultaneously. This ensures that the prediction at the current time step depends only on previous words and is not based on future incorrect connections. We achieve this effect by implementing a mask that prevents relevant values in the matrix. This defines the decoder as unidirectional.</p>
<p class="has-line-data" data-line-start="3" data-line-end="4">The second sub-layer implements a multi-head self-attention layer similar to the encoder. This layer receives queries from the previous sub-layer and keys and values from the encoder’s output. The final layer is a fully connected feed-forward network, which is essentially the same as the one implemented in the encoder. Additionally, we incorporate residual connections and apply a normalization layer after each sub-layer. Figure bellow illustrates the decoder model.</p>

![image](https://github.com/Tsyhankova/Master-s-thesis/assets/52218796/c8f89d27-79f8-48af-a2f4-bbe0de7eb9cb)

<p class="has-line-data" data-line-start="1" data-line-end="10">The described model will operate as follows:<br>
● Firstly, we create an embedding vector for each word in the sequence.<br>
● Next, we augment the embedding with positional encoding.<br>
● Then, the input is fed into the encoder, and we pass through two sub-layers as described earlier.<br>
● Now, the decoder receives its previous output, also enriched with positional encoding.<br>
● This input is passed through three sub-layers.<br>
● We apply a mask in the first sub-layer to prevent incorrect connections.<br>
● In the second sub-layer, we obtain the encoder’s output alongside the output from the first sub-layer of the decoder. We pass it through a fully connected neural network to generate the final decoder output.<br>
● Finally, we pass the decoder’s output through another fully connected neural network and apply the softmax function to generate a prediction for the next word in the sequence</p>


<p class="has-line-data" data-line-start="0" data-line-end="1">The scope of the project and what the project needs to accomplish in order to be considered successful is as follows:</p>
<ol>
<li class="has-line-data" data-line-start="1" data-line-end="7">Seq2Seq (RNN with attention): The project implemented a version of the Sequence to Sequence model with attention capable of generating dialogues.<br>
To achieve this, the project:<br>
•   Implemented the Seq2Seq model architecture with an encoder-decoder structure.<br>
•   Incorporated attention mechanism to focus on relevant parts of the input sequence.<br>
•   Trained the model using appropriate dialogue datasets.<br>
•   Evaluated the model’s performance using automatic evaluation metrics and human evaluation.</li>
<li class="has-line-data" data-line-start="7" data-line-end="13">Transformer: The project implemented the Transformer model, which was capable of generating dialogues.<br>
To accomplish this, the project:<br>
•   Implemented the Transformer model architecture, including multi-head self-attention and feed-forward layers.<br>
•   Incorporated positional encoding to capture word positions in the sequence.<br>
•   Trained the Transformer model using dialogue datasets.<br>
•   Evaluated the model’s performance using automatic evaluation metrics and human evaluation.</li>
<li class="has-line-data" data-line-start="13" data-line-end="20">Model Comparison: The above-mentioned models were trained and compared based on automatic and human evaluation metrics.<br>
To perform this comparison, the project:<br>
•   Trained both the Seq2Seq model and the Transformer model on the same dialogue datasets.<br>
•   Evaluated the models using established automatic evaluation metrics.<br>
•   Conducted human evaluation, which involved having human evaluators assess the generated dialogues for fluency, coherence, and relevance.<br>
•   Compared the performance of the models based on the evaluation results.<br>
The successful completion of these tasks within the project enabled a comprehensive comparison of the Seq2Seq model and the Transformer model in generating dialogues.</li>
</ol>
