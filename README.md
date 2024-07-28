# Research-Tool
In this project, we successfully developed a web-based application that integrates several advanced natural language processing (NLP) techniques to provide users with detailed answers and paraphrased content from web sources. By leveraging the power of pre-trained models from the Hugging Face Transformers library, such as "tuner007/pegasus_paraphrase" for paraphrasing and "deepset/roberta-base-squad2" for question answering, we created a robust and responsive system.

User Authentication and History Management:
1.Implemented user registration and login functionalities, ensuring secure access to the application.
2.Designed a system to store and manage user history, allowing each user to have a personalized experience and maintain a record of their interactions limited to the most recent three entries.

Web Scraping and Content Extraction:
1.Developed a reliable web scraping mechanism using BeautifulSoup to extract relevant content from specified URLs.
2.Implemented a fallback mechanism to perform Google search queries when direct URLs are not provided, enhancing the versatility of the application.

Question Answering and Paraphrasing:
1.Integrated a question-answering pipeline capable of extracting and answering queries from web content, providing users with precise and relevant information.
2.Implemented a paraphrasing module to rephrase extracted content, offering users multiple perspectives on the same information, thereby enhancing understanding and retention.

Responsive Web Interface:
1.Designed a user-friendly interface using Flask, enabling seamless interactions with the backend functionalities.
Included responsive elements like voice recognition and output features to cater to a diverse user base, including those who prefer audio interactions.
