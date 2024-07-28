from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import os
import nltk
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
from transformers import pipeline , AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer=AutoTokenizer.from_pretrained("tuner007/pegasus_paraphrase")
model=AutoModelForSeq2SeqLM.from_pretrained("tuner007/pegasus_paraphrase",ignore_mismatched_sizes=True)
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a strong secret key
# Path to the CSV file
USER_FILE = 'users.csv'

# Load the QA pipeline with the specified model
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
nlp = pipeline("text2text-generation", model=model,tokenizer=tokenizer,truncations=True)

# Load users from CSV
def load_users():
    if os.path.exists(USER_FILE):
        return pd.read_csv(USER_FILE)
    else:
        return pd.DataFrame(columns=['Username', 'Password', 'History'])

# Save users to CSV
def save_users(users):
    users.to_csv(USER_FILE, index=False)

# Perform web search
def search_web(query):
    try:
        query = quote_plus(query)
        url = f"https://www.google.com/search?q={query}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        results=dict({'title':[],'link':[],'description':[]})
        c=1
        for g in soup.find_all('div', class_='tF2Cxc'):
            title_tag = g.find('h3')
            link_tag = g.find('a')
            description_tag = g.find('div', class_='VwiC3b')
            if title_tag and link_tag and description_tag:
                if c==7:
                    break
                results['title'].append(title_tag.text)
                results['link'].append(link_tag['href'])
                results['description'].append(description_tag.text)
            c+=1
        return results
    except requests.exceptions.RequestException as e:
        print(f"Error fetching search results: {e}")
        return dict({'title':[],'link':[],'description':[]})
# Extract text from URL
def extract_text_from_url(url, lines=5):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = [p.get_text() for p in soup.find_all('p')]
        text = ' '.join(paragraphs)#<p>
        content_lines = text.split('. ')[:lines]
        return '. '.join(content_lines) + '.'
    except Exception as e:#<p>
        print(f"Error extracting text from {url}: {str(e)}")
        return ""


def paraphrase_text(text):
    sentences = nltk.sent_tokenize(text)
    paraphrased_sentences = []
    batch_size = 5  # Adjust batch size based on your GPU memory and model capacity
    
    try:
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=100)
            outputs = model.generate(**inputs)
            paraphrased_batch = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            paraphrased_sentences.extend(paraphrased_batch)
        
        return ' '.join(paraphrased_sentences)
    except Exception as e:
        print(f"An error occurred during paraphrasing: {e}")
        return text

# Answer questions from URLs
def answer_questions_from_urls(urls, question, lines=5):
    results = []
    for url in urls:
        text = extract_text_from_url(url, lines)
        if text:
            try:
                result = qa_pipeline(question=question, context=text)
                answer = result.get('answer', 'No answer found')
                content_lines = text.split('. ')
                relevant_lines = content_lines[:7]
                relevant_text = '. '.join(relevant_lines) + '.'
                paraphrased_relevant_text = paraphrase_text(relevant_text)
                results.append({
                    'url': url,
                    'question': question,
                    'answer': answer,
                    'content_excerpt': relevant_text,
                    'paraphrased_content_excerpt': paraphrased_relevant_text
                })
            except Exception as e:
                print(f"Error answering question '{question}' for URL {url}: {str(e)}")
    return results

# Save user history
def save_history(username, history):
    users = load_users()
    if username in users['Username'].values:
        index = users[users['Username'] == username].index[0]
        users.at[index, 'History'] = str(history)
    save_users(users)

# Limit history to the most recent 3 entries
def limit_history(history):
    return history[-3:] if len(history) > 3 else history

@app.route('/')
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['Username']
        password = request.form['Password']
        users = load_users()

        if username not in users['Username'].values:
            return render_template('login.html', message='Username does not exist')
        
        user_data = users[users['Username'] == username].iloc[0]
        if str(user_data['Password']) != password:
            return render_template('login.html', message='Incorrect password')
        
        session['username'] = username

        if 'history' not in session or not isinstance(session['history'], list):
            session['history'] = []

        # Load user history into session
        user_history = user_data['History']
        if isinstance(user_history, str):
            session['history'] = eval(user_history)
        else:
            session['history'] = []

        return redirect(url_for('home'))
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['Username']
        password = request.form['Password']
        users = load_users()

        if username in users['Username'].values:
            return render_template('register.html', message='Username already exists')
        
        new_user = pd.DataFrame([[username, password]], columns=['Username', 'Password'])
        users = pd.concat([users, new_user], ignore_index=True)
        save_users(users)

        return render_template('register.html', message=f'Successfully registered as {username}')
    
    return render_template('register.html')

@app.route('/home', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        question = request.form['Question'].strip()
        urls_input = request.form['Urls'].strip()
        urls = [url.strip() for url in urls_input.split(',') if url.strip().startswith('http')]
        
        if urls:
            answers = answer_questions_from_urls(urls, question)
            username = session['username']
            
            if 'history' not in session or not isinstance(session['history'], list):
                session['history'] = []

            for answer in answers:
                session['history'].append({
                    'question': answer['question'],
                    'answer': answer['answer'],
                    'source': answer['url'],
                    'content_excerpt': answer['content_excerpt'],
                    'paraphrased_content_excerpt': answer['paraphrased_content_excerpt']
                })
            
            session['history'] = limit_history(session['history'])
            save_history(username, session['history'])
        
        else:
            results = search_web(question)
            title = results['title']
            link = results['link']
            description = results['description']
            username = session['username']
            r=''
            d=''
            p=''
            if link and description and title:
                for i in range(len(link)):
                    r+=link[i]+','
                    d+=title[i]+' : '+description[i]
                    if description[i].strip():
                        p+=paraphrase_text(description[i])+'.'

            res = qa_pipeline(question=question, context=d)
            a = res.get('answer', 'No answer found')
            if 'history' not in session or not isinstance(session['history'], list):
                session['history'] = []
            session['history'].append({
                'question': question,
                'answer': a,
                'source': r,
                'content_excerpt': d,
                'paraphrased_content_excerpt': p
            })

            session['history'] = limit_history(session['history'])
            save_history(username, session['history'])

    user_history = session.get('history', [])
    return render_template('home.html', history=user_history)
    
if __name__ == '__main__':
    app.run(debug=True)