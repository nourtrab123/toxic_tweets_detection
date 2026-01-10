import pandas as pd
import random

# --- Fonctions pour générer des commentaires ---
english_toxic_comments = [
    "I hate you so much!",
    "You're the worst person ever!",
    "Go away, nobody likes you!",
    "This is disgusting and stupid!",
    "I can't stand your face!",
    "Everything you do is pathetic!",
    "You're a complete idiot!",
    "Shut up, nobody cares!",
    "You're worthless!",
    "This is the most annoying thing ever!",
    "Stop talking, you fool!",
    "I despise everything about you!",
    "You're ruining everything!",
    "I never want to see you again!",
    "You're completely useless!",
    "What a horrible person!",
    "I hate your guts!",
    "You are hopeless!",
    "You disgust me!",
    "Get lost, nobody wants you here!"
]

english_non_toxic_comments = [
    "I love this!",
    "What a wonderful day!",
    "You did a great job!",
    "This is so amazing!",
    "I appreciate your help!",
    "You are very kind!",
    "Thanks for your support!",
    "This is fantastic!",
    "You make me happy!",
    "Everything is going well!",
    "I'm proud of you!",
    "You are amazing!",
    "What a beautiful experience!",
    "I enjoy this so much!",
    "You are very talented!",
    "This made my day!",
    "Keep up the great work!",
    "I feel inspired!",
    "Such a lovely moment!",
    "You did perfectly!"
]

french_toxic_comments = [
    "Je te déteste totalement !",
    "Tu es la pire personne que j'aie jamais vue !",
    "Dégage, personne ne t'aime ici !",
    "C'est absolument dégoûtant et ridicule !",
    "Je ne supporte pas ton visage !",
    "Tout ce que tu fais est pathétique !",
    "Tu es complètement idiot !",
    "Tais-toi, personne ne s'en soucie !",
    "Tu es sans valeur !",
    "C'est la chose la plus agaçante qui soit !",
    "Arrête de parler, imbécile !",
    "Je méprise tout ce qui te concerne !",
    "Tu ruines tout !",
    "Je ne veux plus jamais te voir !",
    "Tu es complètement inutile !",
    "Quelle horrible personne !",
    "Je déteste tes tripes !",
    "Tu es désespéré !",
    "Tu me dégoûtes !",
    "Casse-toi, personne ne te veut ici !"
]

french_non_toxic_comments = [
    "J'adore ce produit !",
    "Quelle belle journée !",
    "Tu as fait un excellent travail !",
    "C'est vraiment génial !",
    "Merci pour ton aide !",
    "Tu es très gentil !",
    "Merci pour ton soutien !",
    "C'est fantastique !",
    "Tu me rends heureux !",
    "Tout se passe bien !",
    "Je suis fier de toi !",
    "Tu es incroyable !",
    "Quelle belle expérience !",
    "J'apprécie tellement ça !",
    "Tu es très talentueux !",
    "Cela a égayé ma journée !",
    "Continue ton excellent travail !",
    "Je me sens inspiré !",
    "Quel moment charmant !",
    "Tu as parfaitement réussi !"
]

def generate_comment():
    # Choisir aléatoirement langue et type
    lang = random.choice(["en", "fr"])
    is_toxic = random.choice([0, 1])
    
    if lang == "en":
        if is_toxic:
            comment = random.choice(english_toxic_comments)
        else:
            comment = random.choice(english_non_toxic_comments)
    else:
        if is_toxic:
            comment = random.choice(french_toxic_comments)
        else:
            comment = random.choice(french_non_toxic_comments)
    
    return comment, is_toxic

# --- Nombre de commentaires synthétiques à ajouter ---
N = 1000

# --- Chargement et mise à jour de dataset.csv ---
dataset_file = "dataset.csv"
try:
    df_dataset = pd.read_csv(dataset_file)
except FileNotFoundError:
    df_dataset = pd.DataFrame(columns=["comment", "toxicity"])

for _ in range(N):
    comment, toxicity = generate_comment()
    df_dataset = pd.concat([df_dataset, pd.DataFrame({"comment":[comment], "toxicity":[toxicity]})], ignore_index=True)

df_dataset.to_csv(dataset_file, index=False)
print(f"{N} commentaires ajoutés dans {dataset_file}.")

# --- Chargement et mise à jour de raw_tweets.csv ---
raw_file = "raw_tweets.csv"
try:
    df_raw = pd.read_csv(raw_file)
except FileNotFoundError:
    df_raw = pd.DataFrame(columns=["comment"])

for _ in range(N):
    comment, _ = generate_comment()
    df_raw = pd.concat([df_raw, pd.DataFrame({"comment":[comment]})], ignore_index=True)

df_raw.to_csv(raw_file, index=False)
print(f"{N} commentaires ajoutés dans {raw_file}.")
