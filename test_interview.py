import boto3
from pathlib import Path
import tempfile 
import streamlit as st
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from gtts import gTTS
from io import BytesIO
import speech_recognition as sr
import os
import time
import json
import numpy as np
from audiorecorder import audiorecorder


# Initialisation du client Polly
polly_client = boto3.client('polly',
                           aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
                           aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
                           region_name='us-east-1')

# Configuration Groq
client = Groq(api_key=st.secrets["GROQ_API_KEY"])   

# ------------------------------------------------------------
# CLASSES ET FONCTIONS AMÉLIORÉES
# ------------------------------------------------------------
class RecruitingAgent:
    def __init__(self):
        self.phase = "initial"
        self.history = []
        self.vector_db = None
        self.job_description = ""
        self.cv_text = ""
        self.evaluations = []
        self.job_summary = {}
        self.candidate_profile = {
            "competences_identifiees": [],
            "experiences_cles": [],
            "points_forts": [],
            "points_faibles": [],
            "motivations": []
        }
        self.question_count = 0
        self.max_questions = 6
        
    def analyze_job_description(self, job_desc):
        """Analyse et résume la description de poste"""
        prompt = f"""
        [DESCRIPTION DE POSTE À ANALYSER]
        {job_desc}

        [INSTRUCTIONS]
        En tant qu'expert RH, analysez cette offre d'emploi et :
        1. Extrayez un titre concis (max 5 mots)
        2. Listez les 3 compétences techniques principales
        3. Listez les 3 qualités comportementales clés
        4. Identifiez le niveau d'expérience requis
        5. Extrayez 2 indicateurs de performance clés
        6. Résumez en 2 phrases les responsabilités principales

        Formattez la réponse en JSON avec ces clés :
        - "titre"
        - "competences_techniques"
        - "qualites_comportementales"
        - "niveau_experience"
        - "indicateurs_performance"
        - "responsabilites_principales"
        """
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        self.job_summary = eval(response.choices[0].message.content)
        return self.job_summary
    
    def initialize_interview(self, cv_file, job_description):
        """Initialise l'entretien avec analyse préalable"""
        self.job_description = job_description
        self.job_summary = self.analyze_job_description(job_description)
        
        # Traitement des documents
        self.cv_text, self.vector_db = self.process_documents(cv_file, job_description)
        self.phase = "interview"
        self.question_count = 0
        
        # Message d'introduction personnalisé
        intro_message = (
            f"**Entretien pour le poste : {self.job_summary['titre']}**\n\n"
            f"**Compétences recherchées :** {', '.join(self.job_summary['competences_techniques'])}\n"
            f"**Profil attendu :** {self.job_summary['niveau_experience']} avec {', '.join(self.job_summary['qualites_comportementales'])}\n\n"
            "Commençons par votre présentation. Pourriez-vous vous décrire en 2 minutes "
            "en mettant l'accent sur votre adéquation avec ce poste ?"
        )
        
        self.add_system_message(intro_message)
        self.question_count += 1
        
    def process_documents(self, cv_file, job_desc):
        """Traite les documents et extrait les informations clés"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cv_path = os.path.join(temp_dir, cv_file.name)
            with open(cv_path, "wb") as f:
                f.write(cv_file.getvalue())
            
            loader = PyPDFLoader(cv_path)
            cv_data = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
        cv_full_text = ''.join([d.page_content for d in cv_data])
        combined_text = f"CV DU CANDIDAT:\n{cv_full_text}\n\nDESCRIPTION DE POSTE:\n{job_desc}"
        
        return cv_full_text, FAISS.from_texts(
            texts=text_splitter.split_text(combined_text),
            embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        )

    def extract_candidate_insights(self, user_response):
        """Extrait des insights du candidat de sa réponse"""
        prompt = f"""
        [RÉPONSE DU CANDIDAT]
        {user_response}

        [INSTRUCTIONS]
        Extrayez les informations suivantes de la réponse :
        1. Compétences techniques mentionnées
        2. Expériences professionnelles significatives
        3. Points forts démontrés
        4. Points faibles potentiels
        5. Motivations exprimées

        Répondez en JSON avec ces clés :
        - "competences_identifiees": liste
        - "experiences_cles": liste
        - "points_forts": liste
        - "points_faibles": liste
        - "motivations": liste
        """
        
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                response_format={"type": "json_object"}
            )
            
            insights = eval(response.choices[0].message.content)
            
            # Mise à jour du profil candidat
            for key in self.candidate_profile.keys():
                if key in insights and insights[key]:
                    self.candidate_profile[key].extend(insights[key])
            
            return insights
            
        except Exception as e:
            print(f"Erreur extraction insights: {e}")
            return {}
    
    def check_candidate_cannot_continue(self, user_response):
        """Vérifie si le candidat indique ne pas pouvoir suivre la rencontre"""
        stop_phrases = [
            "je ne peux pas continuer",
            "je ne peux pas suivre",
            "je souhaite arrêter",
            "stop l'entretien",
            "arrêter l'entretien",
            "terminer maintenant",
            "je préfère arrêter",
            "je veux arrêter",
            "pas possible de continuer",
            "impossible de poursuivre"
        ]
        
        user_response_lower = user_response.lower()
        return any(phrase in user_response_lower for phrase in stop_phrases)
    
    def generate_contextual_question(self, user_response):
        """Génère des questions contextuelles basées sur l'historique"""
        # Vérifier si le maximum de questions est atteint
        if self.question_count >= self.max_questions:
            thank_you_message = (
                "Merci beaucoup pour vos réponses. Nous avons maintenant suffisamment "
                "d'éléments pour évaluer votre candidature. "
                "Passons à l'évaluation finale de votre profil."
            )
            return thank_you_message
        
        # Extraction des insights de la réponse
        insights = self.extract_candidate_insights(user_response)
        
        prompt = f"""
        [CONTEXTE DE L'ENTRETIEN]

        POSTE RECHERCHÉ:
        - Titre: {self.job_summary['titre']}
        - Compétences techniques requises: {self.job_summary['competences_techniques']}
        - Qualités comportementales: {self.job_summary['qualites_comportementales']}
        - Responsabilités: {self.job_summary['responsabilites_principales']}

        PROFIL DU CANDIDAT (extrait des réponses précédentes):
        - Compétences identifiées: {self.candidate_profile['competences_identifiees'][-3:] if self.candidate_profile['competences_identifiees'] else 'Aucune identifiée'}
        - Expériences clés: {self.candidate_profile['experiences_cles'][-2:] if self.candidate_profile['experiences_cles'] else 'Aucune mentionnée'}
        - Points forts: {self.candidate_profile['points_forts'][-2:] if self.candidate_profile['points_forts'] else 'Non spécifiés'}

        DERNIÈRE RÉPONSE DU CANDIDAT:
        {user_response[:500]}

        HISTORIQUE DE L'ENTRETIEN (dernières interactions):
        {str(self.history[-3:]) if len(self.history) > 3 else 'Début de conversation'}

        [INSTRUCTIONS]
        En tant que recruteur expert, posez UNE seule question qui:
        1. S'appuie sur la dernière réponse du candidat
        2. Explore les compétences requises pour le poste
        3. Vérifie l'adéquation avec les responsabilités du poste
        4. Maximum 70 mots
        5. En français, naturel et conversationnel

        La question doit créer un continuum logique avec l'échange précédent.
        """
        
        messages = [
            {"role": "system", "content": "Vous êtes un recruteur technique expert, posez des questions pertinentes et naturelles."},
            *[{"role": "assistant" if role == "system" else "user", "content": msg} 
              for role, msg in self.history[-4:]],  # Garder les 4 derniers échanges
            {"role": "user", "content": prompt}
        ]
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.7,
            max_tokens=100
        )
        
        self.question_count += 1
        return response.choices[0].message.content.strip()
    
    def add_system_message(self, message):
        """Ajoute un message système avec historique"""
        self.history.append(("system", message))
        st.session_state.messages.append({
            "role": "assistant",
            "content": message
        })

    def evaluate_response(self, user_response):
        """Évalue la réponse du candidat de manière contextuelle"""
        prompt = f"""
        [CONTEXTE D'ÉVALUATION]

        EXIGENCES DU POSTE:
        - Compétences: {self.job_summary['competences_techniques']}
        - Qualités: {self.job_summary['qualites_comportementales']}
        - Niveau: {self.job_summary['niveau_experience']}

        HISTORIQUE RÉCENT:
        {str(self.history[-2:]) if len(self.history) >= 2 else 'Première question'}

        RÉPONSE À ÉVALUER:
        {user_response}

        [CRITÈRES D'ÉVALUATION]
        1. Pertinence (1-5) : Adéquation avec la question posée
        2. Détail technique (1-5) : Précision des informations
        3. Adéquation poste (1-5) : Lien avec les exigences du poste
        4. Progression (1-5) : Amélioration par rapport aux réponses précédentes

        Fournissez un feedback constructif en 2-3 phrases maximum.
        """
        
        evaluation = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        ).choices[0].message.content
        
        self.evaluations.append({
            "reponse": user_response[:200],
            "evaluation": evaluation,
            "timestamp": time.time()
        })
        
        return evaluation

    def evaluate_performance(self):
        """Génère un rapport final complet"""
        prompt = f"""
        [SYNTHÈSE FINALE]

        POSTE: {self.job_summary['titre']}
        COMPÉTENCES REQUISES: {self.job_summary['competences_techniques']}

        PROFIL CANDIDAT IDENTIFIÉ:
        - Compétences: {self.candidate_profile['competences_identifiees']}
        - Expériences: {self.candidate_profile['experiences_cles']}
        - Points forts: {self.candidate_profile['points_forts']}
        - Motivations: {self.candidate_profile['motivations']}

        HISTORIQUE COMPLET: {json.dumps(self.history, ensure_ascii=False)}
        ÉVALUATIONS: {json.dumps(self.evaluations, ensure_ascii=False)}

        [INSTRUCTIONS]
        Rédigez un rapport détaillé en français avec:
        1. Score global sur 10 avec justification
        2. Points forts alignés avec le poste
        3. Points d'amélioration critiques
        4. Recommandations concrètes pour le candidat
        5. Adéquation finale avec le poste (Fort/Moyen/Faible)

        Structurez avec des sections claires et soyez constructif.
        """
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content

# ------------------------------------------------------------
# FONCTIONS AUDIO AMÉLIORÉES
# ------------------------------------------------------------
def text_to_speech(text):
    """Synthèse vocale avec AWS Polly TTS (voix française)"""
    try:
        # Création d'un fichier temporaire pour l'audio
        speech_file_path = Path(tempfile.gettempdir()) / "speech.mp3"
        
        # Appel AWS Polly API TTS
        response = polly_client.synthesize_speech(
            Engine='generative',
            OutputFormat='mp3',
            Text=text,
            VoiceId='Liam',
            TextType='text'
        )
        
        # Sauvegarde du fichier audio
        if 'AudioStream' in response:
            with open(speech_file_path, 'wb') as file:
                file.write(response['AudioStream'].read())
        
        return str(speech_file_path)
        
    except Exception as e:
        print(f"Erreur lors de la synthèse vocale: {e}")
        return None
#############################################################################
def speech_to_text():
    """Reconnaissance vocale via l'enregistrement audio (Streamlit + Whisper)"""
    st.info("🎤 Enregistrez votre réponse vocale")
    audio_input = st.audio_input("Cliquez pour démarrer l'enregistrement")
    if audio_input:
        #st.audio(audio_input, format="audio/wav")
        # Sauvegarde temporaire du fichier audio
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp_file.write(audio_input.getvalue())
        tmp_file.flush()
        tmp_path = tmp_file.name
    try:
            with open(tmp_path, "rb") as file:
                transcription = client.audio.transcriptions.create(
                    file=(tmp_path, file.read()),
                    model="whisper-large-v3-turbo",
                    response_format="verbose_json",
                    language="fr"
                )
                # Nettoyer le fichier temporaire
                os.unlink(tmp_path)
                st.session_state.audio_data = None
                st.rerun()
              
                # Effacer l'audio du widget pour permettre un nouvel enregistrement
                st.session_state.pop("audio_input", None)
            return transcription.text or "Aucune parole détectée. Veuillez réessayer."
    except Exception as e:
            return f"Erreur lors de la transcription: {e}"

        
# ------------------------------------------------------------
# INTERFACE UTILISATEUR AMÉLIORÉE
# ------------------------------------------------------------
def main():
    st.set_page_config(page_title="🤖 Assistant d'Entretien Intelligent", layout="wide")
    st.title("🤖 Assistant d'Entretien Intelligent")
    
    # Initialisation session
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "agent" not in st.session_state:
        st.session_state.agent = RecruitingAgent()
    
    if "waiting_for_response" not in st.session_state:
        st.session_state.waiting_for_response = False
    
    # Sidebar avec configuration
    with st.sidebar:
        st.header("📤 Configuration de l'entretien")
        cv_file = st.file_uploader("Téléversez votre CV (PDF)", type="pdf")
        job_desc = st.text_area("Description du poste", height=150,
                              placeholder="Copiez-collez l'offre d'emploi complète")
        
        if st.button("🔍 Analyser et démarrer", type="primary"):
            if cv_file and job_desc:
                with st.spinner("Analyse approfondie en cours..."):
                    st.session_state.agent.initialize_interview(cv_file, job_desc)
                    st.session_state.waiting_for_response = True
                    
                # Affichage du résumé
                st.success("Analyse terminée!")
                summary = st.session_state.agent.job_summary
                st.subheader("📋 Fiche de poste")
                st.json({
                    "Poste": summary["titre"],
                    "Compétences": summary["competences_techniques"],
                    "Qualités": summary["qualites_comportementales"],
                    "Niveau": summary["niveau_experience"]
                })
                
                st.rerun()
            else:
                st.error("Veuillez fournir CV et description de poste")
    
    # Zone principale de conversation
    if st.session_state.agent.phase == "interview":
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("💬 Entretien en cours")
            st.caption(f"Question {st.session_state.agent.question_count}/{st.session_state.agent.max_questions}")
            
            # Affichage messages
            for msg in st.session_state.messages:
                avatar = "🤖" if msg["role"] == "assistant" else "👤"
                with st.chat_message(msg["role"], avatar=avatar):
                    st.write(msg["content"])
                    if msg["role"] == "assistant":
                        audio_file = text_to_speech(msg["content"])
                        if audio_file:
                            st.audio(audio_file)

        with col2:
            st.subheader("📊 Progression")
            st.progress(st.session_state.agent.question_count / st.session_state.agent.max_questions)
            st.caption(f"{st.session_state.agent.question_count} questions posées sur {st.session_state.agent.max_questions}")
            
            st.subheader("📊 Profil candidat")
            if st.session_state.agent.candidate_profile:
                profile = st.session_state.agent.candidate_profile
                st.metric("Compétences identifiées", len(profile['competences_identifiees']))
                st.metric("Expériences clés", len(profile['experiences_cles']))
                
                with st.expander("Détails du profil"):
                    st.json(profile)
            
            st.subheader("🎯 Mode réponse")
            input_mode = st.radio("Choisissez:", ["🎤 Vocal", "📝 Texte"], index=1)

        # Gestion des réponses
        if st.session_state.waiting_for_response:
            user_input = None
            
            if input_mode == "🎤 Vocal":
                #if st.button("⏺️ Démarrer l'enregistrement", type="secondary"):
                    with st.spinner("Enregistrement en cours..."):
                        user_input = speech_to_text()
                        if user_input:
                            st.success("Transcription réussie!")
                            st.write(f"**Vous avez dit :** {user_input}")
            else:
                user_input = st.chat_input("Tapez votre réponse ici...")
            
            if user_input and not user_input.startswith("Erreur"):
                # Vérifier si le candidat veut arrêter
                if st.session_state.agent.check_candidate_cannot_continue(user_input):
                    st.warning("Le candidat a demandé d'arrêter l'entretien.")
                    st.session_state.agent.phase = "evaluation"
                    st.rerun()
                    return
                
                # Traitement de la réponse
                st.session_state.agent.history.append(("candidat", user_input))
                st.session_state.messages.append({"role": "user", "content": user_input})
                st.session_state.waiting_for_response = False
                
                # Génération question suivante
                with st.spinner("🔍 Analyse et génération de la prochaine question..."):
                    evaluation = st.session_state.agent.evaluate_response(user_input)
                    question = st.session_state.agent.generate_contextual_question(user_input)
                    
                    # Vérifier si c'est le message de remerciement final
                    if "merci beaucoup" in question.lower() or "évaluation finale" in question.lower():
                        st.session_state.agent.add_system_message(question)
                        st.session_state.agent.phase = "evaluation"
                        st.rerun()
                    else:
                        st.session_state.agent.add_system_message(question)
                
                # Affichage feedback
                with st.expander("📝 Feedback immédiat", expanded=True):
                    st.info(evaluation)
                
                st.session_state.waiting_for_response = True
                st.rerun()

        # Bouton terminer
        if st.session_state.waiting_for_response:
            if st.button("✅ Terminer l'entretien", type="primary"):
                st.session_state.agent.phase = "evaluation"
                st.rerun()

    # Évaluation finale
    if st.session_state.agent.phase == "evaluation":
        st.header("📊 Rapport de Performance Final")
        
        with st.spinner("📊 Génération du rapport détaillé..."):
            evaluation = st.session_state.agent.evaluate_performance()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            with st.expander("📝 Rapport Complet", expanded=True):
                st.markdown(evaluation)
        
        with col2:
            st.download_button(
                "💾 Exporter le rapport",
                evaluation,
                file_name="rapport_entretien.md",
                mime="text/markdown"
            )
            
            if st.button("🔄 Nouvel entretien"):
                st.session_state.clear()
                st.rerun()

if __name__ == "__main__":
    main()
