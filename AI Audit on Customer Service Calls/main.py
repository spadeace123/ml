import os
from groq import Groq
import jiwer

def Transcribe(audio_file_path,transcribe_model):
    """
    Transcribes an audio file using Groq API and saves the transcript to a file.
    """
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    file_name = os.path.splitext(os.path.basename(audio_file_path))[0]
    transcript_file = f"{file_name}_transcript.txt"
    
    with open(audio_file_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            file=audio_file,
            model=transcribe_model
        )
    
    transcript = response.text
    
    with open(transcript_file, "w", encoding="utf-8") as f:
        f.write(transcript)
    
    print(f"Transcription saved to {transcript_file}")
    return transcript_file

def WER(ground_truth_file, transcript_file):
    """
    Computes the Word Error Rate (WER) between the ground truth and transcribed text.
    """
    file_name = os.path.splitext(os.path.basename(transcript_file))[0]
    wer_output_file = f"{file_name}_wer_output.txt"
    transform_file = f"{file_name}_transform.txt"
    
    with open(ground_truth_file, "r", encoding="utf-8") as f:
        ground_truth = f.read().strip()
    
    with open(transcript_file, "r", encoding="utf-8") as f:
        hypothesis = f.read().strip()
    
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
    ])
    
    ground_truth_transformed = " ".join(transformation(ground_truth))
    hypothesis_transformed = " ".join(transformation(hypothesis))
    
    wer_score = jiwer.wer(ground_truth_transformed, hypothesis_transformed)
    
    with open(wer_output_file, "w", encoding="utf-8") as f:
        f.write(f"WER Score: {wer_score:.4f}\n")
        f.write("\nGround Truth (Transformed):\n" + ground_truth_transformed + "\n")
        f.write("\nTranscribed Text (Transformed):\n" + hypothesis_transformed + "\n")
    
    with open(transform_file, "w", encoding="utf-8") as f:
        f.write("Ground Truth (Transformed):\n" + ground_truth_transformed + "\n")
        f.write(hypothesis_transformed)
    
    print(f"WER calculation saved to {wer_output_file}")
    print(f"Transformed text saved to {transform_file}")
    return wer_output_file

def Evaluate(transform_file, evaluate_model):
    """
    Evaluates a customer service conversation using Groq API and saves the evaluation to a file.
    """
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    file_name = os.path.splitext(os.path.basename(transform_file))[0]
    evaluation_file = f"{file_name}_evaluation.txt"
    evaluation_csv = f"{file_name}_evaluation.csv"
     
    with open(transform_file, "r", encoding="utf-8") as f:
        conversation_text = f.read().strip()
    
    prompt = f"""
    You are an AI evaluator assessing customer service interactions. Review the conversation below based on the following criteria and assign scores (0-5) with justifications:
    
    1. Introduction: Did the agent introduce themselves properly?
    2. Acquire Customer Information: Did the agent collect necessary customer details?
    3. Politeness and Respect: Was the agent polite and respectful?
    4. Empathy and Understanding: Did the agent engage with the customer's concerns?
    5. Gratitude: Did the agent express gratitude when appropriate?
    6. Provide Conclusion: Did the agent summarize the customer's request?
    7. Clarifying Questions: Did the agent ask clarifying questions when needed?
    8. Clarity of Language: Was the agent's language clear and easy to understand?
    9. Relevance of Information: Did the agent provide relevant and accurate information?
    
    Conversation:
    {conversation_text}
    
    Evaluation Output:
    Provide scores (0-5) for each criterion with justifications. Also, include an overall summary and improvement suggestions.
    """
    
    response = client.chat.completions.create(
        model=evaluate_model,
        messages=[{"role": "system", "content": "You are an expert in evaluating customer service interactions."},
                  {"role": "user", "content": prompt}]
    )
    
    evaluation = response.choices[0].message.content
    
    with open(evaluation_file, "w", encoding="utf-8") as f:
        f.write(evaluation)
    
    print(f"Evaluation saved to {evaluation_file}")
    return evaluation_file


if __name__ == "__main__":
    audio_file = input("Enter the path to the audio file (e.g., example.mp3): ")
    transcribe_model = input("Enter the model name") 
    transcript_file = Transcribe(audio_file, transcribe_model)

    wer_output_file, transform_file = WER(ground_truth_file, transcript_file)
    
    evaluate_model = input("Enter the model name")  
    evaluation_file = Evaluate(transform_file, evaluate_model)

    print(f"Processing completed!\nTranscription saved at: {transcript_file}\nEvaluation saved at: {evaluation_file}\nCSV saved at: {evaluation_csv}")

