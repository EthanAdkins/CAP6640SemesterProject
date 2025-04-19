import pandas as pd
import os 
import matplotlib.pyplot as plt

df = pd.read_csv(os.path.join(os.getcwd(),"results/Run1/log_history.csv"))

train =  df[df['loss'].notna()]   
eval = df[df['eval_loss'].notna()]  
plt.figure()
plt.plot(train['step'],train['loss'], label="Training Loss")
plt.plot(eval['step'],eval['eval_loss'], label="Evaluation Loss")

plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training vs Evaluation Loss")
plt.legend()
plt.grid(True)
plt.show()


# BERT SCORES
plt.figure()
plt.plot(eval['step'],eval['eval_bertscore_precision'], label="BERTScore Precision")

plt.xlabel("Step")
plt.ylabel("BERTScore Precision")
plt.title("Evaluation BERTScore Precision")
plt.legend()
plt.grid(True)
plt.show()

# BERT SCORES
plt.figure()
plt.plot(eval['step'],eval['eval_bertscore_recall'], label="BERTScore Recall", color='orange')


plt.xlabel("Step")
plt.ylabel("BERTScore Recall")
plt.title("Evaluation BERTScore Recall")
plt.legend()
plt.grid(True)
plt.show()

# BERT SCORES
plt.figure()

plt.plot(eval['step'],eval['eval_bertscore_f1'], label="BERTScore F1",color='purple')

plt.xlabel("Step")
plt.ylabel("BERTScore F1")
plt.title("Evaluation BERTScore F1")
plt.legend()
plt.grid(True)
plt.show()


# METEOR SCORES
plt.figure()

plt.plot(eval['step'],eval['eval_meteor'], label="METEOR Score",)

plt.xlabel("Step")
plt.ylabel("METEOR Score")
plt.title("METEOR Score")
plt.legend()
plt.grid(True)
plt.show()
