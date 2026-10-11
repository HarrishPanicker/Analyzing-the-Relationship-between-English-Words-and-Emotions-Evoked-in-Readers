Emotion Classification in Text with Transformers

Final Year Project, B.Sc. Computer Science (Data Science), Multimedia University.

All rights for the proposed research article belong with the organization associated and the technical report copyright belongs to Multimedia University @2026.

Fine-tuned BERT, RoBERTa and ELECTRA on a data-driven 6-class emotion scheme built by clustering GoEmotions' 27 labels. The clustered scheme beats the standard Ekman baseline across all three models.

Live demo: (https://youtu.be/fdn-2qI_xoY) 

Results
Model	Ekman scheme (baseline)	Clustered scheme (this work)
BERT	0.650	0.739
ELECTRA	0.648	0.745
RoBERTa	0.655	0.753

Macro-F1 on the test set. The clustered scheme beats Ekman on every model, and RoBERTa is the strongest model under both schemes.

Approach
Label design: Grouped GoEmotions' 27 emotions into 6 classes using agglomerative hierarchical clustering on TF-IDF features with cosine distance.
Modelling: Fine-tuned BERT, RoBERTa and ELECTRA with Hugging Face Transformers.
Analysis: EDA, correlation analysis and sentence-length error analysis to see where predictions degrade.
Dashboard: Streamlit app for live predictions and result visuals.
Repository structure
<img width="4103" height="5279" alt="diagram" src="https://github.com/user-attachments/assets/5e907737-4fe6-47cb-84e5-a9ef54dae92d" />

Run the dashboard locally
Download the GoEmotions_Artifacts folder,
ensure streamlit is available,
move to inside the folder using Anaconda with the required libraries using cd,
streamlit run app/app.py          

Python version: <!-- TODO, e.g. 3.10 -->

Data and model weights
Dataset: GoEmotions. 
Model weights: Only RoBERTa's weights are preserved, and they are within GoEmotion_Artifacts.
Limitations
GoEmotions skews toward American, young, male Reddit users, so the scheme may not transfer evenly to other groups.
Comments have limited multi-annotation, which caps label reliability.
Built with the NLP techniques I knew at the time; newer methods may do better.

Future work: add the distinct emotions found by clustering to Ekman's scheme, test the method on other datasets, and explore finer-grained schemes.

Author

Harrish Panicker
