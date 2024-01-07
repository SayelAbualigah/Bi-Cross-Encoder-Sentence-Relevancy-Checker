import pandas as pd
from sentence_transformers import SentenceTransformer, util, CrossEncoder

# load biEncoder model
def load_bi_encoder():
    BiEncoder_name  = "sentence-transformers/all-mpnet-base-v2"
    bi_encoder = SentenceTransformer(BiEncoder_name)
    return bi_encoder  

# load crossEncoder model
def load_cross_encoder():
    CrossEncoder_path = r"C:\Users\o27911\OneDrive - First Abu Dhabi Bank\Desktop\AI\RFPs-Semantic_Search\CrossBin"
    cross_encoder = CrossEncoder(CrossEncoder_path)
    return cross_encoder    

# load data
def load_data():
    main_ = pd.read_csv('data/MainFAQ.csv')
    alternative = pd.read_excel('data/FAB_Variations_Only_Added_From_Automation_RD_28.9.23.xlsx')
    return main_, alternative

# generate embeddings for main and alternative
def generate_embeddings(question : str, bi_encoder : SentenceTransformer):
    # generate embeddings for main
    embeddings = bi_encoder.encode(question, convert_to_tensor=True)
    return embeddings

# compute similarity between main and alternative using biEncoder
def compute_similarity(main_embeddings, alternative_embeddings):
    similarity_socre = util.pytorch_cos_sim(main_embeddings, alternative_embeddings)
    return similarity_socre

# predict using crossEncoder
def predict_cross_encoder(main: str, alternative: str, cross_encoder: CrossEncoder):
    cross_encoder_score = cross_encoder.predict([(main, alternative)])
    return cross_encoder_score

# prepare target file 
def prepare(main: pd.DataFrame, alternative: pd.DataFrame):
    my_dict = {"ID":[],"question":[], "alternative":[], "biEncoder score":[], "crossEncoder score":[]}
    
    main_dataSet, alternative_datset = main, alternative
    
    biEncoder = load_bi_encoder()
    CrossEncoder = load_cross_encoder() 
    
    for ID in main["FAQ Id"].values:
        
        """
        # assuming you have a DataFrame called df with columns 'column1' and 'column2'
        # and you want to get all the values in 'column2' where 'column1' equals some value 'x'

        filtered_df = df.loc[df['column1'] == 'x']  # filter the DataFrame based on the condition
        column2_values = filtered_df['column2'].tolist()  # select the column of interest and convert to a list

        print(column2_values)  # print the list of values in 'column2' that meet the condition
        """
        main_question = main_dataSet[main_dataSet["FAQ Id"] == ID]["Question"].values[0]
        main_embeddings = generate_embeddings(main_question, biEncoder)
        
        alternative_list = alternative_datset[alternative_datset["FAQ Id"] == ID]["Question"].values
        
        my_dict["ID"].append(ID)
        my_dict["question"].append(main_question)
        my_dict["alternative"].append(None)
        my_dict["biEncoder score"].append(None)
        my_dict["crossEncoder score"].append(None)
        
        
        for q in alternative_list:
            my_dict["ID"].append(None)
            my_dict["question"].append(None)
            
            alternative_embeddings = generate_embeddings(q, biEncoder)
            similarity_score = compute_similarity(main_embeddings, alternative_embeddings)
            cross_encoder_score = predict_cross_encoder(main_question, q, CrossEncoder)
            my_dict["alternative"].append(q)
            my_dict["biEncoder score"].append(similarity_score[0])
            my_dict["crossEncoder score"].append(cross_encoder_score.item())
            
    return my_dict

if __name__ == "__main__":
    main, alternative =  load_data()
    my_dict = prepare(main, alternative)
    df = pd.DataFrame(my_dict)
    df.to_csv("data/Results.csv", index=False)