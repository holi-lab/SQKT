import sys
sys.path.append('/home/sqkt')

import pandas as pd
import numpy as np
import random, re, logging, itertools
import torch
from torch.utils.data import random_split, Dataset, DataLoader, Subset,  ConcatDataset
import torch.nn as nn
from transformers import BertTokenizer, BertModel, T5ForConditionalGeneration, RobertaModel, RobertaTokenizer
from sklearn.model_selection import train_test_split
from collections import defaultdict



device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device type: {device}")


class ProjectionLayer(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(ProjectionLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU()


    def forward(self, x):
        x = x.view(-1, self.linear.in_features)
        return self.activation(self.linear(x))



class SkillDataset(Dataset):
    def __init__(self, code_file_path, text_file_path, help_center_path, target_dict, question_model, device, debug = False):
        """
        code_file_path: code submission data
        text_file_path: problem text data
        help_center_path: students' questions data
        target_dict: target data
        question_model: codet5 (shared with main.py file)
        device: cuda or cpu
        debug: debug option
        """
        self.device = device
        
        #code submission data 
        self.code_df = pd.read_csv(code_file_path)

        #problem description data
        self.text_df = pd.read_csv(text_file_path)

        #students' questions data
        self.help_center_log = pd.read_csv(help_center_path)

        #target load
        self.target_dict = target_dict

        self.debug = debug


        torch.cuda.empty_cache()


        if self.debug:
            logging.info(f"\n\n Check data type \n\n.")
            logging.info(f"Data type of created_datetime: {self.code_df['created_datetime'].dtype}") #Data type of created_datetime: float64
            logging.info(f"First few values of created_datetime: {self.code_df['created_datetime'].head()}")


        
            if pd.api.types.is_numeric_dtype(self.code_df['created_datetime']):
                logging.info(f"created_datetime appears to be numeric (likely timestamp)")

                min_timestamp = self.code_df['created_datetime'].min()
                max_timestamp = self.code_df['created_datetime'].max()
                logging.info(f"Timestamp range: {min_timestamp} to {max_timestamp}")
                
                if min_timestamp > 1e12:  
                    logging.info(f"Timestamps appear to be in milliseconds")
                else:
                    logging.info(f"Timestamps appear to be in seconds")
            else:
                logging.info(f"created_datetime appears to be non-numeric (likely string)")
                logging.info("Sample datetime strings:")
                logging.info(f"{self.code_df['created_datetime'].sample(5).tolist()}")

            """
            Timestamp range: 1642043709105.688 to 1711014749722.469
            Timestamps appear to be in milliseconds
            """

            # try parsing
            try:
                parsed_dates = pd.to_datetime(self.code_df['created_datetime']/1000, unit='s', utc=True)
                logging.info(f"Successfully parsed as datetime")
                logging.info(f"Parsed date range:")
                logging.info(f"Min: {parsed_dates.min()}")
                logging.info(f"Max: {parsed_dates.max()}")
            except ValueError as e:
                logging.info(f"Error parsing dates: {e}")

            """
            Successfully parsed as datetime
            Parsed date range:
            Min: 2022-01-13 03:15:09.105688064+00:00
            Max: 2024-03-21 09:52:29.722469120+00:00
            """

            logging.info(f"\n\n")

        
        """
        Convert the 'created_datetime' column's millisecond Unix timestamp to a pandas Timestamp object.
        1) Divide by 1000 to convert milliseconds to seconds.
        2) Specify unit='s' to indicate the data is in seconds.
        3) Use utc=True to set the timezone to UTC.
        After conversion, align the timezone formats of both files for easier comparison.
        """
        self.code_df['created_datetime'] = pd.to_datetime(self.code_df['created_datetime'] /1000, unit='s', utc=True)
        self.help_center_log['post_created_datetime'] = pd.to_datetime(self.help_center_log['post_created_datetime'])
        self.code_df['created_datetime'] = self.code_df['created_datetime'].dt.tz_convert('UTC')
        self.help_center_log['post_created_datetime'] = self.help_center_log['post_created_datetime'].dt.tz_convert('UTC')
        # print(self.help_center_log['post_created_datetime'][:5])


        #tokenizer + model 
        self.code_tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        self.text_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.question_tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5p-220m")
        self.code_model = RobertaModel.from_pretrained("microsoft/codebert-base").to(device)
        self.text_model = BertModel.from_pretrained('bert-base-uncased').to(device)
        self.question_model = question_model.to(device)
        self.resize_token_embeddings() 


        #set projection layer
        self.projection_text = ProjectionLayer(768, 512).to(device) #shared between problem text and skill embedding
        self.projection_code = ProjectionLayer(768, 512).to(device)
        self.projection_qna = ProjectionLayer(768, 512).to(device)

        # The structure of 'submission_order' is as follows:
        #        x_user_id                               exercise_id  submission_order
        # 0      40c258bc0488f4a4c10ba6ab7f249eff8992fd24317c45d81df1672685ced824      1548385                 1
        # 1      40c258bc0488f4a4c10ba6ab7f249eff8992fd24317c45d81df1672685ced824      1548385                 2
        # 2      40c258bc0488f4a4c10ba6ab7f249eff8992fd24317c45d81df1672685ced824      1548385                 3
        # 3      40c258bc0488f4a4c10ba6ab7f249eff8992fd24317c45d81df1672685ced824      1548385                 4
        # 4      40c258bc0488f4a4c10ba6ab7f249eff8992fd24317c45d81df1672685ced824      1548385                 5
        # 5      40c258bc0488f4a4c10ba6ab7f249eff8992fd24317c45d81df1672685ced824      1548385                 6
        # 6      40c258bc0488f4a4c10ba6ab7f249eff8992fd24317c45d81df1672685ced824      1548385                 7
        # 7      40c258bc0488f4a4c10ba6ab7f249eff8992fd24317c45d81df1672685ced824      1548415                 1
        # 8      40c258bc0488f4a4c10ba6ab7f249eff8992fd24317c45d81df1672685ced824      1548415                 2
        # 9      40c258bc0488f4a4c10ba6ab7f249eff8992fd24317c45d81df1672685ced824      1548415                 3
        
        # In this example:
        # The user submitted problem 1548385 a total of 7 times, followed by submitting problem 1548415 at least 3 times.

        self.code_df['submission_order'] = self.code_df.groupby(['x_user_id', 'exercise_id']).cumcount() + 1 


        # Stores the ID of the previously processed user.
        # This is used to reset the cumulative embedding whenever the user ID changes.
        self.previous_user_id = None 


        # Stores the cumulative embeddings of the current user's learning progress.
        # It accumulates information from the user's previous problem-solving history and the current problem.
        # This allows the model to track the user's learning progress in chronological order.
        # If the current user is the same as the user from the previous batch, `self.cumulative_embeddings` continues from the previous batch.
        self.cumulative_embeddings = [] 
        


        # Stores the ID of the previously processed problem.
        # Used to perform necessary actions (e.g., adding a new problem description embedding) whenever the problem ID changes.
        self.previous_problem_id = None 
        self.first_call = True

        # Structure of self.user_problem_map:
        # {
        #     'UserID1': {
        #         ProblemID1: [SubmissionIndex1, SubmissionIndex2, ...],
        #         ProblemID2: [SubmissionIndex1, SubmissionIndex2, ...],
        #         ...
        #     },
        #     'UserID2': {
        #         ProblemID1: [SubmissionIndex1, SubmissionIndex2, ...],
        #         ProblemID2: [SubmissionIndex1, SubmissionIndex2, ...],
        #         ...
        #     },
        #     ...
        # }
        
        # This dictionary maps each user ID to their associated problem IDs, which in turn map to a list of submission indices.
        # It is used to track and organize submission data for each user and problem combination.
        self.user_problem_map = self._create_user_problem_map()

        if self.debug:
            first_10_items = dict(itertools.islice(self.user_problem_map.items(), 10))
            logging.info(f"User problem map samples: {first_10_items}\n\n")
            """
            {
            '40c258bc0488f4a4c10ba6ab7f249eff8992fd24317c45d81df1672685ced824': defaultdict(<class 'list'>, 
                {1548385: [0, 1, 2, 3, 4, 5, 6], 
                1548415: [7, 8, 9], 
                1548498: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19], 
                1549215: [20, 21]}), 
            '5d39c1f4640c3cdd6c4bc5ac302a791299a6f2812312bd0eeaef64a246e034f4': defaultdict(<class 'list'>, 
                {1548498: [22, 23, 24, 25, 26], 
                1548709: [27, 28, 29, 30, 31, 32, 33, 34], 
                1557658: [35, 36], 
                1557778: [37, 38, 39], 
                1557907: [40, 41, 42, 43, 44, 45, 46, 47, 48]}), 
            '3addc2ea41ec032df9eca1b663d2b62d3b0fc65f7db0f72c21a88cea2ed4fdb7': defaultdict(<class 'list'>, 
                {1548498: [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69], 
                1548709: [70]}), 
            'c57caa469e36a8b8ca951fe44883cc84827e59b1618492aa934bedfa6ca531a9': defaultdict(<class 'list'>, 
                {1548415: [71, 72], 
                1551598: [73, 74], 
                1551714: [75, 76], 
                1554876: [77, 78]}), 
            '95481986af69e048d9b46d33e942076809a770c867f0c7243f638bc23a69ba43': defaultdict(<class 'list'>, 
                {1548385: [79, 80, 81], 
                1548498: [82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134], 1558476: [135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182], 
                1559312: [183, 184]}), 
            '72d00c455d2b166025dd8b63c1006a3d16c324d871221e22ba7379ad55cd01b9': defaultdict(<class 'list'>, 
                {1548498: [185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205], 
                1548709: [206, 207, 208, 209, 210, 211, 212], 
                1549445: [213, 214, 215, 216, 217, 218], 
                1550975: [219, 220], 
                1551307: [221, 222, 223], 
                1551465: [224, 225], 
                1558476: [226], 
                1558550: [227], 
                1559424: [228, 229]}), 
            }
            """

        self.previous_codet5_input_ids = None
        self.previous_codet5_attention_mask = None
        self.previous_codet5_targets = None



    def resize_token_embeddings(self):
        """
        Add special tokens like [SKILL], [TARGET]. 
        """
        special_tokens_dict = {'additional_special_tokens': ['[SKILL]', '[TARGET SKILL]', '[QUESTION]', '[TARGET]', '[CODE]', '[TEXT]']}
        self.text_tokenizer.add_special_tokens(special_tokens_dict)
        self.code_tokenizer.add_special_tokens(special_tokens_dict)
        self.question_tokenizer.add_special_tokens(special_tokens_dict)

        self.text_model.resize_token_embeddings(len(self.text_tokenizer))
        self.code_model.resize_token_embeddings(len(self.code_tokenizer))
        self.question_model.resize_token_embeddings(len(self.question_tokenizer))


    def __len__(self):
        # Calculates the total number of submissions for all problems solved by each student, excluding the final submission for each problem.
        # The final submission is excluded as it is intended to be used for testing or evaluation purposes rather than training.
        length = sum(len(problems) - 1 for problems in self.user_problem_map.values())
        return length

        
    def __getitem__(self, idx):
            """
            Args:
                idx (int): Index from the dataset.
            
            Returns:
                tuple: A tuple containing the following elements:
                    - cumulative_embeddings (torch.Tensor): Tensor of cumulative embeddings (student learning history). Shape: [N, 512] (N is the cumulative length).
                    - target (int): Target value for the given user and problem. Shape: scalar.
                    - anchor_embeddings (torch.Tensor): Anchor embeddings for the current problem. Shape: [M, 512] (M is the number of submissions for the current problem).
                    - positive_emb (torch.Tensor): Embeddings for positive samples. Shape: [M, 512].
                    - negative_emb (torch.Tensor): Embeddings for negative samples. Shape: [1, 512].
                    - positive_qna_emb (torch.Tensor): Embeddings for positive question-answer samples. Shape: [1, 512].
                    - negative_qna_emb (torch.Tensor): Embeddings for negative question-answer samples. Shape: [1, 512].
                    - codet5_input_ids (torch.Tensor): Input IDs for the Codet5 model. Shape: [P, 512] (P is the number of question-answer pairs).
                    - codet5_attention_mask (torch.Tensor): Attention mask for the Codet5 model. Shape: [P, 512].
                    - codet5_targets (torch.Tensor): Target IDs for the Codet5 model. Shape: [P, 512].
                    - original_data (dict): Dictionary containing information about the original data.
            """

            user_id = list(self.user_problem_map.keys())[user_idx]
            problem_list = list(self.user_problem_map[user_id].keys())
            problem_id = problem_list[problem_idx]


            # Detects a new user and resets the cumulative embeddings accordingly.
            if self.previous_user_id != user_id:
                self.cumulative_embeddings = []
                if self.debug:
                    logging.info("\n\nNew user detected. Resetting cumulative embeddings.")

        
            # Executes when the user changes or when transitioning to a different problem for the same user.
            # Generates a text embedding for the new problem and appends it to cumulative_embeddings.
            if self.previous_user_id != user_id or self.previous_problem_id != problem_id:
                text_embedding = self._embed_text(problem_id, next = False)
                self.cumulative_embeddings.append(text_embedding)
                if self.debug:
                    logging.info(f"\n\nNew problem detected. Adding text embedding to cumulative embeddings.")

                

            # Retrieves the submission indices for the student's current problem.
            submission_indices = self.user_problem_map[user_id][problem_id] #ex. [20, 21]

            if self.debug:
                logging.info(f"\n\nuser_id: {user_id}, problem_id: {problem_id}")
                logging.info(f"submission_indices: {submission_indices}")
                
            code_embeddings = []  # Stores the embeddings of code submissions.
            question_embeddings = []  # Stores the embeddings of questions if available; otherwise, a tensor of shape (1, 512).
            current_problem_embeddings = []  # Stores embeddings for the current problem, which later serve as the anchor.
            codet5_input_ids_list = []  # Stores student question tokens for Codet5 fine-tuning.
            codet5_attention_mask_list = []  # Stores attention masks for Codet5 fine-tuning.
            codet5_targets_list = []  # Stores teacher response tokens for Codet5 fine-tuning.

            flag = True 

            # Iterates over all submissions made by the current user for a single problem.
            for i, submission_idx in enumerate(submission_indices):
                if isinstance(submission_idx, int):
                    row = self.code_df.iloc[submission_idx]
                else:
                    row = self.code_df.iloc[submission_idx[1]] 

                code_embedding = self._embed_code(row['contents']) #[1, 512]
                code_embeddings.append(code_embedding) 
                
                # This list stores all code embeddings for multiple submissions made by a single student for the given problem. 
                # By the end of the loop, its length corresponds to the number of submissions.
                current_problem_embeddings.append(code_embedding) 

                current_submission_time = row['created_datetime']
                
                if i > 0:  # If not first submission
                    prev_submission_idx = submission_indices[i-1]
                    if isinstance(prev_submission_idx, int):
                        prev_submission_time = self.code_df.iloc[prev_submission_idx]['created_datetime']
                    else:
                        prev_submission_time = self.code_df.iloc[prev_submission_idx[1]]['created_datetime']
                else:  # if first submission
                    prev_submission_time = pd.Timestamp.min
                

                if current_submission_time.tzinfo is None:
                    current_submission_time = current_submission_time.tz_localize('UTC')
                if prev_submission_time.tzinfo is None:
                    prev_submission_time = prev_submission_time.tz_localize('UTC')

                # Checks the history of questions asked prior to the current submission.
                help_center_questions = self.help_center_log[
                    (self.help_center_log['x_user_id'] == user_id) &
                    (self.help_center_log['exercise_id'] == problem_id) &
                    (self.help_center_log['post_created_datetime'] > prev_submission_time) &
                    (self.help_center_log['post_created_datetime'] <= current_submission_time)
                ]
                
                
                if not help_center_questions.empty:
                    # if there is questions
                    if self.debug:
                        logging.info(f"\n\n============================================================")
                        logging.info(f"There is questions.")
                    questions, codet5_input_ids, codet5_attention_mask, codet5_targets = self._embed_questions_with_codet5(help_center_questions)
                    # questions = [#of questions, 512] or [1, 512]
                    question_embeddings.append(questions)
                    skills = self._embed_skills(help_center_questions)



                    question_embeddings.append(skills)
                    codet5_input_ids_list.append(codet5_input_ids)
                    codet5_attention_mask_list.append(codet5_attention_mask)
                    codet5_targets_list.append(codet5_targets)
                    self.previous_codet5_input_ids = codet5_input_ids #캐시 저장
                    self.previous_codet5_attention_mask = codet5_attention_mask
                    self.previous_codet5_targets = codet5_targets
                else:
                    flag = False
                   
                    # if no questions asked
                    question_embeddings.append(torch.zeros(1, 512).to(self.device))
                    
                    # Adds logic to fetch random fallback questions.
                    # This ensures Codet5 continues training even in the absence of student questions by providing a random question-answer pair.
                    # Since the loop iterates over the student's submission count, in cases where the question is unrelated to submissions, only one question is sent to the model.
                    random_question, random_answer = self._fetch_random_question()
                    skills = self._embed_skills(random_question)
                    if self.debug:
                        logging.info(f"\n\n============================================================")
                        logging.info(f"There is NO question. So we made random questions.")
                        logging.info(f"random_question: {random_question}")
                        logging.info(f"random_answer: {random_answer}")
                        
            
            code_embeddings = torch.cat(code_embeddings, dim=0) #[# of code submission, 512]
            question_embeddings = torch.cat(question_embeddings, dim=0) #[# of questions , 512] or [# of code submission, 512] if no question asked
            anchor_embeddings = torch.cat(current_problem_embeddings, dim=0) #[#of code submission, 512]

        
            pad_tensor = torch.full((1, 512), self.question_tokenizer.pad_token_id, dtype=torch.long).to(self.device)
            if flag == True:
                # If questions are present, all question-answer pairs related to the code submission are used for Codet5 training.
                codet5_input_ids = torch.cat(codet5_input_ids_list, dim=0) if codet5_input_ids_list else pad_tensor #[#of questions, 512] or [1, 512]
                codet5_attention_mask = torch.cat(codet5_attention_mask_list, dim=0) if codet5_attention_mask_list else pad_tensor
                codet5_targets = torch.cat(codet5_targets_list, dim=0) if codet5_targets_list else pad_tensor
            else:
                # If no questions are present, the question-answer pair used for Codet5 training is unrelated to the code submission.
                # In this case, only the first question-answer pair is utilized.
                codet5_input_ids = codet5_input_ids_list[0] if codet5_input_ids_list else pad_tensor #[#of questions, 512] or [1, 512]
                codet5_attention_mask = codet5_attention_mask_list[0] if codet5_attention_mask_list else pad_tensor
                codet5_targets = codet5_targets_list[0] if codet5_targets_list else pad_tensor
                # print("random shape: ", codet5_input_ids.shape)
                # always [1, 512] 

            
            self.cumulative_embeddings.append(code_embeddings)
            self.cumulative_embeddings.append(question_embeddings)

            next_problem_id = self._get_next_problem_id(user_id, problem_idx)
            next_problem_emb = self._embed_text(next_problem_id, next = True) if next_problem_id is not None and self.text_df['exercise_id'].eq(next_problem_id).any() else None
            next_problem_skill = self._embed_next_skills(next_problem_id) if next_problem_id is not None and self.text_df['exercise_id'].eq(next_problem_id).any() else None

            self.cumulative_embeddings.append(next_problem_emb) if next_problem_emb is not None else None
            self.cumulative_embeddings.append(next_problem_skill) if next_problem_skill is not None else None
            cumulative_embeddings = torch.cat(self.cumulative_embeddings, dim=0)
            # print("cumulative_embeddings: ", cumulative_embeddings.shape)

            self.previous_user_id = user_id
            self.previous_problem_id = problem_id
            
            if self.debug == True:
                logging.info(f"Current user_id: {user_id}, problem_id: {problem_id}, next_problem_id: {next_problem_id}")
                print(f"Cumulative Embeddings type: {cumulative_embeddings.dtype}")
                logging.info(f"Anchor Embeddings Shape: {anchor_embeddings.shape}")
                print(f"Anchor Embeddings type: {cumulative_embeddings.dtype}")
                logging.info(f"Anchor Embeddings type: {cumulative_embeddings.dtype}")
                logging.info(f"Submissions for current problem: {len(submission_indices)}")
                logging.info(f"how many question embeddings: {len(question_embeddings)}")
                logging.info(f"then how many real code submissions: {len(anchor_embeddings)}")
                logging.info(f"Cumulative Embeddings Shape: {cumulative_embeddings.shape}")
                logging.info(f"codet5_input_ids: {codet5_input_ids.shape}")
                logging.info(f"codet5_input_ids samples: {codet5_input_ids[0]}")
                logging.info(f"codet5_attention_mask: {codet5_attention_mask.shape}")
                logging.info(f"codet5_targets: {codet5_targets.shape}")
                logging.info(f"Final codet5_input_ids dtype: {codet5_input_ids.dtype}")
                codet5_input_ids = codet5_input_ids.long()
                output = self.question_model.generate(
                    input_ids=codet5_input_ids,
                    attention_mask=codet5_attention_mask,
                    max_length=100,
                    num_return_sequences=1,
                    no_repeat_ngram_size=2,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.7
                )
                decoded = self.question_tokenizer.decode(output[0], skip_special_tokens=True)
                logging.info(f"Tokenized output: {output[0]}")
                logging.info(f"Decoded output: {decoded}")
                logging.info(f"what is the decoded question?: {decoded}")
                codet5_input_ids = codet5_input_ids.long()
                codet5_targets = codet5_targets.long()
                codet5_loss = self.question_model(input_ids=codet5_input_ids, attention_mask=codet5_attention_mask, labels=codet5_targets).loss
                print(f"codet5_loss shape: {codet5_loss.shape}")
                logging.info(f"codet5_loss shape: {codet5_loss.shape}")
                print(f"codet5_loss type: {codet5_loss.dtype}")
                logging.info(f"codet5_loss type: {codet5_loss.dtype}")


            positive_emb = self._embed_text(problem_id).cpu()
            negative_problem_id = self._get_negative_problem_id(user_id, problem_id)
            negative_emb = self._embed_text(negative_problem_id).cpu()
            positive_qna_emb = self._get_question_positive_sample(user_id, problem_id).cpu()
            negative_qna_emb = self._get_question_negative_sample(user_id, problem_id).cpu()

            original_data = {
                'user_id': user_id,
                'problem_id': problem_id,
                'problem_order': problem_idx,
                'code_submissions': self._get_code_submissions(user_id, problem_id),
                'problem_text': self._get_problem_text(problem_id),
                'help_center_questions': self._get_question_text(user_id, problem_id),
                'skills' : skills
            }

            return cumulative_embeddings, self.target_dict.get((user_id, problem_id), 0), anchor_embeddings, positive_emb, negative_emb, positive_qna_emb, negative_qna_emb, codet5_input_ids, codet5_attention_mask, codet5_targets, original_data

        
    def _get_indices(self, idx):
        """
        self.user_problem_map = {
            'user1': {1001: [...], 1002: [...], 1003: [...]},
            'user2': {2001: [...], 2002: [...]},
            'user3': {3001: [...], 3002: [...], 3003: [...], 3004: [...]}
        }
        1. _get_indices(0):

        count = 0, user_idx = 0, problem_idx = 0 (user1's 1001)
        count == idx (0 == 0)


        2. _get_indices(2):

        count = 0, user_idx = 0, problem_idx = 0 (user1's 1001)
        count = 1, user_idx = 0, problem_idx = 1 (user1's 1002)
        count = 2, user_idx = 0, problem_idx = 2 (user1's 1003)
        count == idx (2 == 2)


        3. _get_indices(3):

        user1's all learning history (count = 0, 1, 2)
        count = 3, user_idx = 1, problem_idx = 0 (user2's 2001)
        count == idx (3 == 3)


        4. _get_indices(5):

        user1's all learning history (count = 0, 1, 2)
        user2's all learning history (count = 3, 4)
        count = 5, user_idx = 2, problem_idx = 0 (user3's 3001)
        count == idx (5 == 5)


        5. _get_indices(8):

        user1's all learning history (count = 0, 1, 2)
        user2's all learning history (count = 3, 4)
        user3's all learning history (count = 5, 6, 7, 8)
        count == idx (8 == 8)
        """
        
        count = 0
        users_list = list(self.user_problem_map.keys()) 
        
        for i, (user_id, problems) in enumerate(self.user_problem_map.items()):
            start_count = count  # Starting index for the current user.

            for problem_idx, problem_id in enumerate(problems.keys()):
                if count == idx:
                    if hasattr(self, 'first_call') and self.first_call: 
                        self.first_call = False  
                        if start_count != idx: 
                            for next_i in range(i + 1, len(users_list)):
                                next_user = users_list[next_i]
                                if self.user_problem_map[next_user]:
                                    return next_i, 0
                    return i, problem_idx 
                count += 1

        raise IndexError(f"Index {idx} out of range")


    def _get_next_problem_id(self, user_id, current_problem_idx):
        # Function that returns the ID of the next problem for a specific student (user_id) after the current problem (current_problem_idx).
        problem_list = list(self.user_problem_map[user_id].keys())
        if current_problem_idx < len(problem_list) - 1:
            return problem_list[current_problem_idx + 1]
        return None

    def _embed_code(self, content):
        """
            Function to embed the contents of a student's code submission.
        
            Args:
                content (str): A string containing the code content to be embedded.
        
            Returns:
                torch.Tensor: An embedding vector for the given code content. Shape: [1, 512]
        """

        token_types = '[CODE]'
        content = token_types + ' ' + str(content)
        code_tokens = self.code_tokenizer.tokenize(content)
        tokens = [self.code_tokenizer.cls_token] + code_tokens + [self.code_tokenizer.eos_token]
        tokens_ids = self.code_tokenizer.convert_tokens_to_ids(tokens)

        max_length = 256
        if len(tokens_ids) > max_length:
            tokens_ids = tokens_ids[:max_length]
        else: 
            tokens_ids = tokens_ids + [0] * (max_length - len(tokens_ids))

        with torch.no_grad():
            embedding = self.code_model(torch.tensor(tokens_ids)[None, :].to(self.device)).last_hidden_state.max(dim=1).values

        return self.projection_code(embedding)
    
    

    def _embed_text(self, ex_id, next = True):
        """
            Function to embed the description of a given problem ID.
    
            Args:
                ex_id (int): The ID of the problem to embed.
                next (bool): Determines whether to use information about the next problem. 
                             If True, uses the '[TARGET]' token type; if False, uses the '[TEXT]' token type.
    
            Returns:
                torch.Tensor: An embedding vector for the given problem description. Shape: [1, 512]
        """
        if next:
            max_length = 512
            token_type = '[TARGET]'
            text = self.text_df.loc[self.text_df['exercise_id'] == ex_id, ['Instruction Content', 'Solution Content']].values[0]
            text = ' '.join(str(t) for t in text)
            text = token_type + ' ' + text

            tokens_ids = self.text_tokenizer.encode(text, add_special_tokens=False, max_length=max_length-2, truncation=True)
            input_ids = torch.tensor([self.text_tokenizer.cls_token_id] + tokens_ids + [self.text_tokenizer.sep_token_id]).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.text_model(input_ids)
                embeddings = outputs.last_hidden_state.max(dim=1).values

            return self.projection_text(embeddings)
        else:
            max_length = 512
            token_type = '[TEXT]'
            text = self.text_df.loc[self.text_df['exercise_id'] == ex_id, ['Instruction Content', 'Solution Content']].values[0]
            text = ' '.join(str(t) for t in text)
            text = token_type + ' ' + text
            tokens_ids = self.text_tokenizer.encode(text, add_special_tokens=False, max_length=max_length-2, truncation=True)
            input_ids = torch.tensor([self.text_tokenizer.cls_token_id] + tokens_ids + [self.text_tokenizer.sep_token_id]).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.text_model(input_ids)
                embeddings = outputs.last_hidden_state.max(dim=1).values

            return self.projection_text(embeddings)
        
    def _embed_questions_with_codet5(self, help_center_questions):
       """
            Args:
                help_center_questions: A list of questions asked before submitting the current submission.
    
            Returns:
                If the question-answer pairs are consistent:
                    - question_embeddings (torch.Tensor): Embeddings of the questions. Shape: [number of questions, 512].
                    - codet5_input_ids (torch.Tensor): Input IDs for Codet5. Shape: [number of questions, 512].
                    - codet5_attention_masks (torch.Tensor): Attention masks for Codet5. Shape: [number of questions, 512].
                    - codet5_targets (torch.Tensor): Target tokens for Codet5. Shape: [number of questions, 512].
    
                Otherwise:
                    - question_embeddings (torch.Tensor): Embeddings of the questions. Shape: [number of questions, 512].
                    - codet5_input_ids (torch.Tensor): Input IDs for Codet5. Shape: [1, 512].
                    - codet5_attention_masks (torch.Tensor): Attention masks for Codet5. Shape: [1, 512].
                    - codet5_targets (torch.Tensor): Target tokens for Codet5. Shape: [1, 512].
        """

        question_embeddings = []  # List to store question embeddings.
        codet5_input_ids_list = []  # List to store question tokens for the Codet5 auxiliary task.
        codet5_attention_mask_list = []  # List to store attention masks for questions in the Codet5 auxiliary task.
        codet5_targets_list = []  # List to store response tokens for the Codet5 auxiliary task.

        for _, question in help_center_questions.iterrows():
            if question['is_student'] == True:
                question_text = f"question: {question['content']}"
                if self.debug:
                    logging.info(f"\n\nwhat is the question?: {question_text}")
                tokenized = self.question_tokenizer(question_text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
                inputs = tokenized.input_ids.to(self.device)
                attention_mask = tokenized.attention_mask.to(self.device)
                
                with torch.no_grad():
                    embedding = self.question_model.encoder(input_ids=inputs, attention_mask=attention_mask).last_hidden_state.mean(dim=1)
                    # embedding: [1, 512]
                    question_embeddings.append(embedding)
                
                codet5_input_ids_list.append(inputs)
                codet5_attention_mask_list.append(attention_mask)
            else:
                target_text = question['content']  # assuming the ground truth is stored in 'content'
                targets = self.question_tokenizer(target_text, return_tensors="pt", padding="max_length", truncation=True, max_length=512).input_ids.to(self.device)
                codet5_targets_list.append(targets)
                
        if not codet5_input_ids_list or not codet5_targets_list:
            codet5_input_ids_list = []
            codet5_attention_mask_list = []
            codet5_targets_list = []

            random_question, random_answer = self._fetch_random_question()

            random_question_text = f"question: {random_question}"
            tokenized = self.question_tokenizer(random_question_text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
            inputs = tokenized.input_ids.to(self.device)
            attention_mask = tokenized.attention_mask.to(self.device)

            codet5_input_ids_list.append(inputs)
            codet5_attention_mask_list.append(attention_mask)
                
            random_answer_text = f"answer: {random_answer}"
            tokenized_ans = self.question_tokenizer(random_answer_text, return_tensors='pt', padding="max_length", truncation=True, max_length=512)
            target_inputs = tokenized_ans.input_ids.to(self.device)
            codet5_targets_list.append(target_inputs)
            if self.debug:
                logging.info(f"No match. Random fetch.")
                logging.info(f"question: {random_question_text}")
                logging.info(f"answer: {random_answer_text}")

        max_length = max(len(codet5_input_ids_list), len(codet5_targets_list))
        for list_to_pad in [codet5_input_ids_list, codet5_attention_mask_list, codet5_targets_list]:
            while len(list_to_pad) < max_length:
                list_to_pad.append(torch.full((1, 512), self.question_tokenizer.pad_token_id, dtype=torch.long).to(self.device))


        pad_tensor = torch.full((1, 512), self.question_tokenizer.pad_token_id, dtype=torch.long).to(self.device)

        if question_embeddings:
            question_embeddings = self.projection_qna(torch.cat(question_embeddings, dim=0)) #[# of questions, 512]
        else:
            question_embeddings = pad_tensor

        if codet5_input_ids_list:
            codet5_input_ids = torch.cat(codet5_input_ids_list, dim = 0) #[# of questions, 512]
        else:
            codet5_input_ids = pad_tensor

        if codet5_attention_mask_list:
            codet5_attention_mask = torch.cat(codet5_attention_mask_list, dim = 0) #[# of questions, 512]
        else:
            codet5_attention_mask = pad_tensor

        if codet5_targets_list:
            codet5_targets = torch.cat(codet5_targets_list, dim = 0) #[# of answers, 512]
        else:
            codet5_targets = pad_tensor

        return question_embeddings, codet5_input_ids, codet5_attention_mask, codet5_targets

    def _fetch_random_question(self):
        """
        Fetch a random student question and its corresponding teacher answer from the dataset.

        Returns:
            A randomly selected student question and its corresponding teacher answer as a string.
        """
        student_questions = self.help_center_log[self.help_center_log['is_student'] == True]

        
        if not student_questions.empty:
            random_index = random.randint(0, len(student_questions) - 1)
            random_question_row = student_questions.iloc[random_index]
            random_question = random_question_row['content'] 
            exercise_id = random_question_row['exercise_id'] 
            post_time = random_question_row['post_created_datetime'] 


            # Fetching the corresponding teacher response (intended to retrieve the first teacher response following the student's question).
            teacher_answers = self.help_center_log[
                (self.help_center_log['exercise_id'] == exercise_id) &
                (self.help_center_log['is_student'] == False) &
                (self.help_center_log['post_created_datetime'] >= post_time) 
            ]

            if not teacher_answers.empty:
                teacher_answer = teacher_answers.iloc[0]['content']
            else:
                teacher_answer = "No corresponding teacher response found." 


            return random_question, teacher_answer

        return None, None

    def _embed_question(self, question_text):
        """
        Function to embed the given question text using the Codet5 model.

        Args:
            question_text (str): A string containing the content of the question.

        Returns:
            torch.Tensor: The question embedding generated by the Codet5 model.
        """

        max_length = 512
        question_text = f"question: {question_text}"
        
        inputs = self.question_tokenizer(question_text, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length).input_ids.to(self.device)
        attention_mask = self.question_tokenizer(question_text, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length).attention_mask.to(self.device)
        
        with torch.no_grad():
            embedding = self.question_model.encoder(input_ids=inputs, attention_mask=attention_mask).last_hidden_state.mean(dim=1)
        
        return self.projection_qna(embedding)
    
    def _embed_skills(self, questions):
        """
        Extracts skills from the given question data (using the _process_skills function) and creates embeddings.

        Args:
            questions (str, pd.DataFrame, list): The question data, which can be provided in various formats 
                                                 (DataFrame if questions exist, string if none).

        Returns:
            torch.Tensor: A tensor containing embeddings of all extracted skills, concatenated vertically.
                          If no skills are extracted, returns a zero-filled tensor of size [1, 768].
                          This will later pass through a projection layer to become size [1, 512].

        Process:
            1. If the input question is a string, it is converted into a list for consistent processing.
            2. If the question is a pandas DataFrame, skills are extracted and embeddings are created 
               for each 'content' row.
            3. If the question is a list, skills are extracted and embeddings are created for each item.
            4. All skill embeddings are concatenated into a tensor and returned. 
               If no skill embeddings exist, a zero-filled tensor of size [1, 768] is returned.
        """

        skill_embeddings = []

        if isinstance(questions, str):
            questions = [questions] 

        if isinstance(questions, pd.DataFrame):
            for _, question in questions.iterrows(): 
                content = str(question['content'])
                embeddings = self._process_skill(content) #skill extract
                skill_embeddings.append(embeddings)
        

        elif isinstance(questions, list):
            for question in questions: 
                embeddings = self._process_skill(question)
                skill_embeddings.append(embeddings)

        skill_embeddings = torch.cat(skill_embeddings, dim=0) if skill_embeddings else torch.zeros(1, 768).to(self.device)

        return skill_embeddings


    def _process_skill(self, content):
        """
        Helper function to extract skills from a given question (str) and create embeddings.

        Args:
            content (str): The sentence from which skills need to be extracted.

        Returns:
            torch.Tensor: A tensor embedding the extracted skills. If no skills are extracted, returns a default embedding. Shape: [1, 512]

        Process:
            1. Use regular expressions to extract code blocks and error messages from the given content.
            2. Remove the code and error messages, then analyze the remaining text to determine the analysis type.
            3. Extract skills based on the extracted code, errors, text, and analysis type (using the _extract_skills function).
            4. Combine the extracted skills into a single space-separated string and convert it into an embedding vector.
        """

        
        text = content.strip()
        
        error_pattern = r'(\w+Error): (.+)'  # error pattern
        code_pattern = r'```(.*?)```'  # code block pattern

        
        errors = re.findall(error_pattern, content, re.DOTALL)

        
        remaining_content = re.sub(error_pattern, '', content, re.DOTALL)
        codes = re.findall(code_pattern, remaining_content, re.DOTALL)

        
        if "설명해" in content or "문제" in content or "설명해" in content:
            analysis = "Explanation needed"
        elif "버그" in content:
            analysis = "Bug fixing"
        elif "번역" in content:
            analysis = "Translation"
        elif "리팩터" in content:
            analysis = "Refactoring"
        elif "시간복잡도" in content:
            analysis = "Time complexity"
        elif "의도" in content:
            analysis = "Intent understanding"
        else:
            analysis = "Unknown"

        
        codes = self._parse_content_column(codes)
        errors = self._parse_content_column(errors)
        text = self._parse_content_column(text)
        analysis = self._parse_content_column(analysis)

        
        skills = self._extract_skills(codes, errors, text, analysis)

        if self.debug:
            logging.info(f"Extracted skills: {skills}")

        
        skill_text = ' '.join(skills)

        
        embedding = self._embed_skill(skill_text)

        return embedding
    

    def _parse_content_column(self, content):
        """
        Function to prepare list-format data required for subsequent tasks such as skill extraction.
        """
        if isinstance(content, str):
            content = content.strip("[]").replace("'", "").split(", ")
        return content if isinstance(content, list) else [str(content)]


    def _add_skill(self, skills, skill):
        if skill not in skills:
            skills.append(skill)

    def _extract_skills(self, codes, errors, texts, analysis):
        """
        Function responsible for the model's skill extractor system.

        Args:
            codes (list): List of code blocks.
            errors (list): List of error messages.
            texts (list): List of textual descriptions.
            analysis (list): List of analysis types.

        Returns:
            list: A list of extracted skills, with duplicates removed.

        Process:
            1. Extracts skills by identifying patterns within the code blocks.
            2. Extracts skills by identifying specific error types in the error messages.
            3. Extracts skills based on specific keywords found in the text descriptions.
            4. Adds specific skills based on the analysis type.
            5. Ensures each skill is added only once (using the add_skill function).
        """

        skills = []

        code_patterns = {
            'Value': r'\b\d+(\.\d+)?\b',
            'Variable Assign': r'\b\w+\s*=\s*.+',
            'Operators': r'\b\w+\s*[+\-*/%]\s*\w+|\b\d+\s*[+\-*/%]\s*\d+',
            'Operands': r'\b\w+\s*[+\-*/%]\s*\w+|\b\d+\s*[+\-*/%]\s*\d+',
            'Type Convertor': r'\b(int|float|str)\b',
            'input function': r'\binput\(.+\)',
            'print function': r'\bprint\(.+\)',
            'Boolean Values': r'\b(True|False)\b',
            'Boolean Expressions': r'\b==\b|\b<=\b|\b>=\b|\b>\b|\b<\b',
            'If-Else Statements': r'\bif\b',
            'For Loops': r'\bfor\b',
            'While Loops': r'\bwhile\b',
            'Break Statement': r'\bbreak\b',
            'Continue Statement': r'\bcontinue\b',
            'return Statement': r'\breturn\b',
            'Local, Global Scope': r'\b(global|local)\b',
            'Lists': r'\[.*\]|\.append\(|\.insert\(|\.count\(|\.extend\(|\.index\(|\.reverse\(|\.sort\(|\.remove\(|\s*\+\s*\[.*\]|\s*\*\s*\d+|\[.*:.*\]|\.copy\(',
            'Dictionaries': r'\.keys\(|\.values\(|\.items\(|\.setdefault\(|\.copy\(',
            'Strings': r'\bstr\b',
            'Indexing': r'\[.*\]',
            'Import Statement': r'\bimport\b',
            'random': r'\brandom\b',
        }

        error_patterns = {
            'SyntaxError': r'SyntaxError',
            'NameError': r'NameError',
            'TypeError': r'TypeError',
            'IndentationError': r'IndentationError',
            'ValueError': r'ValueError',
            'AttributeError': r'AttributeError',
            'IndexError': r'IndexError',
            'KeyError': r'KeyError',
            'TabError': r'TabError',
            'UnicodeDecodeError': r'UnicodeDecodeError',
            'FileNotFoundError': r'FileNotFoundError',
            'ModuleNotFoundError': r'ModuleNotFoundError',
            'ZeroDivisionError': r'ZeroDivisionError',
            'UnboundLocalError': r'UnboundLocalError',
            'ImportError': r'ImportError',
            'UnicodeEncodeError': r'UnicodeEncodeError',
            'LookupError': r'LookupError',
            'ConnectionError': r'ConnectionError',
            'RuntimeError': r'RuntimeError',
        }

        text_patterns = {
            'Value': r'\b숫자\b|\b소수\b|\b정수\b|\b부동 소수\b',
            'Variable Assign': r'\b변수\b|\b할당\b|\b넣\b|\b넣을\b',
            'Operators': r'[\+\-\*/%]|\b더하기\b|\b빼기\b|\b곱하기\b|\b나누기\b|\b제곱\b',
            'Operands': r'\b\d+\s*[+\-*/%]\s*\d+\b|\b넣\b|\b빼\b',
            'Type Convertor': r'\b정수 변환\b|\b부동 소수\b|\b문자열\b|\btype\b',
            'input function': r'\b입력\b|\binput\b',
            'print function': r'\b출력\b|\b프린트\b|\bprint\b',
            'Boolean Values': r'\b참\b|\b거짓\b|\btrue\b|\bfalse\b',
            'Boolean Expressions': r'\b같음\b|\b같은\b|\b다름\b|\b다른\b|\b크다\b|\b큰\b|\b작다\b|\b작은\b|\b크거나 같다\b|\b크거나 같은\b|\b작거나 같다\b|\b작거나 같은\b',
            'Logical Operators': r'\b그리고\b|\b또는\b|\b부정\b|\b아닐 때\b',
            'If-Else Statements': r'\bif\b|\belse\b',
            'For Loops': r'\bfor\b|\bfor 루프\b|\bfor 문\b',
            'While Loops': r'\bwhile\b|\bwhile 조건문\b',
            'Break Statement': r'\b중단\b|\bbreak\b',
            'Continue Statement': r'\b계속\b|\bcontinue\b',
            'return Statement': r'\b리턴\b|\breturn\b',
            'Local, Global Scope': r'\b전역\b|\b지역\b',
            'Strings': r'\b문자열\b|\b스트링\b',
            'Indexing': r'\[.*\]|\b인덱스\b|\b인덱싱\b',
            'Lists': r'\b리스트\b',
            'Dictionaries': r'\b딕셔너리\b',
            'Import Statement': r'\bimport\b|\b수입\b',
            'random': r'\b랜덤\b',
            'File Operations': r'\b파일\b',
        }

        analysis_patterns = {
            "Explanation Skill": r'Explanation needed',
            "Debugging Skill": r'Bug fixing',
            "Translation Skill": r'Translation',
            "Refactoring Skill": r'Refactoring',
            "Time Complexity Analysis": r'Time complexity',
            "Intent Understanding": r'Intent understanding'
        }


        for contents in codes:
            contents = str(contents)
            for skill, pattern in code_patterns.items():
                if re.search(pattern, contents):
                    self._add_skill(skills, skill)
                

        for contents in errors:
            contents = str(contents)
            for skill, pattern in error_patterns.items():
                if re.search(pattern, contents):
                    self._add_skill(skills, skill)


        for text in texts:
            text = str(text)
            for skill, pattern in text_patterns.items():
                if re.search(pattern, text):
                    self._add_skill(skills, skill)


        for item in analysis:
            item = str(item)
            for skill, pattern in analysis_patterns.items():
                if re.search(pattern, item):
                    self._add_skill(skills, skill)



        return skills


        

    def _embed_skill(self, skill_text):
        """
        Function to embed the given skill text.

        Args:
            skill_text (str): A string describing the skill.

        Returns:
            torch.Tensor: An embedding for the skill text. Shape: [1, 512]
        """

        max_length = 512
        token_type = '[SKILL]'
        text = f"{token_type} {skill_text}" if skill_text else token_type

        tokens_ids = self.text_tokenizer.encode(text, add_special_tokens=False, max_length=max_length-2, truncation=True)
        input_ids = torch.tensor([self.text_tokenizer.cls_token_id] + tokens_ids + [self.text_tokenizer.sep_token_id]).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.text_model(input_ids)
            embeddings = outputs.last_hidden_state.max(dim=1).values

        return self.projection_text(embeddings)
    
    def _embed_next_skills(self, ex_id):
        """
        Function to embed the skills required to solve the next problem for the given ex_id.

        Args:
            ex_id (int): The ID of a specific problem.

        Returns:
            torch.Tensor: An embedding of the extracted skill text. Shape: [1, 512]
        """

        max_length = 512
        token_type = '[TARGET SKILL]'
        
        row = self.text_df.loc[self.text_df['exercise_id'] == ex_id].iloc[0]
        
        instruction_content = row['Instruction Content']
        solution_content = row['Solution Content']
        
        instruction_content = self._parse_content_column(instruction_content)
        solution_content = self._parse_content_column(solution_content)

        skills = self._extract_skills(solution_content, [], instruction_content, [])
        
        if self.debug:
            logging.info(f"Extracted skills from solution: {skills}")
            logging.info(f"Instruction: {instruction_content}")
            logging.info(f"Solution: {solution_content}")
        
        skill_text = ' '.join(skills)

        text = token_type + ' ' + skill_text
        
        tokens_ids = self.text_tokenizer.encode(text, add_special_tokens=False, max_length=max_length-2, truncation=True)
        input_ids = torch.tensor([self.text_tokenizer.cls_token_id] + tokens_ids + [self.text_tokenizer.sep_token_id]).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.text_model(input_ids)
            embeddings = outputs.last_hidden_state.max(dim=1).values
        
        return self.projection_text(embeddings)

    
    def _create_user_problem_map(self):
    
        # Stores the indices of submitted codes (actual indices from code_df) for each problem (exercise_id) solved by each student (x_user_id).
        # This allows for managing each student's problem-solving history and provides quick access to their submission records when needed.

        user_problem_map = defaultdict(lambda: defaultdict(list))
        for idx, row in self.code_df.iterrows():
            user_problem_map[row['x_user_id']][row['exercise_id']].append(idx)
            
        if self.debug:
            logging.info(f"\n\nAdding index {idx} for user {row['x_user_id']} and problem {row['exercise_id']}")
                
        return user_problem_map


    def _get_positive_sample(self, user_id, problem_id, current_idx):
        """
        Function to retrieve another submission made by a given user for a specific problem.
        Selects one submission different from the current submission (current_idx) for the same user and problem.

        Args:
            user_id (str): The user ID.
            problem_id (int): The current problem ID.
            current_idx (int): The index of the current submission.

        Returns:
            int: The index of the submission selected as the positive sample.
        """

        possible_positives = [i for i in self.user_problem_map[user_id][problem_id] if i != current_idx]
        positive_idx = np.random.choice(possible_positives) if possible_positives else current_idx
        return positive_idx

    def _get_negative_sample(self, user_id, problem_id, current_idx):
        """
        Function to retrieve a submission either from a different problem solved by the given user 
        or from a completely different user. Selects one submission as a negative sample.

        Args:
            user_id (str): The user ID.
            problem_id (int): The current problem ID.
            current_idx (int): The index of the current submission.

        Returns:
            int: The index of the submission selected as the negative sample.
        """
    
        possible_negatives = []
        for prob_id in self.user_problem_map[user_id]:
            if prob_id != problem_id:
                possible_negatives.extend(self.user_problem_map[user_id][prob_id])
        if not possible_negatives:
            possible_negatives = [i for i in range(len(self.code_df)) if i != current_idx and self.code_df.iloc[i]['x_user_id'] != user_id]
        negative_idx = np.random.choice(possible_negatives) if possible_negatives else current_idx
        return negative_idx


    def _get_negative_problem_id(self, user_id, problem_id):
        possible_negatives = [prob_id for prob_id in self.user_problem_map[user_id] if prob_id != problem_id]
        
        if possible_negatives:
            negative_problem_id = random.choice(possible_negatives)
        else:
            negative_problem_id = problem_id
        
        return negative_problem_id


    def _get_question_positive_sample(self, user_id, problem_id):

        questions = self.help_center_log[(self.help_center_log['x_user_id'] == user_id) & (self.help_center_log['exercise_id'] == problem_id)]
        if not questions.empty:
            question = questions.iloc[0]['content']
            return self._embed_question(question).cpu()

        return torch.zeros(1, 512).cpu()

    def _get_question_negative_sample(self, user_id, problem_id):

        questions = self.help_center_log[(self.help_center_log['x_user_id'] != user_id) & (self.help_center_log['exercise_id'] != problem_id)]
        if not questions.empty:
            question = questions.iloc[0]['content']
            return self._embed_question(question).cpu()

        return torch.zeros(1, 512).cpu()

    def _get_code_submissions(self, user_id, problem_id):
        
        submissions = self.code_df[(self.code_df['x_user_id'] == user_id) & (self.code_df['exercise_id'] == problem_id)]
        return submissions['contents'].tolist()

    def _get_problem_text(self, problem_id):
        
        problem = self.text_df[self.text_df['exercise_id'] == problem_id]
        return problem['Instruction Content'].iloc[0] if not problem.empty else ""

    def _get_question_text(self, user_id, problem_id):
        questions = self.help_center_log[(self.help_center_log['x_user_id'] == user_id) & 
                                         (self.help_center_log['exercise_id'] == problem_id)&
                                         self.help_center_log['is_student'] == True]
        return questions['content'].tolist() if not questions.empty else []
    
    def _get_next_problem_id(self, user_id, current_problem_idx):
        problem_list = list(self.user_problem_map[user_id].keys())
        if current_problem_idx < len(problem_list) - 1:
            return problem_list[current_problem_idx + 1]
        return None



def collate_fn(batch):
    max_len = max(len(item[0]) for item in batch) #가장 긴 시퀀스 길이
    embedding_dim = batch[0][0].size(1) #512

    # Initialize padded tensors
    padded_embeddings = torch.zeros((len(batch), max_len, embedding_dim), dtype=torch.float)
    padded_anchors = torch.zeros((len(batch), max_len, embedding_dim), dtype=torch.float)
    padded_positives = torch.zeros((len(batch), embedding_dim), dtype=torch.float)
    padded_negatives = torch.zeros((len(batch), embedding_dim), dtype=torch.float)
    padded_question_positives = torch.zeros((len(batch), embedding_dim), dtype=torch.float)
    padded_question_negatives = torch.zeros((len(batch), embedding_dim), dtype=torch.float)
    targets = torch.zeros((len(batch),), dtype=torch.float)
    original_data = []

    for i, (embeddings, target, anchor, positive, negative, question_positive, question_negative, codet5_input_ids, codet5_attention_mask, codet5_targets, data) in enumerate(batch):
        padded_embeddings[i, :len(embeddings)] = embeddings
        padded_anchors[i, :len(anchor)] = anchor
        padded_positives[i, :] = positive
        padded_negatives[i, :] = negative
        padded_question_positives[i, :] = question_positive
        padded_question_negatives[i, :] = question_negative
        targets[i] = target
        original_data.append(data)
        
    return padded_embeddings, targets, padded_anchors, padded_positives, padded_negatives, padded_question_positives, padded_question_negatives, codet5_input_ids, codet5_attention_mask, codet5_targets, original_data

        
def load_targets(target_file_path):
    """
        target_dict = {
            ('user1', 1001): 1,
            ('user1', 1002): 0,
            ('user2', 1001): 1
        }

        return example:

        ('de832cc1dc9bfecd21edbead3cfd08f97cc418e0b2cf7abddf7a06672e86737a', 1549839.0): 1.0, 
        ('de832cc1dc9bfecd21edbead3cfd08f97cc418e0b2cf7abddf7a06672e86737a', 1549976.0): 1.0, 
        ('de832cc1dc9bfecd21edbead3cfd08f97cc418e0b2cf7abddf7a06672e86737a', 1550130.0): 1.0, 
        ('de832cc1dc9bfecd21edbead3cfd08f97cc418e0b2cf7abddf7a06672e86737a', 1550975.0): 1.0, 
        ('de832cc1dc9bfecd21edbead3cfd08f97cc418e0b2cf7abddf7a06672e86737a', 1551307.0): 0.0, 
        ('de832cc1dc9bfecd21edbead3cfd08f97cc418e0b2cf7abddf7a06672e86737a', 1551465.0): 1.0,
        ...

    """
    target_df = pd.read_csv(target_file_path)

    target_df = target_df.sort_values(by=['x_user_id', 'exercise_id'])

    target_df['next_qualified'] = target_df.groupby('x_user_id')['qualified'].shift(-1)

    target_df['key'] = list(zip(target_df['x_user_id'], target_df['exercise_id']))

    target_df = target_df.dropna(subset=['next_qualified'])

    target_dict = dict(zip(target_df['key'], target_df['next_qualified']))

    return target_dict

def get_indices_for_users(users):
    indices = []
    for idx in range(len(dataset)):
        user_idx, _ = dataset._get_indices(idx)
        user_id = list(dataset.user_problem_map.keys())[user_idx]
        if user_id in users:
            indices.append(idx)
    return indices

def load_and_sample_dataset(config, question_model, device, cross):
    target_file_path = config['targets']
    target_dict = load_targets(target_file_path)

    
    
    dataset = SkillDataset(
        config['exercises'],
        config['submissions'],
        config['questions'],
        target_dict,
        question_model,
        device=device,
        debug=False
    )
    
    unique_users = list(set(dataset.code_df['x_user_id']))
    num_users = len(unique_users)
    
    if cross == False:
        train_users, temp_users = train_test_split(unique_users, test_size=0.2, shuffle = False)
        val_users, test_users = train_test_split(temp_users, test_size=0.5, shuffle = False)
        train_indices = get_indices_for_users(train_users)
        val_indices = get_indices_for_users(val_users)
        test_indices = get_indices_for_users(test_users)
    
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        test_dataset = Subset(dataset, test_indices)
        return train_dataset, val_dataset, test_dataset
    else:
        train_users, temp_users = train_test_split(unique_users, test_size=0.2, shuffle = False)
        val_users, test_users = train_test_split(temp_users, test_size=0.5, shuffle = False)
        train_indices = get_indices_for_users(train_users)
        val_indices = get_indices_for_users(val_users)
        test_indices = get_indices_for_users(test_users)
        
        train_indices, test_indices = train_test_split(indices, test_size=0.2, shuffle = False)
        train_dataset = Subset(dataset, train_indices)
        test_dataset = Subset(dataset, test_indices)
        return train_dataset, test_dataset


def prepare_and_split_data_loaders(dataset_configs, question_model, batch_size=4, shuffle=False, device='cuda', cross=False, dataset_key=None):
    
    
    if isinstance(dataset_key, str):
        dataset_key = [dataset_key] 
    
    train_datasets = []
    val_datasets = []
    test_datasets = []

    
    if cross == False:
        for key in dataset_key:
            if key not in dataset_configs:
                raise ValueError(f"dataset_key '{key}'가 dataset_configs에 없습니다.")
            
            single_config = dataset_configs[key]
            train_dataset, val_dataset, test_dataset = load_and_sample_dataset(single_config, question_model, device, cross)
            
            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)
            test_datasets.append(test_dataset)
        
        combined_train_dataset = ConcatDataset(train_datasets)
        combined_val_dataset = ConcatDataset(val_datasets)
        combined_test_dataset = ConcatDataset(test_datasets)
        
        
        train_loader = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
        val_loader = DataLoader(combined_val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(combined_test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        return train_loader, val_loader, test_loader
    
    else:
        last_key = dataset_key[-1]  
        for key in dataset_key[:-1]: 
            if key not in dataset_configs:
                raise ValueError(f"dataset_key '{key}'is not in dataset.")
            
            train_dataset, val_dataset = load_and_sample_dataset(dataset_configs[key], question_model, device, cross)
            print(f"Loaded train dataset '{key}' size: {len(train_dataset)}") 
            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)

        test_dataset, val_dataset = load_and_sample_dataset(dataset_configs[last_key], question_model, device, cross)
        test_datasets.append(test_dataset)
        test_datasets.append(val_dataset)
        
        combined_train_dataset = ConcatDataset(train_datasets)
        combined_val_dataset = ConcatDataset(val_datasets)
        combined_test_dataset = ConcatDataset(test_datasets)


        train_loader = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
        val_loader = DataLoader(combined_val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(combined_test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        
        print(f"Train dataset size: {len(train_loader)}")
        print(f"Validation dataset size: {len(val_loader)}")
        print(f"Test dataset size: {len(test_dataset)}")
        
        return train_loader, val_loader, test_loader




    


def debug():

    question_tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5p-220m")
    question_tokenizer.add_special_tokens({'additional_special_tokens': ['[SKILL]', '[TARGET SKILL]', '[QUESTION]', '[TARGET]', '[CODE]', '[TEXT]']})
    question_model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5p-220m")
    question_model.resize_token_embeddings(len(question_tokenizer))
    dataset_configs = {
            "18873": {
                "exercises": "/home/Transformer/data/18873/18873_pre_dropone.csv",
                "submissions": "/home/Transformer/data/18873/18873_exercises_skill.csv",
                "questions": "/home/Transformer/data/18873/18873_helpcenter_log_skill.csv",
                "targets": "/home/Transformer/data/18873/18873_final_scores.csv"
            },
            "18818": {
                "exercises": "/home/Transformer/data/18818/18818_pre_dropone.csv",
                "submissions": "/home/Transformer/data/18818/18818_exercises_skill.csv",
                "questions": "/home/Transformer/data/18818/18818_helpcenter_log_skill.csv",
                "targets": "/home/Transformer/data/18818/18818_final_scores.csv"
            },
            "18945": {
                "exercises": "/home/Transformer/data/18945/18945_pre_dropone.csv",
                "submissions": "/home/Transformer/data/18945/18945_exercises_skill.csv",
                "questions": "/home/Transformer/data/18945/18945_helpcenter_log_skill.csv",
                "targets": "/home/Transformer/data/18945/18945_final_scores.csv"
            }, 
            "18888": {
                "exercises": "/home/Transformer/data/18888/18888_pre_dropone.csv",
                "submissions": "/home/Transformer/data/18888/18888_exercises_skill.csv",
                "questions": "/home/Transformer/data/18888/18888_helpcenter_log_skill.csv",
                "targets": "/home/Transformer/data/18888/18888_final_scores.csv"
            }
        }
    
    train_loader, val_loader, test_loader = prepare_and_split_data_loaders(
        dataset_configs,
        question_model, 
        batch_size=1,
        shuffle=False,
        device='cuda',
        cross=False,  
        # dataset_key=['18818', '18945', '18873', '18888'] 
        dataset_key=['18818'] 
        # dataset_key = ['18945', '18873']
    )


    i = 0
    for batch_idx, (embeddings, targets, anchor_embeddings, positive_embeddings, negative_embeddings, question_positive_emb, question_negative_emb, codet5_input_ids, codet5_attention_mask, codet5_targets
        ) in enumerate(train_loader):
            print(f"Batch {batch_idx + 1}:")
            print(f"Embeddings Shape: {embeddings.shape}")
            # print(f"Embeddings: {embeddings}")
            print(f"inputs type: {embeddings.dtype}")
            print(f"Targets: {targets}")
            print(f"Targets shape: {targets.shape}")
            print(f"Anchor Embeddings Shape: {anchor_embeddings.shape}")
            print(f"Anchor Embeddings Type: {anchor_embeddings.dtype}")
            print(f"Positive Embeddings Shape: {positive_embeddings.shape}")
            print(f"Negative Embeddings Shape: {negative_embeddings.shape}")
            print(f"Question Positive Embeddings Shape: {question_positive_emb.shape}")
            print(f"Question Negative Embeddings Shape: {question_negative_emb.shape}")
            print(f"codet5_input_ids: {codet5_input_ids.shape}")
            print(f"codet5_input_ids type: {codet5_input_ids.dtype}")
            print(f"codet5_attention_mask: {codet5_attention_mask.shape}")
            print(f"codet5_targets: {codet5_targets.shape}")
            codet5_loss = question_model(input_ids=codet5_input_ids.long(), attention_mask=codet5_attention_mask, labels=codet5_targets.long()).loss
            print(f"Codet5 Loss: {codet5_loss}")
            print(f"Codet5 Loss shape: {codet5_loss.shape}")
            print(f"Codet5 Loss type: {codet5_loss.dtype}")    
            print()
            logging.info(f"\n\n=========================================================")
            logging.info(f"Batch {batch_idx + 1}:")
            logging.info(f"Embeddings Shape: {embeddings.shape}")
            logging.info(f"Targets: {targets}")
            logging.info(f"Anchor Embeddings Shape: {anchor_embeddings.shape}")
            logging.info(f"Positive Embeddings Shape: {positive_embeddings.shape}")
            logging.info(f"Negative Embeddings Shape: {negative_embeddings.shape}")
            logging.info(f"Question Positive Embeddings Shape: {question_positive_emb.shape}")
            logging.info(f"Question Negative Embeddings Shape: {question_negative_emb.shape}")
            logging.info(f"codet5_input_ids: {codet5_input_ids.shape}")
            logging.info(f"codet5_attention_mask: {codet5_attention_mask.shape}")
            logging.info(f"codet5_targets: {codet5_targets.shape}")
            codet5_loss = question_model(input_ids=codet5_input_ids.long(), attention_mask=codet5_attention_mask, labels=codet5_targets.long()).loss
            logging.info(f"Codet5 Loss: {codet5_loss}")
            logging.info(f"Codet5 Loss: {codet5_loss}")
            logging.info(f"Codet5 Loss shape: {codet5_loss.shape}")
            logging.info(f"Codet5 Loss type: {codet5_loss.dtype}") 
            logging.info(f"\n\n=========================================================")
            print()
            i += 1
            if i == 100:
                break

if __name__ == "__main__":
    debug()
