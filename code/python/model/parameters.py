ml_pythonVersion = '3.5'
ml_runtimeVersion = '1.10'
ml_region = 'europe-west1'

ml_typology = {
				'preprocess': { 
					'ml_scaleTier': 'CUSTOM',
                    'ml_masterType': 'large_model'
                    },
				'train': {
					'ml_scaleTier': 'CUSTOM',
	   				'ml_masterType': 'standard_gpu'
	   				},
				'test': { 
					'ml_scaleTier': 'CUSTOM',
	   				'ml_masterType': 'standard_gpu'
	   				}
			}

bucket_path = 'IA'

local_dir_path = '/tmp'

dataset_id = "myDataset"

df = "myDataFrame"

query = """SELECT * 
        FROM (SELECT *, ROW_NUMBER() OVER(PARTITION BY person_id ORDER BY timestamp DESC) AS rn
              FROM `{project_account}.visits.audience_*`)
        WHERE rn<50"""

regex = {'train': '[0-9][2_9]$',
		 'test': '0[0-1]$'}

df_preprocessed = 'myDataFramePreProcessed'

key = 'person_id'

event_timestamp = 'timestamp'

dummy = ['env_channel', 'user_logged']

embedding = ['env_template', 'page_cat1', 'page_cat2', 'product_name', 'utm_medium' ]

numerical = ['product_unitprice', 'product_discount']

target = ('order_products_number', 1)

emb_dim = 20

max_len = 50

epochs = 2 # 64

batch_size = 128 # 8192

model_name = "myModel"
