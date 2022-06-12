# Yahoo-CHQ-Summ


# CHQ-Summ: A Dataset for Consumer Healthcare Question Summarization




## Pre-requistite
Download the following Transformers repo 
 https://github.com/huggingface/transformers/tree/v4.1.1/examples/seq2seq

Place the content of Yahoo-CHQ-Summ into `transformers/examples/seq2seq`

The code requires **Python 3** and please install the Python dependencies with the command:
```bash
pip install -r requirements.txt
```

## Data Preparation
1) Download the CHQ-Summ  dataset from [OSF repository](https://doi.org/10.17605/OSF.IO/X5RGM) and place train.json/val.json/test.json in `data/dataset/CHQ-Summ` directory
2) Download the Yahoo L6 dataset from [here](https://webscope.sandbox.yahoo.com/catalog.php?datatype=l&did=11) and place the xml file in `data/dataset/Yahoo-L6`
3) Run the following code to extract the `subject` and `content` from the Yahoo L6 dataset.
    ```
    python read_yahoo_data.py --Yahoo_data_path /path/to/the/Yahoo-L6/dataset  --CHQ_summ_path data/dataset/CHQ-Summ
    ```

### Running the code 
1.  Update the `CURRENT_DIR` path in `run_4_chq_sum.sh`

2. Train and evalaute the models on CHQ-Summ dataset.

    ```
   bash run_4_chq_sum.sh

    ```
