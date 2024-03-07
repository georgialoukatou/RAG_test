import argparse
import structlog
from src.modules.prediction_pipeline import function_qa_with_sources, evaluate_rag_model
from src.modules.dataloader import DataLoader 
from src.utils.label_dicts import labels_queries
from dotenv import load_dotenv
load_dotenv()

logger = structlog.get_logger(__name__)

def main(query, create_retriever_flag):
    url = 'https://classics.mit.edu/Homer/iliad.mb.txt'  # Define the URL here


    data_loader = DataLoader(url, create_docs=create_retriever_flag)


    docs = data_loader.create_or_load_retriever()



    print(docs, len(docs), type(docs))

    # Call prediction pipeline

    result = ""
    result = function_qa_with_sources(query, docs)
    print("main.py result " + str(result))
    
    # Evaluate the RAG model
    if result and query in labels_queries:
            evaluation_metrics = evaluate_rag_model(query, result, labels_queries[query], docs)
            print("Evaluation Metrics:", evaluation_metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RAG-based question answering.')
    parser.add_argument('--query', type=str, help='The query to be answered.')
    parser.add_argument('--create_docs', action='store_true', help='Whether to download Docs or load from existing Docs')
    args = parser.parse_args()

    if args.query.lower() == "testbed":
        # Run specific list of queries for the testbed
        # Replace the following with your specific list of queries
        testbed_queries = labels_queries.keys()  # Example list of queries
        
        # Execute each query
        for query in testbed_queries:
            logger.debug("The query is : " + str(query))
            main(query, args.create_docs)
    else:
        # Run the main function with the provided query
        logger.debug("The query is : " + str(args.query))
        main(args.query, args.create_docs)
