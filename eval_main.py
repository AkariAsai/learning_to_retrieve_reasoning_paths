import json

from eval_odqa import ODQAEval
from eval_utils import evaluate, evaluate_w_sp_facts, convert_qa_sp_results_into_hp_eval_format


def main():

    odqa = ODQAEval()

    if odqa.args.sequential_sentence_selector_path is not None:
        tfidf_retrieval_output, selector_output, reader_output, sp_selector_output = odqa.eval()
        if odqa.args.sp_eval is True:
            # eval the performance; based on F1 & EM.
            predictions = convert_qa_sp_results_into_hp_eval_format(
            reader_output, sp_selector_output, odqa.args.db_path)
            results = evaluate_w_sp_facts(
                odqa.args.eval_file_path_sp, predictions, odqa.args.sampled)
        else:
            results = evaluate(odqa.args.eval_file_path, reader_output)
        print(results)
        
    else:
        tfidf_retrieval_output, selector_output, reader_output = odqa.eval()
        # eval the performance; based on F1 & EM. 
        results = evaluate(odqa.args.eval_file_path, reader_output)
    
        print("EM :{0}, F1: {1}".format(results['exact_match'], results['f1']))

    # Save the intermediate results.
    if odqa.args.tfidf_results_save_path is not None:
        print('#### save TFIDF Retrieval results to {}####'.format(
            odqa.args.tfidf_results_save_path))
        with open(odqa.args.tfidf_results_save_path, "w") as writer:
            writer.write(json.dumps(tfidf_retrieval_output, indent=4) + "\n")
    
    if odqa.args.selector_results_save_path is not None:
        print('#### save graph-based Retrieval results to {} ####'.format(
            odqa.args.selector_results_save_path))
        with open(odqa.args.selector_results_save_path, "w") as writer:
            writer.write(json.dumps(selector_output, indent=4) + "\n")

    if odqa.args.reader_results_save_path is not None:
        print('#### save reader results to {} ####'.format(
            odqa.args.reader_results_save_path))
        with open(odqa.args.reader_results_save_path, "w") as writer:
            writer.write(json.dumps(reader_output, indent=4) + "\n")
            
    if odqa.args.sequence_sentence_selector_save_path is not None:
        print("#### save sentence selector results to {} ####".format(
            odqa.args.sequence_sentence_selector_save_path))
        with open(odqa.args.sequence_sentence_selector_save_path, "w") as writer:
            writer.write(json.dumps(sp_selector_output, indent=4) + "\n")


if __name__ == "__main__":
    main()
