from util import get_entities

def accuracy(predictions, tags):
    def equal(pred_ne, tag_ne):
        return pred_ne
    total, correct = 0, 0
    for prediction, tag in zip(predictions, tags):
        pred_entities = get_entities(prediction)
        tag_entities =  get_entities(tag)
        t_len, p_len, offset = len(tag_entities), len(pred_entities), 0
        for tag_ne in tag_entities:
            for i in range(offset, p_len):
                if tag_ne == pred_entities[i]:
                    correct += 1
                    offset += 1
        total += t_len
        
    return correct / total, total, correct
                        
    
if __name__ == '__main__':
    pass