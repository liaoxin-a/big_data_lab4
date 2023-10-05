import os
import configparser
config = configparser.ConfigParser()
config['default'] = {'SEGMENT_LEN': '5',
                    'NUM_FOLDS'  : '10',
                    'NUM_MODELS' : '2',
                    'BATCH_SIZE' : '24',
                    'LOG_FREQ'   : '1000',
                    'NUM_CLASSES': '1000',
                    'NUM_EPOCHS' : '5',
                    'FEATURES_DIM':'1152',
                    'LEARNING_RATE': '0.00011729283760398037',
                    'WEIGHT_DECAY' : '0.0011412688966608406',
                    'LR_FACTOR'    : '0.1',
                    'LR_PATIENCE'  : '3',
                    'LR_MINIMUM'   : '3e-7',
                    'LR_THRESHOLD' : '1e-3'
                    }

base_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(base_dir,'./config.ini'), 'w') as configfile:
    config.write(configfile)