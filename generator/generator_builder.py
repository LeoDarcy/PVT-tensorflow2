

#from generator.custom_generator import CustomGenerator
#from generator.cifar_generator import CIFARGenerator
from generator.laval_generator import LavalGenerator
def get_generator(args):
    if args.dataset.strip()[0:5] == 'laval':
        train_generator = LavalGenerator(args, mode="train")
        val_generator = LavalGenerator(args, mode="valid")
        return train_generator, val_generator
    else:
        raise ValueError("{} dataset is not supported!".format(args.dataset))
        print("Error in generator_builder, input arg is not laval please check")
        assert(1==0)
