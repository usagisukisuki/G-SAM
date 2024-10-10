from models.segment_anything.build_sam import sam_model_registry
#from models.mobile_sam.build_sam import mobilesam_model_registry
#from models.mobile_sam_adaptformer.build_sam import mobilesam_adaptformer_model_registry
from models.segment_anything_lora.build_sam import sam_lora_model_registry
from models.segment_anything_convlora.build_sam import sam_convlora_model_registry
from models.segment_anything_adaptformer.build_sam import sam_adaptformer_model_registry
from models.segment_anything_samus.build_sam_us import samus_model_registry
from models.generalized_segment_anything.build_sam import gsam_model_registry

def get_model(modelname="SAM", args=None):
    if modelname == "SAM":
        model = sam_model_registry['vit_b'](args=args, checkpoint=args.sam_ckpt)
    #elif modelname == "MobileSAM":
    #    model = mobilesam_model_registry['vit_t'](args=args, checkpoint=args.sam_ckpt)
    #elif modelname == "MobileSAM_AdaptFormer":
    #    model = mobilesam_adaptformer_model_registry['vit_t'](args=args, checkpoint=args.sam_ckpt)
    elif modelname == "SAM_LoRA":
        model = sam_lora_model_registry['vit_b'](args=args, checkpoint=args.sam_ckpt)
    elif modelname == "SAM_ConvLoRA":
        model = sam_convlora_model_registry['vit_b'](args=args, checkpoint=args.sam_ckpt)
    elif modelname == "SAM_AdaptFormer":
        model = sam_adaptformer_model_registry['vit_b'](args=args, checkpoint=args.sam_ckpt)
    elif modelname == "SAMUS":
        model = samus_model_registry['vit_b'](args=args, checkpoint=args.sam_ckpt)
    elif modelname == "GSAM":
        model = gsam_model_registry['vit_b'](args=args, checkpoint=args.sam_ckpt)
    else:
        raise RuntimeError("Could not find the model:", modelname)
    return model
