import torch
import os


def SaveModelTorchscript(model, example_input, model_name, torchscript_folder_path, epoch):
    """ Function that takes a trained model, an example input, and path to the trained model checkpoint
            (a) loads the trained model
            (b) sets it to evaluation model
            (c) compiles and saves a trained model as a TorchScript for efficient inference
        This traced model can be used for inference without the need for the original model's sourcecode or training checkpoint

    Args:
        model (): trained model
        example_input (torch.tensor): with this example input tensor, PyTorch can execute the model oncewith this input abd capture the tensor shapes
                                        at each stage of the computation. This information is then used to construct the computation graph in TorchScript.
        model_name (str): with this name, traced model will be saved
        torchscript_folder_path (str): path where the trained model is saved as .pth file
        epoch (int): current epoch number
    """
    # Load the trained model
    #model.load_state_dict(torch.load(trained_model_path))
    model.eval()

    # Trace the model using TorchScript
    traced_model = torch.jit.trace(model, example_input)
    model_name = model_name
    model_torchscript_save_path = os.path.join(torchscript_folder_path, f'{model_name}_epoch_{epoch}.pt')
    # Save the TorchScript model to a file
    torch.jit.save(traced_model, model_torchscript_save_path)
