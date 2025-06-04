import argparse
import torch
import torch.nn as nn
import coremltools as ct
import numpy as np
import os
import sys

# Assuming the script is in the root of the project,
# these imports should work directly.
from ASD import ASD as ASDTrainWrapper
from loss import lossAV, lossV


# --- CoreML Wrapper Definition ---
class CoreMLWrapper(nn.Module):
    """
    A wrapper around the ASD_Model and relevant FC layers from loss modules
    for CoreML export.
    """

    def __init__(self, asd_model_instance, loss_av_fc_instance, loss_v_fc_instance):
        super().__init__()
        self.asd_model = asd_model_instance
        self.fc_av = loss_av_fc_instance
        self.fc_v = loss_v_fc_instance

    def forward(self, audio_feature, visual_feature):
        audio_embed = self.asd_model.forward_audio_frontend(audio_feature)
        visual_embed = self.asd_model.forward_visual_frontend(visual_feature)
        outs_av_embed = self.asd_model.forward_audio_visual_backend(audio_embed, visual_embed)
        outs_v_embed = self.asd_model.forward_visual_backend(visual_embed)
        scores_av = self.fc_av(outs_av_embed)
        scores_v = self.fc_v(outs_v_embed)
        return scores_av, scores_v


def convert_to_coreml(pytorch_model_path,
                      output_coreml_path,
                      default_visual_frames=25,
                      max_visual_frames=250,
                      num_mfcc_features=13,
                      visual_frame_height=112,
                      visual_frame_width=112,
                      minimum_deployment_target=ct.target.iOS15):
    """
    Loads the PyTorch ASD model, wraps it, and converts it to CoreML format.
    """
    print(f"Loading PyTorch model from: {pytorch_model_path}")
    asd_train_wrapper_instance = ASDTrainWrapper(lr=0.001, lrDecay=0.95)

    asd_train_wrapper_instance.loadParameters(pytorch_model_path)
    print("PyTorch model parameters loaded.")

    asd_model_pytorch = asd_train_wrapper_instance.model.cpu()
    loss_av_fc_pytorch = asd_train_wrapper_instance.lossAV.FC.cpu()
    loss_v_fc_pytorch = asd_train_wrapper_instance.lossV.FC.cpu()

    asd_model_pytorch.eval()
    loss_av_fc_pytorch.eval()
    loss_v_fc_pytorch.eval()

    coreml_model_wrapper = CoreMLWrapper(asd_model_pytorch, loss_av_fc_pytorch, loss_v_fc_pytorch)
    coreml_model_wrapper.eval()
    print("CoreML wrapper created.")

    default_audio_frames = default_visual_frames * 4
    max_audio_frames = max_visual_frames * 4

    example_audio_input = torch.randn(
        1, default_audio_frames, num_mfcc_features,
        dtype=torch.float32
    ).cpu()
    example_visual_input = torch.randn(
        1, default_visual_frames, visual_frame_height, visual_frame_width,
        dtype=torch.float32
    ).cpu()
    print(f"Example audio input shape: {example_audio_input.shape}")
    print(f"Example visual input shape: {example_visual_input.shape}")

    try:
        traced_model = torch.jit.trace(coreml_model_wrapper, (example_audio_input, example_visual_input))
        print("Model traced successfully.")
    except Exception as e:
        print(f"Error during model tracing: {e}")
        return

    print(f"Using RangeDim upper bound for visual: {max_visual_frames}, for audio: {max_audio_frames}")
    input_audio_shape = ct.TensorType(
        name="audio_feature",  # Original name before renaming
        shape=(
        1, ct.RangeDim(lower_bound=1, upper_bound=max_audio_frames, default=default_audio_frames), num_mfcc_features),
        dtype=np.float32
    )
    input_visual_shape = ct.TensorType(
        name="visual_feature",  # Original name before renaming
        shape=(1, ct.RangeDim(lower_bound=1, upper_bound=max_visual_frames, default=default_visual_frames),
               visual_frame_height, visual_frame_width),
        dtype=np.float32
    )
    inputs = [input_audio_shape, input_visual_shape]

    # Define target output names for clarity, these might be used if ct.convert respects them,
    # or we use them when setting metadata.
    target_av_output_name = "audioVisualScoresOutput"
    target_v_output_name = "visualOnlyScoresOutput"

    # For the `outputs` parameter in `ct.convert`, it's often better to let CoreML
    # infer the output structure and names initially, then rename them in the spec.
    # If you provide ct.TensorType here, their names should match what the traced model produces.
    # The traced_model will produce outputs probably named like 'output_0', 'output_1' or similar.
    # Let's omit `outputs` from ct.convert and rename from spec later.

    print("Starting CoreML conversion...")
    try:
        mlmodel = ct.convert(
            traced_model,
            inputs=inputs,
            # outputs=outputs, # Omit for now, will rename from spec
            convert_to="mlprogram",
            minimum_deployment_target=minimum_deployment_target,
            compute_units=ct.ComputeUnit.ALL
        )
        print("CoreML conversion successful.")
    except Exception as e:
        print(f"Error during CoreML conversion: {e}")
        return

    # --- Modify metadata on the spec ---
    spec = mlmodel.get_spec()

    # Define the new names
    new_audio_input_name = "audioFeatureInput"
    new_visual_input_name = "visualFeatureInput"
    new_av_output_name = target_av_output_name  # Use our target name
    new_v_output_name = target_v_output_name  # Use our target name

    # Get the original input/output names from the spec as CoreML sees them
    # These names are what coremltools assigned *before* any renaming.
    original_audio_input_name = spec.description.input[0].name
    original_visual_input_name = spec.description.input[1].name
    original_av_output_name = spec.description.output[0].name
    original_v_output_name = spec.description.output[1].name

    print(f"Original CoreML input names: {[inp.name for inp in spec.description.input]}")
    print(f"Original CoreML output names: {[out.name for out in spec.description.output]}")

    # Rename features in the spec
    ct.utils.rename_feature(spec, original_audio_input_name, new_audio_input_name)
    ct.utils.rename_feature(spec, original_visual_input_name, new_visual_input_name)
    ct.utils.rename_feature(spec, original_av_output_name, new_av_output_name)
    ct.utils.rename_feature(spec, original_v_output_name, new_v_output_name)
    print(f"Renamed inputs to: {new_audio_input_name}, {new_visual_input_name}")
    print(f"Renamed outputs to: {new_av_output_name}, {new_v_output_name}")

    # Create the updated model from the modified spec
    # This step is crucial: mlmodel_updated is the one with the renamed features.
    mlmodel_updated = ct.models.MLModel(spec, weights_dir=mlmodel.weights_dir)

    # Now, set metadata on mlmodel_updated using the NEW names
    mlmodel_updated.author = "Converted using script"
    mlmodel_updated.license = "N/A"
    mlmodel_updated.short_description = "Active Speaker Detection (ASD) model. Outputs AV and V scores."

    mlmodel_updated.input_description[
        new_audio_input_name] = f"Audio MFCC features (1, T_audio, {num_mfcc_features}), T_audio is variable up to {max_audio_frames}."
    mlmodel_updated.input_description[
        new_visual_input_name] = f"Visual face crops (1, T_visual, {visual_frame_height}, {visual_frame_width}), T_visual is variable up to {max_visual_frames}."

    mlmodel_updated.output_description[new_av_output_name] = "Scores from Audio-Visual stream (T_common, 2)."
    mlmodel_updated.output_description[new_v_output_name] = "Scores from Visual-only stream (T_common, 2)."
    print("Metadata set on the updated model.")

    # --- Save the CoreML model ---
    try:
        output_dir = os.path.dirname(output_coreml_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        mlmodel_updated.save(output_coreml_path)  # Save the updated model
        print(f"CoreML model saved to: {output_coreml_path}")
    except Exception as e:
        print(f"Error saving CoreML model: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert PyTorch ASD model to CoreML")
    parser.add_argument(
        '--pytorch_model_path', type=str, required=True,
        help="Path to the PyTorch ASD model weights (.model or .pt file)"
    )
    parser.add_argument(
        '--output_coreml_path', type=str, required=True,
        help="Path to save the output CoreML model (e.g., coreml_models/ASD_Model.mlpackage)"
    )
    parser.add_argument(
        '--default_visual_frames', type=int, default=25,
        help="Default number of visual frames for tracing (e.g., 1 sec @ 25 FPS)"
    )
    parser.add_argument(
        '--max_visual_frames', type=int, default=250,
        help="Maximum number of visual frames the CoreML model should support."
    )
    parser.add_argument(
        '--num_mfcc_features', type=int, default=13,
        help="Number of MFCC features for audio input"
    )
    parser.add_argument(
        '--visual_frame_height', type=int, default=112,
        help="Height of the visual frame input"
    )
    parser.add_argument(
        '--visual_frame_width', type=int, default=112,
        help="Width of the visual frame input"
    )
    parser.add_argument(
        '--min_deployment_target', type=str, default='iOS15',
        choices=['iOS13', 'iOS14', 'iOS15', 'iOS16', 'iOS17',
                 'macOS10_15', 'macOS11', 'macOS12', 'macOS13', 'macOS14'],
        help="Minimum deployment target (e.g., iOS15, macOS12)"
    )

    args = parser.parse_args()

    target_mapping = {
        'iOS13': ct.target.iOS13, 'iOS14': ct.target.iOS14, 'iOS15': ct.target.iOS15,
        'iOS16': ct.target.iOS16, 'iOS17': ct.target.iOS17,
        'macOS10_15': ct.target.macOS10_15, 'macOS11': ct.target.macOS11,
        'macOS12': ct.target.macOS12, 'macOS13': ct.target.macOS13, 'macOS14': ct.target.macOS14,
    }
    deployment_target = target_mapping.get(args.min_deployment_target, ct.target.iOS15)

    convert_to_coreml(
        pytorch_model_path=args.pytorch_model_path,
        output_coreml_path=args.output_coreml_path,
        default_visual_frames=args.default_visual_frames,
        max_visual_frames=args.max_visual_frames,
        num_mfcc_features=args.num_mfcc_features,
        visual_frame_height=args.visual_frame_height,
        visual_frame_width=args.visual_frame_width,
        minimum_deployment_target=deployment_target
    )
    print("Script finished.")