using FaceONNX.Properties; // Added for resources
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using UMapx.Core; // For normalization
using UMapx.Imaging; // For resizing and color conversion

namespace FaceONNX
{
    /// <summary>
    /// Defines face liveness detector using Deep Pixel-wise Binary Supervision (DeepPixBiS).
    /// Model: OULU_Protocol_2_model_0_0.onnx
    /// Reference: https://github.com/ffletcherr/face-recognition-liveness
    /// </summary>
    public class FaceLivenessDetector : IFaceLivenessDetector
    {
        #region Private data

        /// <summary>
        /// Inference session.
        /// </summary>
        private readonly InferenceSession _session;
        private readonly Size _inputSize = new Size(224, 224);
        private readonly float[] _mean = new float[] { 0.485f, 0.456f, 0.406f };
        private readonly float[] _std = new float[] { 0.229f, 0.224f, 0.225f };

        #endregion

        #region Constructor

        /// <summary>
        /// Initializes face liveness detector using embedded resources.
        /// </summary>
        public FaceLivenessDetector()
        {
            // IMPORTANT: Add OULU_Protocol_2_model_0_0.onnx to Resources.resx 
            // with the name 'liveness_oulu_protocol_2'.
            _session = new InferenceSession(Resources.liveness_oulu_protocol_2); // Pass the byte[] resource
        }

        /// <summary>
        /// Initializes face liveness detector using embedded resources.
        /// </summary>
        /// <param name="options">Session options</param>
        public FaceLivenessDetector(SessionOptions options)
        {
            // IMPORTANT: Add OULU_Protocol_2_model_0_0.onnx to Resources.resx 
            // with the name 'liveness_oulu_protocol_2'.
            _session = new InferenceSession((byte[])Resources.liveness_oulu_protocol_2, options); // Explicitly cast resource to byte[]
        }

        #endregion

        #region Methods

        /// <inheritdoc/>
        public float Forward(Bitmap faceImage)
        {
            // Convert Bitmap to RGB float array
            // The Python code converts BGR to RGB, then uses PIL. UMapx.BitmapEx.ToRGB handles Bitmap -> float[][,]
            // Assuming ToRGB outputs channels in RGB order.
            var rgb = faceImage.ToRGB(false);
            return Forward(rgb);
        }

        /// <inheritdoc/>
        public float Forward(float[][,] faceImage)
        {
            if (faceImage.Length != 3)
                throw new ArgumentException("Image must be in RGB terms with 3 channels.");

            // Resize
            var resized = new float[3][,];
            for (int i = 0; i < 3; i++)
            {
                // Using Bilinear interpolation similar to torchvision's default resize
                resized[i] = faceImage[i].Resize(_inputSize.Height, _inputSize.Width, InterpolationMode.Bilinear);
            }

            // Normalize and prepare tensor
            var inputData = new float[1 * 3 * _inputSize.Height * _inputSize.Width];
            for (int c = 0; c < 3; c++)
            {
                for (int h = 0; h < _inputSize.Height; h++)
                {
                    for (int w = 0; w < _inputSize.Width; w++)
                    {
                        int index = c * _inputSize.Height * _inputSize.Width + h * _inputSize.Width + w;
                        // Normalize: (pixel - mean) / std
                        inputData[index] = (resized[c][h, w] / 255f - _mean[c]) / _std[c];
                    }
                }
            }

            var dimensions = new[] { 1, 3, _inputSize.Height, _inputSize.Width };
            var inputTensor = new DenseTensor<float>(inputData, dimensions);

            // Prepare inputs
            var inputs = new List<NamedOnnxValue>()
            {
                NamedOnnxValue.CreateFromTensor("input", inputTensor)
            };

            // Run inference
            using var outputs = _session.Run(inputs);

            // Process outputs
            var outputPixelTensor = outputs.FirstOrDefault(o => o.Name == "output_pixel")?.AsTensor<float>();
            var outputBinaryTensor = outputs.FirstOrDefault(o => o.Name == "output_binary")?.AsTensor<float>();

            if (outputPixelTensor == null || outputBinaryTensor == null)
            {
                throw new Exception("Could not retrieve expected output tensors ('output_pixel', 'output_binary') from the model.");
            }

            // Calculate score as per Python code: (mean(output_pixel) + mean(output_binary)) / 2.0
            float meanPixel = outputPixelTensor.ToArray().Average(); // Use ToArray()
            float meanBinary = outputBinaryTensor.ToArray().Average(); // Use ToArray()

            float livenessScore = (meanPixel + meanBinary) / 2.0f;

            return livenessScore;
        }

        #endregion

        #region IDisposable

        private bool _disposed;

        /// <inheritdoc/>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <inheritdoc/>
        protected virtual void Dispose(bool disposing) // Made virtual
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    _session?.Dispose();
                }
                _disposed = true;
            }
        }

        /// <summary>
        /// Destructor.
        /// </summary>
        ~FaceLivenessDetector()
        {
            Dispose(false);
        }

        #endregion
    }
}