using System;
using System.Drawing;

namespace FaceONNX
{
    /// <summary>
    /// Defines face liveness detector interface.
    /// </summary>
    public interface IFaceLivenessDetector : IDisposable
    {
        #region Interface

        /// <summary>
        /// Returns the liveness score for the given face image.
        /// A higher score indicates a higher probability of being a live face.
        /// </summary>
        /// <param name="faceImage">Bitmap of the face</param>
        /// <returns>Liveness score (float)</returns>
        float Forward(Bitmap faceImage);

        /// <summary>
        /// Returns the liveness score for the given face image.
        /// A higher score indicates a higher probability of being a live face.
        /// </summary>
        /// <param name="faceImage">Face image in RGB terms (float[3][,])</param>
        /// <returns>Liveness score (float)</returns>
        float Forward(float[][,] faceImage);

        #endregion
    }
}