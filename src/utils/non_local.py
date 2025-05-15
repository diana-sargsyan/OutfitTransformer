import cv2
import numpy as np

def remove_wrinkles(image, strength=10):
    """
    Remove wrinkles from fabric using Non-Local Means denoising.
    This approach preserves fabric texture better than bilateral filtering.
    
    Parameters:
        image: Input BGR image
        strength: Wrinkle removal strength (5-30 recommended)
    
    Returns:
        Processed image with reduced wrinkles
    """
    # Convert strength parameter to NLM parameters
    h = strength     # Filter strength (higher values = more smoothing)
    search_window = 21  # Search window size
    block_size = 7    # Comparison block size
    
    # Apply Non-Local Means denoising (preserves patterns and texture better)
    result = cv2.fastNlMeansDenoisingColored(
        image, 
        None,
        h=h,              # Luminance filtering strength 
        hColor=h//2,      # Color filtering strength (less aggressive)
        templateWindowSize=block_size,
        searchWindowSize=search_window
    )
    
    # Enhance edges to maintain garment structure
    enhanced = cv2.detailEnhance(result, sigma_s=10, sigma_r=0.15)
    
    # Blend the enhanced edges with the denoised image
    result = cv2.addWeighted(result, 0.5, enhanced, 0.3, 0)
    
    return result



image = cv2.imread('/Users/dinul/Desktop/proj/outfit-transformer/wrinkle.jpg')


# Remove wrinkles
# Lower values (5-10): Light wrinkle reduction
# Medium values (10-20): Moderate smoothing
# Higher values (20-30): Strong wrinkle removal
result = remove_wrinkles(image, strength=20)

# Save or display result
cv2.imwrite('smooth_shirt.jpg', result)