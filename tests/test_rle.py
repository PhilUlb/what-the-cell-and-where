"""Tests for RLE utilities."""

import numpy as np
import pytest

from wtcell.data.rle import rle_encode, rle_decode, validate_rle_string

def test_rle_encode_decode():
    """Test RLE encoding and decoding."""
    # Create a simple test mask
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 1  # 2x2 square in the middle
    
    # Encode
    rle_string = rle_encode(mask)
    
    # Decode
    decoded_mask = rle_decode(rle_string, (4, 4))
    
    # Check that they match
    np.testing.assert_array_equal(mask, decoded_mask)

def test_rle_encode_empty_mask():
    """Test RLE encoding of empty mask."""
    mask = np.zeros((3, 3), dtype=np.uint8)
    rle_string = rle_encode(mask)
    
    # Empty mask should result in empty string
    assert rle_string == ""

def test_rle_decode_empty_string():
    """Test RLE decoding of empty string."""
    mask = rle_decode("", (3, 3))
    
    # Should return zero mask
    expected = np.zeros((3, 3), dtype=np.uint8)
    np.testing.assert_array_equal(mask, expected)

def test_validate_rle_string():
    """Test RLE string validation."""
    # Valid RLE string
    valid_rle = "0 2 5 3"
    assert validate_rle_string(valid_rle) == True
    
    # Invalid RLE string (odd number of values)
    invalid_rle = "0 2 5"
    assert validate_rle_string(invalid_rle) == False
    
    # Empty string is valid
    assert validate_rle_string("") == True
    
    # String with negative values is invalid
    invalid_negative = "0 -2 5 3"
    assert validate_rle_string(invalid_negative) == False

def test_rle_encode_decode_complex():
    """Test RLE encoding/decoding with more complex mask."""
    # Create a more complex mask
    mask = np.array([
        [0, 1, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [1, 0, 1, 0]
    ], dtype=np.uint8)
    
    # Encode and decode
    rle_string = rle_encode(mask)
    decoded_mask = rle_decode(rle_string, (4, 4))
    
    # Check that they match
    np.testing.assert_array_equal(mask, decoded_mask) 