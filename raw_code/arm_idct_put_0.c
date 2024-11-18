static void arm_idct_put(UINT8 *dest, int line_size, DCTELEM *block)

{

    j_rev_dct_ARM (block);

    put_pixels_clamped(block, dest, line_size);

}
