static void ff_wmv2_idct_put_c(uint8_t *dest, int line_size, DCTELEM *block)

{

    ff_wmv2_idct_c(block);

    put_pixels_clamped_c(block, dest, line_size);

}
