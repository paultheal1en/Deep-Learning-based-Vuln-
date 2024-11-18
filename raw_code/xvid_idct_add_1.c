static void xvid_idct_add(uint8_t *dest, ptrdiff_t line_size, int16_t *block)

{

    ff_xvid_idct(block);

    ff_add_pixels_clamped(block, dest, line_size);

}
