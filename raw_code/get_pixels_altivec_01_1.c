static void get_pixels_altivec(int16_t *restrict block, const uint8_t *pixels,

                               ptrdiff_t line_size)

{

    int i;

    vec_u8 perm = vec_lvsl(0, pixels);

    const vec_u8 zero = (const vec_u8)vec_splat_u8(0);



    for (i = 0; i < 8; i++) {

        /* Read potentially unaligned pixels.

         * We're reading 16 pixels, and actually only want 8,

         * but we simply ignore the extras. */

        vec_u8 pixl = vec_ld(0, pixels);

        vec_u8 pixr = vec_ld(7, pixels);

        vec_u8 bytes = vec_perm(pixl, pixr, perm);



        // Convert the bytes into shorts.

        vec_s16 shorts = (vec_s16)vec_mergeh(zero, bytes);



        // Save the data to the block, we assume the block is 16-byte aligned.

        vec_st(shorts, i * 16, (vec_s16 *)block);



        pixels += line_size;

    }

}
