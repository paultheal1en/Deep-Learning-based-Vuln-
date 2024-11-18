void ff_add_pixels_clamped_c(const DCTELEM *block, uint8_t *restrict pixels,

                             int line_size)

{

    int i;

    uint8_t *cm = ff_cropTbl + MAX_NEG_CROP;



    /* read the pixels */

    for(i=0;i<8;i++) {

        pixels[0] = cm[pixels[0] + block[0]];

        pixels[1] = cm[pixels[1] + block[1]];

        pixels[2] = cm[pixels[2] + block[2]];

        pixels[3] = cm[pixels[3] + block[3]];

        pixels[4] = cm[pixels[4] + block[4]];

        pixels[5] = cm[pixels[5] + block[5]];

        pixels[6] = cm[pixels[6] + block[6]];

        pixels[7] = cm[pixels[7] + block[7]];

        pixels += line_size;

        block += 8;

    }

}
