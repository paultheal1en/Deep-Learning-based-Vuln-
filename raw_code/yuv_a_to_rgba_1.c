static void yuv_a_to_rgba(const uint8_t *ycbcr, const uint8_t *alpha, uint32_t *rgba, int num_values)

{

    const uint8_t *cm = ff_crop_tab + MAX_NEG_CROP;

    uint8_t r, g, b;

    int i, y, cb, cr;

    int r_add, g_add, b_add;



    for (i = num_values; i > 0; i--) {

        y = *ycbcr++;

        cr = *ycbcr++;

        cb = *ycbcr++;

        YUV_TO_RGB1_CCIR(cb, cr);

        YUV_TO_RGB2_CCIR(r, g, b, y);

        *rgba++ = (*alpha++ << 24) | (r << 16) | (g << 8) | b;

    }

}
