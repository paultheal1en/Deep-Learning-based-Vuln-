static void png_filter_row(DSPContext *dsp, uint8_t *dst, int filter_type,

                           uint8_t *src, uint8_t *top, int size, int bpp)

{

    int i;



    switch(filter_type) {

    case PNG_FILTER_VALUE_NONE:

        memcpy(dst, src, size);

        break;

    case PNG_FILTER_VALUE_SUB:

        dsp->diff_bytes(dst, src, src-bpp, size);

        memcpy(dst, src, bpp);

        break;

    case PNG_FILTER_VALUE_UP:

        dsp->diff_bytes(dst, src, top, size);

        break;

    case PNG_FILTER_VALUE_AVG:

        for(i = 0; i < bpp; i++)

            dst[i] = src[i] - (top[i] >> 1);

        for(; i < size; i++)

            dst[i] = src[i] - ((src[i-bpp] + top[i]) >> 1);

        break;

    case PNG_FILTER_VALUE_PAETH:

        for(i = 0; i < bpp; i++)

            dst[i] = src[i] - top[i];

        sub_png_paeth_prediction(dst+i, src+i, top+i, size-i, bpp);

        break;

    }

}
