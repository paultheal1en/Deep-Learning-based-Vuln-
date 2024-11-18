static void RENAME(chrRangeToJpeg)(int16_t *dst, int width)

{

    int i;

    for (i = 0; i < width; i++) {

        dst[i     ] = (FFMIN(dst[i     ],30775)*4663 - 9289992)>>12; //-264

        dst[i+VOFW] = (FFMIN(dst[i+VOFW],30775)*4663 - 9289992)>>12; //-264

    }

}
