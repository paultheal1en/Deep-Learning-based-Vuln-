static void RENAME(chrRangeFromJpeg)(int16_t *dst, int width)

{

    int i;

    for (i = 0; i < width; i++) {

        dst[i     ] = (dst[i     ]*1799 + 4081085)>>11; //1469

        dst[i+VOFW] = (dst[i+VOFW]*1799 + 4081085)>>11; //1469

    }

}
