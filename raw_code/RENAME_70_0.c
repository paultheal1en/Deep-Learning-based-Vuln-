static void RENAME(resample_one)(DELEM *dst, const DELEM *src,

                                 int dst_size, int64_t index2, int64_t incr)

{

    int dst_index;



    for (dst_index = 0; dst_index < dst_size; dst_index++) {

        dst[dst_index] = src[index2 >> 32];

        index2 += incr;

    }

}
