static void SET_TYPE(resample_nearest)(void *dst0, int dst_index, const void *src0, int index)

{

    FELEM *dst = dst0;

    const FELEM *src = src0;

    dst[dst_index] = src[index];

}
