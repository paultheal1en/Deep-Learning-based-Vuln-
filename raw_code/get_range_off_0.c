static int get_range_off(int *off, int *y_rng, int *uv_rng,

                         enum AVColorRange rng, int depth)

{

    switch (rng) {

    case AVCOL_RANGE_MPEG:

        *off = 16 << (depth - 8);

        *y_rng = 219 << (depth - 8);

        *uv_rng = 224 << (depth - 8);

        break;

    case AVCOL_RANGE_JPEG:

        *off = 0;

        *y_rng = *uv_rng = (256 << (depth - 8)) - 1;

        break;

    default:

        return AVERROR(EINVAL);

    }



    return 0;

}
