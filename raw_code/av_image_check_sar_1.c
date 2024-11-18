int av_image_check_sar(unsigned int w, unsigned int h, AVRational sar)

{

    int64_t scaled_dim;



    if (!sar.den)

        return AVERROR(EINVAL);



    if (!sar.num || sar.num == sar.den)

        return 0;



    if (sar.num < sar.den)

        scaled_dim = av_rescale_rnd(w, sar.num, sar.den, AV_ROUND_ZERO);

    else

        scaled_dim = av_rescale_rnd(h, sar.den, sar.num, AV_ROUND_ZERO);



    if (scaled_dim > 0)

        return 0;



    return AVERROR(EINVAL);

}
