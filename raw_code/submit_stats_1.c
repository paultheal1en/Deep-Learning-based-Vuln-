static int submit_stats(AVCodecContext *avctx)

{

#ifdef TH_ENCCTL_2PASS_IN

    TheoraContext *h = avctx->priv_data;

    int bytes;


        if (!avctx->stats_in) {

            av_log(avctx, AV_LOG_ERROR, "No statsfile for second pass\n");

            return AVERROR(EINVAL);


        h->stats_size = strlen(avctx->stats_in) * 3/4;

        h->stats      = av_malloc(h->stats_size);





        h->stats_size = av_base64_decode(h->stats, avctx->stats_in, h->stats_size);


    while (h->stats_size - h->stats_offset > 0) {

        bytes = th_encode_ctl(h->t_state, TH_ENCCTL_2PASS_IN,

                              h->stats + h->stats_offset,

                              h->stats_size - h->stats_offset);

        if (bytes < 0) {

            av_log(avctx, AV_LOG_ERROR, "Error submitting stats\n");

            return AVERROR_EXTERNAL;


        if (!bytes)

            return 0;

        h->stats_offset += bytes;


    return 0;

#else

    av_log(avctx, AV_LOG_ERROR, "libtheora too old to support 2pass\n");

    return AVERROR(ENOSUP);

#endif
