static int tb_unreliable(AVCodecContext *c)

{

    if (c->time_base.den >= 101L * c->time_base.num ||

        c->time_base.den <    5L * c->time_base.num ||

        // c->codec_tag == AV_RL32("DIVX") ||

        // c->codec_tag == AV_RL32("XVID") ||

        c->codec_id == AV_CODEC_ID_MPEG2VIDEO ||

        c->codec_id == AV_CODEC_ID_H264)

        return 1;

    return 0;

}
