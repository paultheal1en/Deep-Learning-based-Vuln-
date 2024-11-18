static int mov_read_pasp(MOVContext *c, ByteIOContext *pb, MOVAtom atom)

{

    const int num = get_be32(pb);

    const int den = get_be32(pb);

    AVStream * const st = c->fc->streams[c->fc->nb_streams-1];

    if (den != 0) {

        if ((st->sample_aspect_ratio.den != 1 || st->sample_aspect_ratio.num) && // default

            (den != st->sample_aspect_ratio.den || num != st->sample_aspect_ratio.num))

            av_log(c->fc, AV_LOG_WARNING,

                   "sample aspect ratio already set to %d:%d, overriding by 'pasp' atom\n",

                   st->sample_aspect_ratio.num, st->sample_aspect_ratio.den);

        st->sample_aspect_ratio.num = num;

        st->sample_aspect_ratio.den = den;

    }

    return 0;

}
