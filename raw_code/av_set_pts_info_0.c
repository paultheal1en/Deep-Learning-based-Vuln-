void av_set_pts_info(AVStream *s, int pts_wrap_bits,

                     unsigned int pts_num, unsigned int pts_den)

{

    unsigned int gcd= av_gcd(pts_num, pts_den);

    s->pts_wrap_bits = pts_wrap_bits;

    s->time_base.num = pts_num/gcd;

    s->time_base.den = pts_den/gcd;



    if(gcd>1)

        av_log(NULL, AV_LOG_DEBUG, "st:%d removing common factor %d from timebase\n", s->index, gcd);

}
