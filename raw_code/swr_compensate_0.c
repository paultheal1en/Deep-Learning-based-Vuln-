void swr_compensate(struct SwrContext *s, int sample_delta, int compensation_distance){

    ResampleContext *c= s->resample;

//    sample_delta += (c->ideal_dst_incr - c->dst_incr)*(int64_t)c->compensation_distance / c->ideal_dst_incr;

    c->compensation_distance= compensation_distance;

    c->dst_incr = c->ideal_dst_incr - c->ideal_dst_incr * (int64_t)sample_delta / compensation_distance;

}
