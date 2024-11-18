AVResampleContext *av_resample_init(int out_rate, int in_rate, int filter_size, int phase_shift, int linear, double cutoff){

    AVResampleContext *c= av_mallocz(sizeof(AVResampleContext));

    double factor= FFMIN(out_rate * cutoff / in_rate, 1.0);

    int phase_count= 1<<phase_shift;

    

    c->phase_shift= phase_shift;

    c->phase_mask= phase_count-1;

    c->linear= linear;



    c->filter_length= FFMAX(ceil(filter_size/factor), 1);

    c->filter_bank= av_mallocz(c->filter_length*(phase_count+1)*sizeof(FELEM));

    av_build_filter(c->filter_bank, factor, c->filter_length, phase_count, 1<<FILTER_SHIFT, 1);

    memcpy(&c->filter_bank[c->filter_length*phase_count+1], c->filter_bank, (c->filter_length-1)*sizeof(FELEM));

    c->filter_bank[c->filter_length*phase_count]= c->filter_bank[c->filter_length - 1];



    c->src_incr= out_rate;

    c->ideal_dst_incr= c->dst_incr= in_rate * phase_count;

    c->index= -phase_count*((c->filter_length-1)/2);



    return c;

}
