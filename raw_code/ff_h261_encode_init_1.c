void ff_h261_encode_init(MpegEncContext *s){

    static int done = 0;

    

    if (!done) {

        done = 1;

        init_rl(&h261_rl_tcoeff);

    }



    s->min_qcoeff= -127;

    s->max_qcoeff=  127;

    s->y_dc_scale_table=

    s->c_dc_scale_table= ff_mpeg1_dc_scale_table;

}
