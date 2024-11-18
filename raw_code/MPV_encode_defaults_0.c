static void MPV_encode_defaults(MpegEncContext *s){

    static int done=0;



    MPV_common_defaults(s);



    if(!done){

        int i;

        done=1;



        for(i=-16; i<16; i++){

            default_fcode_tab[i + MAX_MV]= 1;

        }

    }

    s->me.mv_penalty= default_mv_penalty;

    s->fcode_tab= default_fcode_tab;

}
