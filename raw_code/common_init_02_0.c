static av_cold void common_init(MpegEncContext * s)

{

    static int initialized=0;



    switch(s->msmpeg4_version){

    case 1:

    case 2:

        s->y_dc_scale_table=

        s->c_dc_scale_table= ff_mpeg1_dc_scale_table;

        break;

    case 3:

        if(s->workaround_bugs){

            s->y_dc_scale_table= old_ff_y_dc_scale_table;

            s->c_dc_scale_table= wmv1_c_dc_scale_table;

        } else{

            s->y_dc_scale_table= ff_mpeg4_y_dc_scale_table;

            s->c_dc_scale_table= ff_mpeg4_c_dc_scale_table;

        }

        break;

    case 4:

    case 5:

        s->y_dc_scale_table= wmv1_y_dc_scale_table;

        s->c_dc_scale_table= wmv1_c_dc_scale_table;

        break;

#if CONFIG_WMV3_DECODER || CONFIG_VC1_DECODER

    case 6:

        s->y_dc_scale_table= wmv3_dc_scale_table;

        s->c_dc_scale_table= wmv3_dc_scale_table;

        break;

#endif



    }





    if(s->msmpeg4_version>=4){

        ff_init_scantable(s->dsp.idct_permutation, &s->intra_scantable  , wmv1_scantable[1]);

        ff_init_scantable(s->dsp.idct_permutation, &s->intra_h_scantable, wmv1_scantable[2]);

        ff_init_scantable(s->dsp.idct_permutation, &s->intra_v_scantable, wmv1_scantable[3]);

        ff_init_scantable(s->dsp.idct_permutation, &s->inter_scantable  , wmv1_scantable[0]);

    }

    //Note the default tables are set in common_init in mpegvideo.c



    if(!initialized){

        initialized=1;



        init_h263_dc_for_msmpeg4();

    }

}
