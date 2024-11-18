static void free_picture(MpegEncContext *s, Picture *pic){

    int i;



    if(pic->data[0] && pic->type!=FF_BUFFER_TYPE_SHARED){

        free_frame_buffer(s, pic);

    }



    av_freep(&pic->mb_var);

    av_freep(&pic->mc_mb_var);

    av_freep(&pic->mb_mean);

    av_freep(&pic->mbskip_table);

    av_freep(&pic->qscale_table);

    av_freep(&pic->mb_type_base);

    av_freep(&pic->dct_coeff);

    av_freep(&pic->pan_scan);

    pic->mb_type= NULL;

    for(i=0; i<2; i++){

        av_freep(&pic->motion_val_base[i]);

        av_freep(&pic->ref_index[i]);

    }



    if(pic->type == FF_BUFFER_TYPE_SHARED){

        for(i=0; i<4; i++){

            pic->base[i]=

            pic->data[i]= NULL;

        }

        pic->type= 0;

    }

}
