int ff_alloc_picture(MpegEncContext *s, Picture *pic, int shared){

    const int big_mb_num= s->mb_stride*(s->mb_height+1) + 1; //the +1 is needed so memset(,,stride*height) does not sig11

    const int mb_array_size= s->mb_stride*s->mb_height;

    const int b8_array_size= s->b8_stride*s->mb_height*2;

    const int b4_array_size= s->b4_stride*s->mb_height*4;

    int i;

    int r= -1;



    if(shared){

        assert(pic->data[0]);

        assert(pic->type == 0 || pic->type == FF_BUFFER_TYPE_SHARED);

        pic->type= FF_BUFFER_TYPE_SHARED;

    }else{

        assert(!pic->data[0]);



        if (alloc_frame_buffer(s, pic) < 0)

            return -1;



        s->linesize  = pic->linesize[0];

        s->uvlinesize= pic->linesize[1];

    }



    if(pic->qscale_table==NULL){

        if (s->encoding) {

            FF_ALLOCZ_OR_GOTO(s->avctx, pic->mb_var   , mb_array_size * sizeof(int16_t)  , fail)

            FF_ALLOCZ_OR_GOTO(s->avctx, pic->mc_mb_var, mb_array_size * sizeof(int16_t)  , fail)

            FF_ALLOCZ_OR_GOTO(s->avctx, pic->mb_mean  , mb_array_size * sizeof(int8_t )  , fail)

        }



        FF_ALLOCZ_OR_GOTO(s->avctx, pic->mbskip_table , mb_array_size * sizeof(uint8_t)+2, fail) //the +2 is for the slice end check

        FF_ALLOCZ_OR_GOTO(s->avctx, pic->qscale_table , mb_array_size * sizeof(uint8_t)  , fail)

        FF_ALLOCZ_OR_GOTO(s->avctx, pic->mb_type_base , (big_mb_num + s->mb_stride) * sizeof(uint32_t), fail)

        pic->mb_type= pic->mb_type_base + 2*s->mb_stride+1;

        if(s->out_format == FMT_H264){

            for(i=0; i<2; i++){

                FF_ALLOCZ_OR_GOTO(s->avctx, pic->motion_val_base[i], 2 * (b4_array_size+4)  * sizeof(int16_t), fail)

                pic->motion_val[i]= pic->motion_val_base[i]+4;

                FF_ALLOCZ_OR_GOTO(s->avctx, pic->ref_index[i], 4*mb_array_size * sizeof(uint8_t), fail)

            }

            pic->motion_subsample_log2= 2;

        }else if(s->out_format == FMT_H263 || s->encoding || (s->avctx->debug&FF_DEBUG_MV) || (s->avctx->debug_mv)){

            for(i=0; i<2; i++){

                FF_ALLOCZ_OR_GOTO(s->avctx, pic->motion_val_base[i], 2 * (b8_array_size+4) * sizeof(int16_t), fail)

                pic->motion_val[i]= pic->motion_val_base[i]+4;

                FF_ALLOCZ_OR_GOTO(s->avctx, pic->ref_index[i], 4*mb_array_size * sizeof(uint8_t), fail)

            }

            pic->motion_subsample_log2= 3;

        }

        if(s->avctx->debug&FF_DEBUG_DCT_COEFF) {

            FF_ALLOCZ_OR_GOTO(s->avctx, pic->dct_coeff, 64 * mb_array_size * sizeof(DCTELEM)*6, fail)

        }

        pic->qstride= s->mb_stride;

        FF_ALLOCZ_OR_GOTO(s->avctx, pic->pan_scan , 1 * sizeof(AVPanScan), fail)

    }



    /* It might be nicer if the application would keep track of these

     * but it would require an API change. */

    memmove(s->prev_pict_types+1, s->prev_pict_types, PREV_PICT_TYPES_BUFFER_SIZE-1);

    s->prev_pict_types[0]= s->dropable ? AV_PICTURE_TYPE_B : s->pict_type;

    if(pic->age < PREV_PICT_TYPES_BUFFER_SIZE && s->prev_pict_types[pic->age] == AV_PICTURE_TYPE_B)

        pic->age= INT_MAX; // Skipped MBs in B-frames are quite rare in MPEG-1/2 and it is a bit tricky to skip them anyway.

    pic->owner2 = NULL;



    return 0;

fail: //for the FF_ALLOCZ_OR_GOTO macro

    if(r>=0)

        free_frame_buffer(s, pic);

    return -1;

}
