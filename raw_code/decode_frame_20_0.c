static int decode_frame(AVCodecContext *avctx, 

                             void *data, int *data_size,

                             uint8_t *buf, int buf_size)

{

    H264Context *h = avctx->priv_data;

    MpegEncContext *s = &h->s;

    AVFrame *pict = data; 

    float new_aspect;

    int buf_index;

    

    s->flags= avctx->flags;



    *data_size = 0;

   

   /* no supplementary picture */

    if (buf_size == 0) {

        return 0;

    }

    

    if(s->flags&CODEC_FLAG_TRUNCATED){

        int next= find_frame_end(s, buf, buf_size);

        

        if( ff_combine_frame(s, next, &buf, &buf_size) < 0 )

            return buf_size;

//printf("next:%d buf_size:%d last_index:%d\n", next, buf_size, s->parse_context.last_index);

    }



    if(s->avctx->extradata_size && s->picture_number==0){

        if(0 < decode_nal_units(h, s->avctx->extradata, s->avctx->extradata_size) ) 

            return -1;

    }



    buf_index=decode_nal_units(h, buf, buf_size);

    if(buf_index < 0) 

        return -1;



    //FIXME do something with unavailable reference frames    

 

//    if(ret==FRAME_SKIPED) return get_consumed_bytes(s, buf_index, buf_size);

#if 0

    if(s->pict_type==B_TYPE || s->low_delay){

        *pict= *(AVFrame*)&s->current_picture;

    } else {

        *pict= *(AVFrame*)&s->last_picture;

    }

#endif

    *pict= *(AVFrame*)&s->current_picture; //FIXME 

    assert(pict->data[0]);

//printf("out %d\n", (int)pict->data[0]);

    if(avctx->debug&FF_DEBUG_QP){

        int8_t *qtab= pict->qscale_table;

        int x,y;

        

        for(y=0; y<s->mb_height; y++){

            for(x=0; x<s->mb_width; x++){

                printf("%2d ", qtab[x + y*s->mb_width]);

            }

            printf("\n");

        }

        printf("\n");

    }

#if 0 //?



    /* Return the Picture timestamp as the frame number */

    /* we substract 1 because it is added on utils.c    */

    avctx->frame_number = s->picture_number - 1;

#endif

#if 0

    /* dont output the last pic after seeking */

    if(s->last_picture_ptr || s->low_delay)

    //Note this isnt a issue as a IDR pic should flush teh buffers

#endif

        *data_size = sizeof(AVFrame);

    return get_consumed_bytes(s, buf_index, buf_size);

}
