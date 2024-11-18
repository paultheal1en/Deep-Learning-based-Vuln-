void ff_mpeg_flush(AVCodecContext *avctx){

    int i;

    MpegEncContext *s = avctx->priv_data;



    if(s==NULL || s->picture==NULL)

        return;



    for(i=0; i<MAX_PICTURE_COUNT; i++){

       if(s->picture[i].data[0] && (   s->picture[i].type == FF_BUFFER_TYPE_INTERNAL

                                    || s->picture[i].type == FF_BUFFER_TYPE_USER))

        avctx->release_buffer(avctx, (AVFrame*)&s->picture[i]);

    }

    s->current_picture_ptr = s->last_picture_ptr = s->next_picture_ptr = NULL;



    s->mb_x= s->mb_y= 0;



    s->parse_context.state= -1;

    s->parse_context.frame_start_found= 0;

    s->parse_context.overread= 0;

    s->parse_context.overread_index= 0;

    s->parse_context.index= 0;

    s->parse_context.last_index= 0;

    s->bitstream_buffer_size=0;


}