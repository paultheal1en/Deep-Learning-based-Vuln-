static int rv10_decode_frame(AVCodecContext *avctx,

                             void *data, int *data_size,

                             AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

    MpegEncContext *s = avctx->priv_data;

    int i;

    AVFrame *pict = data;

    int slice_count;

    const uint8_t *slices_hdr = NULL;



    av_dlog(avctx, "*****frame %d size=%d\n", avctx->frame_number, buf_size);



    /* no supplementary picture */

    if (buf_size == 0) {

        return 0;

    }



    if(!avctx->slice_count){

        slice_count = (*buf++) + 1;

        slices_hdr = buf + 4;

        buf += 8 * slice_count;

    }else

        slice_count = avctx->slice_count;



    for(i=0; i<slice_count; i++){

        int offset= get_slice_offset(avctx, slices_hdr, i);

        int size, size2;



        if(i+1 == slice_count)

            size= buf_size - offset;

        else

            size= get_slice_offset(avctx, slices_hdr, i+1) - offset;



        if(i+2 >= slice_count)

            size2= buf_size - offset;

        else

            size2= get_slice_offset(avctx, slices_hdr, i+2) - offset;



        if(rv10_decode_packet(avctx, buf+offset, size, size2) > 8*size)

            i++;

    }



    if(s->current_picture_ptr != NULL && s->mb_y>=s->mb_height){

        ff_er_frame_end(s);

        MPV_frame_end(s);



        if (s->pict_type == AV_PICTURE_TYPE_B || s->low_delay) {

            *pict= *(AVFrame*)s->current_picture_ptr;

        } else if (s->last_picture_ptr != NULL) {

            *pict= *(AVFrame*)s->last_picture_ptr;

        }



        if(s->last_picture_ptr || s->low_delay){

            *data_size = sizeof(AVFrame);

            ff_print_debug_info(s, pict);

        }

        s->current_picture_ptr= NULL; //so we can detect if frame_end wasnt called (find some nicer solution...)

    }



    return buf_size;

}
