static int rv10_decode_frame(AVCodecContext *avctx, 

                             void *data, int *data_size,

                             UINT8 *buf, int buf_size)

{

    MpegEncContext *s = avctx->priv_data;

    int i, mb_count, mb_pos, left;

    DCTELEM block[6][64];

    AVPicture *pict = data; 



#ifdef DEBUG

    printf("*****frame %d size=%d\n", avctx->frame_number, buf_size);

#endif



    /* no supplementary picture */

    if (buf_size == 0) {

        *data_size = 0;

        return 0;

    }



    init_get_bits(&s->gb, buf, buf_size);



    mb_count = rv10_decode_picture_header(s);

    if (mb_count < 0) {

#ifdef DEBUG

        printf("HEADER ERROR\n");

#endif

        return -1;

    }

    

    if (s->mb_x >= s->mb_width ||

        s->mb_y >= s->mb_height) {

#ifdef DEBUG

        printf("POS ERROR %d %d\n", s->mb_x, s->mb_y);

#endif

        return -1;

    }

    mb_pos = s->mb_y * s->mb_width + s->mb_x;

    left = s->mb_width * s->mb_height - mb_pos;

    if (mb_count > left) {

#ifdef DEBUG

        printf("COUNT ERROR\n");

#endif

        return -1;

    }



    if (s->mb_x == 0 && s->mb_y == 0) {

        MPV_frame_start(s, avctx);

    }



#ifdef DEBUG

    printf("qscale=%d\n", s->qscale);

#endif



    /* default quantization values */

    s->y_dc_scale = 8;

    s->c_dc_scale = 8;

    s->rv10_first_dc_coded[0] = 0;

    s->rv10_first_dc_coded[1] = 0;

    s->rv10_first_dc_coded[2] = 0;



    s->block_wrap[0]=

    s->block_wrap[1]=

    s->block_wrap[2]=

    s->block_wrap[3]= s->mb_width*2 + 2;

    s->block_wrap[4]=

    s->block_wrap[5]= s->mb_width + 2;

    s->block_index[0]= s->block_wrap[0]*(s->mb_y*2 + 1) - 1 + s->mb_x*2;

    s->block_index[1]= s->block_wrap[0]*(s->mb_y*2 + 1)     + s->mb_x*2;

    s->block_index[2]= s->block_wrap[0]*(s->mb_y*2 + 2) - 1 + s->mb_x*2;

    s->block_index[3]= s->block_wrap[0]*(s->mb_y*2 + 2)     + s->mb_x*2;

    s->block_index[4]= s->block_wrap[4]*(s->mb_y + 1)                    + s->block_wrap[0]*(s->mb_height*2 + 2) + s->mb_x;

    s->block_index[5]= s->block_wrap[4]*(s->mb_y + 1 + s->mb_height + 2) + s->block_wrap[0]*(s->mb_height*2 + 2) + s->mb_x;

    /* decode each macroblock */

    for(i=0;i<mb_count;i++) {

        s->block_index[0]+=2;

        s->block_index[1]+=2;

        s->block_index[2]+=2;

        s->block_index[3]+=2;

        s->block_index[4]++;

        s->block_index[5]++;

#ifdef DEBUG

        printf("**mb x=%d y=%d\n", s->mb_x, s->mb_y);

#endif

        

        memset(block, 0, sizeof(block));

        s->mv_dir = MV_DIR_FORWARD;

        s->mv_type = MV_TYPE_16X16; 

        if (h263_decode_mb(s, block) < 0) {

#ifdef DEBUG

            printf("ERROR\n");

#endif

            return -1;

        }

        MPV_decode_mb(s, block);

        if (++s->mb_x == s->mb_width) {

            s->mb_x = 0;

            s->mb_y++;

            s->block_index[0]= s->block_wrap[0]*(s->mb_y*2 + 1) - 1;

            s->block_index[1]= s->block_wrap[0]*(s->mb_y*2 + 1);

            s->block_index[2]= s->block_wrap[0]*(s->mb_y*2 + 2) - 1;

            s->block_index[3]= s->block_wrap[0]*(s->mb_y*2 + 2);

            s->block_index[4]= s->block_wrap[4]*(s->mb_y + 1)                    + s->block_wrap[0]*(s->mb_height*2 + 2);

            s->block_index[5]= s->block_wrap[4]*(s->mb_y + 1 + s->mb_height + 2) + s->block_wrap[0]*(s->mb_height*2 + 2);

        }

    }



    if (s->mb_x == 0 &&

        s->mb_y == s->mb_height) {

        MPV_frame_end(s);

        

        pict->data[0] = s->current_picture[0];

        pict->data[1] = s->current_picture[1];

        pict->data[2] = s->current_picture[2];

        pict->linesize[0] = s->linesize;

        pict->linesize[1] = s->uvlinesize;

        pict->linesize[2] = s->uvlinesize;

        

        avctx->quality = s->qscale;

        *data_size = sizeof(AVPicture);

    } else {

        *data_size = 0;

    }

    return buf_size;

}
