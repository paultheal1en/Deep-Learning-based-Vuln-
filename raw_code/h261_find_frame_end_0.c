static int h261_find_frame_end(ParseContext *pc, AVCodecContext* avctx, const uint8_t *buf, int buf_size){

    int vop_found, i, j, bits_left, last_bits;

    uint32_t state;



    H261Context *h = avctx->priv_data;



    if(h){

        bits_left = h->bits_left;

        last_bits = h->last_bits;

    }

    else{

        bits_left = 0;

        last_bits = 0;

    }



    vop_found= pc->frame_start_found;

    state= pc->state;

    if(bits_left!=0 && !vop_found)

        state = state << (8-bits_left) | last_bits;

    i=0;

    if(!vop_found){

        for(i=0; i<buf_size; i++){

            state= (state<<8) | buf[i];

            for(j=0; j<8; j++){

                if(( (  (state<<j)  |  (buf[i]>>(8-j))  )>>(32-20) == 0x10 )&&(((state >> (17-j)) & 0x4000) == 0x0)){

                    i++;

                    vop_found=1;

                    break;

                }

            }

            if(vop_found)

                    break;    

        }

    }

    if(vop_found){

        for(; i<buf_size; i++){

            if(avctx->flags & CODEC_FLAG_TRUNCATED)//XXX ffplay workaround, someone a better solution?

                state= (state<<8) | buf[i];

            for(j=0; j<8; j++){

                if(( (  (state<<j)  |  (buf[i]>>(8-j))  )>>(32-20) == 0x10 )&&(((state >> (17-j)) & 0x4000) == 0x0)){

                    pc->frame_start_found=0;

                    pc->state=-1;

                    return i-3;

                }

            }

        }

    }



    pc->frame_start_found= vop_found;

    pc->state= state;

    return END_NOT_FOUND;

}
