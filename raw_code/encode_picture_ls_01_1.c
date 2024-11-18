static int encode_picture_ls(AVCodecContext *avctx, unsigned char *buf, int buf_size, void *data){

    JpeglsContext * const s = avctx->priv_data;

    AVFrame *pict = data;

    AVFrame * const p= (AVFrame*)&s->picture;

    const int near = avctx->prediction_method;

    PutBitContext pb, pb2;

    GetBitContext gb;

    uint8_t *buf2, *zero, *cur, *last;

    JLSState *state;

    int i, size;

    int comps;



    buf2 = av_malloc(buf_size);



    init_put_bits(&pb, buf, buf_size);

    init_put_bits(&pb2, buf2, buf_size);



    *p = *pict;

    p->pict_type= FF_I_TYPE;

    p->key_frame= 1;



    if(avctx->pix_fmt == PIX_FMT_GRAY8 || avctx->pix_fmt == PIX_FMT_GRAY16)

        comps = 1;

    else

        comps = 3;



    /* write our own JPEG header, can't use mjpeg_picture_header */

    put_marker(&pb, SOI);

    put_marker(&pb, SOF48);

    put_bits(&pb, 16, 8 + comps * 3); // header size depends on components

    put_bits(&pb,  8, (avctx->pix_fmt == PIX_FMT_GRAY16) ? 16 : 8); // bpp

    put_bits(&pb, 16, avctx->height);

    put_bits(&pb, 16, avctx->width);

    put_bits(&pb,  8, comps);         // components

    for(i = 1; i <= comps; i++) {

        put_bits(&pb,  8, i);    // component ID

        put_bits(&pb,  8, 0x11); // subsampling: none

        put_bits(&pb,  8, 0);    // Tiq, used by JPEG-LS ext

    }



    put_marker(&pb, SOS);

    put_bits(&pb, 16, 6 + comps * 2);

    put_bits(&pb,  8, comps);

    for(i = 1; i <= comps; i++) {

        put_bits(&pb,  8, i);  // component ID

        put_bits(&pb,  8, 0);  // mapping index: none

    }

    put_bits(&pb,  8, near);

    put_bits(&pb,  8, (comps > 1) ? 1 : 0); // interleaving: 0 - plane, 1 - line

    put_bits(&pb,  8, 0); // point transform: none



    state = av_mallocz(sizeof(JLSState));

    /* initialize JPEG-LS state from JPEG parameters */

    state->near = near;

    state->bpp = (avctx->pix_fmt == PIX_FMT_GRAY16) ? 16 : 8;

    reset_ls_coding_parameters(state, 0);

    ls_init_state(state);



    ls_store_lse(state, &pb);



    zero = av_mallocz(p->linesize[0]);

    last = zero;

    cur = p->data[0];

    if(avctx->pix_fmt == PIX_FMT_GRAY8){

        int t = 0;



        for(i = 0; i < avctx->height; i++) {

            ls_encode_line(state, &pb2, last, cur, t, avctx->width, 1, 0,  8);

            t = last[0];

            last = cur;

            cur += p->linesize[0];

        }

    }else if(avctx->pix_fmt == PIX_FMT_GRAY16){

        int t = 0;



        for(i = 0; i < avctx->height; i++) {

            ls_encode_line(state, &pb2, last, cur, t, avctx->width, 1, 0, 16);

            t = *((uint16_t*)last);

            last = cur;

            cur += p->linesize[0];

        }

    }else if(avctx->pix_fmt == PIX_FMT_RGB24){

        int j, width;

        int Rc[3] = {0, 0, 0};



        width = avctx->width * 3;

        for(i = 0; i < avctx->height; i++) {

            for(j = 0; j < 3; j++) {

                ls_encode_line(state, &pb2, last + j, cur + j, Rc[j], width, 3, j, 8);

                Rc[j] = last[j];

            }

            last = cur;

            cur += s->picture.linesize[0];

        }

    }else if(avctx->pix_fmt == PIX_FMT_BGR24){

        int j, width;

        int Rc[3] = {0, 0, 0};



        width = avctx->width * 3;

        for(i = 0; i < avctx->height; i++) {

            for(j = 2; j >= 0; j--) {

                ls_encode_line(state, &pb2, last + j, cur + j, Rc[j], width, 3, j, 8);

                Rc[j] = last[j];

            }

            last = cur;

            cur += s->picture.linesize[0];

        }

    }



    av_free(zero);

    av_free(state);



    flush_put_bits(&pb2);

    /* do escape coding */

    size = put_bits_count(&pb2) >> 3;

    init_get_bits(&gb, buf2, size);

    while(get_bits_count(&gb) < size * 8){

        int v;

        v = get_bits(&gb, 8);

        put_bits(&pb, 8, v);

        if(v == 0xFF){

            v = get_bits(&gb, 7);

            put_bits(&pb, 8, v);

        }

    }

    align_put_bits(&pb);

    av_free(buf2);



    /* End of image */

    put_marker(&pb, EOI);

    flush_put_bits(&pb);



    emms_c();



    return put_bits_count(&pb) >> 3;

}
