int ff_jpegls_decode_picture(MJpegDecodeContext *s, int near, int point_transform, int ilv){

    int i, t = 0;

    uint8_t *zero, *last, *cur;

    JLSState *state;

    int off = 0, stride = 1, width, shift;



    zero = av_mallocz(s->picture.linesize[0]);

    last = zero;

    cur = s->picture.data[0];



    state = av_mallocz(sizeof(JLSState));

    /* initialize JPEG-LS state from JPEG parameters */

    state->near = near;

    state->bpp = (s->bits < 2) ? 2 : s->bits;

    state->maxval = s->maxval;

    state->T1 = s->t1;

    state->T2 = s->t2;

    state->T3 = s->t3;

    state->reset = s->reset;

    ff_jpegls_reset_coding_parameters(state, 0);

    ff_jpegls_init_state(state);



    if(s->bits <= 8)

        shift = point_transform + (8 - s->bits);

    else

        shift = point_transform + (16 - s->bits);



    if (s->avctx->debug & FF_DEBUG_PICT_INFO) {

        av_log(s->avctx, AV_LOG_DEBUG, "JPEG-LS params: %ix%i NEAR=%i MV=%i T(%i,%i,%i) RESET=%i, LIMIT=%i, qbpp=%i, RANGE=%i\n",

                s->width, s->height, state->near, state->maxval,

                state->T1, state->T2, state->T3,

                state->reset, state->limit, state->qbpp, state->range);

        av_log(s->avctx, AV_LOG_DEBUG, "JPEG params: ILV=%i Pt=%i BPP=%i, scan = %i\n",

                ilv, point_transform, s->bits, s->cur_scan);

    }

    if(ilv == 0) { /* separate planes */

        stride = (s->nb_components > 1) ? 3 : 1;

        off = av_clip(s->cur_scan - 1, 0, stride);

        width = s->width * stride;

        cur += off;

        for(i = 0; i < s->height; i++) {

            if(s->bits <= 8){

                ls_decode_line(state, s, last, cur, t, width, stride, off,  8);

                t = last[0];

            }else{

                ls_decode_line(state, s, last, cur, t, width, stride, off, 16);

                t = *((uint16_t*)last);

            }

            last = cur;

            cur += s->picture.linesize[0];



            if (s->restart_interval && !--s->restart_count) {

                align_get_bits(&s->gb);

                skip_bits(&s->gb, 16); /* skip RSTn */

            }

        }

    } else if(ilv == 1) { /* line interleaving */

        int j;

        int Rc[3] = {0, 0, 0};

        stride = (s->nb_components > 1) ? 3 : 1;

        memset(cur, 0, s->picture.linesize[0]);

        width = s->width * stride;

        for(i = 0; i < s->height; i++) {

            for(j = 0; j < stride; j++) {

                ls_decode_line(state, s, last + j, cur + j, Rc[j], width, stride, j, 8);

                Rc[j] = last[j];



                if (s->restart_interval && !--s->restart_count) {

                    align_get_bits(&s->gb);

                    skip_bits(&s->gb, 16); /* skip RSTn */

                }

            }

            last = cur;

            cur += s->picture.linesize[0];

        }

    } else if(ilv == 2) { /* sample interleaving */

        av_log(s->avctx, AV_LOG_ERROR, "Sample interleaved images are not supported.\n");

        av_free(state);

        av_free(zero);

        return -1;

    }



    if(shift){ /* we need to do point transform or normalize samples */

        int x, w;



        w = s->width * s->nb_components;



        if(s->bits <= 8){

            uint8_t *src = s->picture.data[0];



            for(i = 0; i < s->height; i++){

                for(x = off; x < w; x+= stride){

                    src[x] <<= shift;

                }

                src += s->picture.linesize[0];

            }

        }else{

            uint16_t *src = (uint16_t*) s->picture.data[0];



            for(i = 0; i < s->height; i++){

                for(x = 0; x < w; x++){

                    src[x] <<= shift;

                }

                src += s->picture.linesize[0]/2;

            }

        }

    }

    av_free(state);

    av_free(zero);



    return 0;

}
