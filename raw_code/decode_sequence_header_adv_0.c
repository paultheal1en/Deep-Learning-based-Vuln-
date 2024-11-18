static int decode_sequence_header_adv(VC1Context *v, GetBitContext *gb)

{

    v->res_rtm_flag = 1;

    v->level = get_bits(gb, 3);

    if (v->level >= 5) {

        av_log(v->s.avctx, AV_LOG_ERROR, "Reserved LEVEL %i\n",v->level);

    }

    v->chromaformat = get_bits(gb, 2);

    if (v->chromaformat != 1) {

        av_log(v->s.avctx, AV_LOG_ERROR,

               "Only 4:2:0 chroma format supported\n");

        return -1;

    }



    // (fps-2)/4 (->30)

    v->frmrtq_postproc       = get_bits(gb, 3); //common

    // (bitrate-32kbps)/64kbps

    v->bitrtq_postproc       = get_bits(gb, 5); //common

    v->postprocflag          = get_bits1(gb);   //common



    v->s.avctx->coded_width  = (get_bits(gb, 12) + 1) << 1;

    v->s.avctx->coded_height = (get_bits(gb, 12) + 1) << 1;

    v->s.avctx->width        = v->s.avctx->coded_width;

    v->s.avctx->height       = v->s.avctx->coded_height;

    v->broadcast             = get_bits1(gb);

    v->interlace             = get_bits1(gb);

    v->tfcntrflag            = get_bits1(gb);

    v->finterpflag           = get_bits1(gb);

    skip_bits1(gb); // reserved



    av_log(v->s.avctx, AV_LOG_DEBUG,

           "Advanced Profile level %i:\nfrmrtq_postproc=%i, bitrtq_postproc=%i\n"

           "LoopFilter=%i, ChromaFormat=%i, Pulldown=%i, Interlace: %i\n"

           "TFCTRflag=%i, FINTERPflag=%i\n",

           v->level, v->frmrtq_postproc, v->bitrtq_postproc,

           v->s.loop_filter, v->chromaformat, v->broadcast, v->interlace,

           v->tfcntrflag, v->finterpflag);



    v->psf = get_bits1(gb);

    if (v->psf) { //PsF, 6.1.13

        av_log(v->s.avctx, AV_LOG_ERROR, "Progressive Segmented Frame mode: not supported (yet)\n");

        return -1;

    }

    v->s.max_b_frames = v->s.avctx->max_b_frames = 7;

    if (get_bits1(gb)) { //Display Info - decoding is not affected by it

        int w, h, ar = 0;

        av_log(v->s.avctx, AV_LOG_DEBUG, "Display extended info:\n");

        w = get_bits(gb, 14) + 1;

        h = get_bits(gb, 14) + 1;

        av_log(v->s.avctx, AV_LOG_DEBUG, "Display dimensions: %ix%i\n", w, h);

        if (get_bits1(gb))

            ar = get_bits(gb, 4);

        if (ar && ar < 14) {

            v->s.avctx->sample_aspect_ratio = ff_vc1_pixel_aspect[ar];

        } else if (ar == 15) {

            w = get_bits(gb, 8) + 1;

            h = get_bits(gb, 8) + 1;

            v->s.avctx->sample_aspect_ratio = (AVRational){w, h};

        } else {

            av_reduce(&v->s.avctx->sample_aspect_ratio.num,

                      &v->s.avctx->sample_aspect_ratio.den,

                      v->s.avctx->height * w,

                      v->s.avctx->width * h,

                      1 << 30);

        }

        av_log(v->s.avctx, AV_LOG_DEBUG, "Aspect: %i:%i\n",

               v->s.avctx->sample_aspect_ratio.num,

               v->s.avctx->sample_aspect_ratio.den);



        if (get_bits1(gb)) { //framerate stuff

            if (get_bits1(gb)) {

                v->s.avctx->time_base.num = 32;

                v->s.avctx->time_base.den = get_bits(gb, 16) + 1;

            } else {

                int nr, dr;

                nr = get_bits(gb, 8);

                dr = get_bits(gb, 4);

                if (nr && nr < 8 && dr && dr < 3) {

                    v->s.avctx->time_base.num = ff_vc1_fps_dr[dr - 1];

                    v->s.avctx->time_base.den = ff_vc1_fps_nr[nr - 1] * 1000;

                }

            }

            if (v->broadcast) { // Pulldown may be present

                v->s.avctx->time_base.den  *= 2;

                v->s.avctx->ticks_per_frame = 2;

            }

        }



        if (get_bits1(gb)) {

            v->color_prim    = get_bits(gb, 8);

            v->transfer_char = get_bits(gb, 8);

            v->matrix_coef   = get_bits(gb, 8);

        }

    }



    v->hrd_param_flag = get_bits1(gb);

    if (v->hrd_param_flag) {

        int i;

        v->hrd_num_leaky_buckets = get_bits(gb, 5);

        skip_bits(gb, 4); //bitrate exponent

        skip_bits(gb, 4); //buffer size exponent

        for (i = 0; i < v->hrd_num_leaky_buckets; i++) {

            skip_bits(gb, 16); //hrd_rate[n]

            skip_bits(gb, 16); //hrd_buffer[n]

        }

    }

    return 0;

}
