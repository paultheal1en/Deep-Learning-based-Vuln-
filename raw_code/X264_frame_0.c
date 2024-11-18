static int X264_frame(AVCodecContext *ctx, AVPacket *pkt, const AVFrame *frame,

                      int *got_packet)

{

    X264Context *x4 = ctx->priv_data;

    x264_nal_t *nal;

    int nnal, i, ret;

    x264_picture_t pic_out = {0};

    int pict_type;



    x264_picture_init( &x4->pic );

    x4->pic.img.i_csp   = x4->params.i_csp;

    if (x264_bit_depth > 8)

        x4->pic.img.i_csp |= X264_CSP_HIGH_DEPTH;

    x4->pic.img.i_plane = avfmt2_num_planes(ctx->pix_fmt);



    if (frame) {

        for (i = 0; i < x4->pic.img.i_plane; i++) {

            x4->pic.img.plane[i]    = frame->data[i];

            x4->pic.img.i_stride[i] = frame->linesize[i];

        }



        x4->pic.i_pts  = frame->pts;



        switch (frame->pict_type) {

        case AV_PICTURE_TYPE_I:

            x4->pic.i_type = x4->forced_idr > 0 ? X264_TYPE_IDR

                                                : X264_TYPE_KEYFRAME;

            break;

        case AV_PICTURE_TYPE_P:

            x4->pic.i_type = X264_TYPE_P;

            break;

        case AV_PICTURE_TYPE_B:

            x4->pic.i_type = X264_TYPE_B;

            break;

        default:

            x4->pic.i_type = X264_TYPE_AUTO;

            break;

        }

        reconfig_encoder(ctx, frame);



        if (x4->a53_cc) {

            void *sei_data;

            size_t sei_size;



            ret = ff_alloc_a53_sei(frame, 0, &sei_data, &sei_size);

            if (ret < 0) {

                av_log(ctx, AV_LOG_ERROR, "Not enough memory for closed captions, skipping\n");

            } else if (sei_data) {

                x4->pic.extra_sei.payloads = av_mallocz(sizeof(x4->pic.extra_sei.payloads[0]));

                if (x4->pic.extra_sei.payloads == NULL) {

                    av_log(ctx, AV_LOG_ERROR, "Not enough memory for closed captions, skipping\n");

                    av_free(sei_data);

                } else {

                    x4->pic.extra_sei.sei_free = av_free;



                    x4->pic.extra_sei.payloads[0].payload_size = sei_size;

                    x4->pic.extra_sei.payloads[0].payload = sei_data;

                    x4->pic.extra_sei.num_payloads = 1;

                    x4->pic.extra_sei.payloads[0].payload_type = 4;

                }

            }

        }

    }



    do {

        if (x264_encoder_encode(x4->enc, &nal, &nnal, frame? &x4->pic: NULL, &pic_out) < 0)

            return AVERROR_EXTERNAL;



        ret = encode_nals(ctx, pkt, nal, nnal);

        if (ret < 0)

            return ret;

    } while (!ret && !frame && x264_encoder_delayed_frames(x4->enc));



    pkt->pts = pic_out.i_pts;

    pkt->dts = pic_out.i_dts;





    switch (pic_out.i_type) {

    case X264_TYPE_IDR:

    case X264_TYPE_I:

        pict_type = AV_PICTURE_TYPE_I;

        break;

    case X264_TYPE_P:

        pict_type = AV_PICTURE_TYPE_P;

        break;

    case X264_TYPE_B:

    case X264_TYPE_BREF:

        pict_type = AV_PICTURE_TYPE_B;

        break;

    default:

        pict_type = AV_PICTURE_TYPE_NONE;

    }

#if FF_API_CODED_FRAME

FF_DISABLE_DEPRECATION_WARNINGS

    ctx->coded_frame->pict_type = pict_type;

FF_ENABLE_DEPRECATION_WARNINGS

#endif



    pkt->flags |= AV_PKT_FLAG_KEY*pic_out.b_keyframe;

    if (ret) {

        ff_side_data_set_encoder_stats(pkt, (pic_out.i_qpplus1 - 1) * FF_QP2LAMBDA, NULL, 0, pict_type);



#if FF_API_CODED_FRAME

FF_DISABLE_DEPRECATION_WARNINGS

        ctx->coded_frame->quality = (pic_out.i_qpplus1 - 1) * FF_QP2LAMBDA;

FF_ENABLE_DEPRECATION_WARNINGS

#endif

    }



    *got_packet = ret;

    return 0;

}
