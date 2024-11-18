static int X264_frame(AVCodecContext *ctx, AVPacket *pkt, const AVFrame *frame,

                      int *got_packet)

{

    X264Context *x4 = ctx->priv_data;

    x264_nal_t *nal;

    int nnal, i, ret;

    x264_picture_t pic_out;



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

        x4->pic.i_type =

            frame->pict_type == AV_PICTURE_TYPE_I ? X264_TYPE_KEYFRAME :

            frame->pict_type == AV_PICTURE_TYPE_P ? X264_TYPE_P :

            frame->pict_type == AV_PICTURE_TYPE_B ? X264_TYPE_B :

                                            X264_TYPE_AUTO;

        if (x4->params.b_interlaced && x4->params.b_tff != frame->top_field_first) {

            x4->params.b_tff = frame->top_field_first;

            x264_encoder_reconfig(x4->enc, &x4->params);

        }

        if (x4->params.vui.i_sar_height != ctx->sample_aspect_ratio.den ||

            x4->params.vui.i_sar_width  != ctx->sample_aspect_ratio.num) {

            x4->params.vui.i_sar_height = ctx->sample_aspect_ratio.den;

            x4->params.vui.i_sar_width  = ctx->sample_aspect_ratio.num;

            x264_encoder_reconfig(x4->enc, &x4->params);

        }

    }



    do {

        if (x264_encoder_encode(x4->enc, &nal, &nnal, frame? &x4->pic: NULL, &pic_out) < 0)

            return -1;



        ret = encode_nals(ctx, pkt, nal, nnal);

        if (ret < 0)

            return -1;

    } while (!ret && !frame && x264_encoder_delayed_frames(x4->enc));



    pkt->pts = pic_out.i_pts;

    pkt->dts = pic_out.i_dts;



    switch (pic_out.i_type) {

    case X264_TYPE_IDR:

    case X264_TYPE_I:

        x4->out_pic.pict_type = AV_PICTURE_TYPE_I;

        break;

    case X264_TYPE_P:

        x4->out_pic.pict_type = AV_PICTURE_TYPE_P;

        break;

    case X264_TYPE_B:

    case X264_TYPE_BREF:

        x4->out_pic.pict_type = AV_PICTURE_TYPE_B;

        break;

    }



    pkt->flags |= AV_PKT_FLAG_KEY*pic_out.b_keyframe;

    if (ret)

        x4->out_pic.quality = (pic_out.i_qpplus1 - 1) * FF_QP2LAMBDA;



    *got_packet = ret;

    return 0;

}
