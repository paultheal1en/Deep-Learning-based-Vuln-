static int X264_frame(AVCodecContext *ctx, uint8_t *buf,

                      int bufsize, void *data)

{

    X264Context *x4 = ctx->priv_data;

    AVFrame *frame = data;

    x264_nal_t *nal;

    int nnal, i;

    x264_picture_t pic_out;



    x4->pic.img.i_csp   = X264_CSP_I420;

    x4->pic.img.i_plane = 3;



    if (frame) {

        for (i = 0; i < 3; i++) {

            x4->pic.img.plane[i]    = frame->data[i];

            x4->pic.img.i_stride[i] = frame->linesize[i];

        }



        x4->pic.i_pts  = frame->pts;

        x4->pic.i_type = X264_TYPE_AUTO;

    }



    if (x264_encoder_encode(x4->enc, &nal, &nnal, frame? &x4->pic: NULL, &pic_out) < 0)

        return -1;



    bufsize = encode_nals(ctx, buf, bufsize, nal, nnal, 0);

    if (bufsize < 0)

        return -1;



    /* FIXME: dts */

    x4->out_pic.pts = pic_out.i_pts;



    switch (pic_out.i_type) {

    case X264_TYPE_IDR:

    case X264_TYPE_I:

        x4->out_pic.pict_type = FF_I_TYPE;

        break;

    case X264_TYPE_P:

        x4->out_pic.pict_type = FF_P_TYPE;

        break;

    case X264_TYPE_B:

    case X264_TYPE_BREF:

        x4->out_pic.pict_type = FF_B_TYPE;

        break;

    }



    x4->out_pic.key_frame = pic_out.i_type == X264_TYPE_IDR;

    x4->out_pic.quality   = (pic_out.i_qpplus1 - 1) * FF_QP2LAMBDA;



    return bufsize;

}
