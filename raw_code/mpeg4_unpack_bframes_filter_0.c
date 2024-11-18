static int mpeg4_unpack_bframes_filter(AVBSFContext *ctx, AVPacket *out)

{

    UnpackBFramesBSFContext *s = ctx->priv_data;

    int pos_p = -1, nb_vop = 0, pos_vop2 = -1, ret = 0;

    AVPacket *in;



    ret = ff_bsf_get_packet(ctx, &in);

    if (ret < 0)

        return ret;



    scan_buffer(in->data, in->size, &pos_p, &nb_vop, &pos_vop2);

    av_log(ctx, AV_LOG_DEBUG, "Found %d VOP startcode(s) in this packet.\n", nb_vop);



    if (pos_vop2 >= 0) {

        if (s->b_frame_buf) {

            av_log(ctx, AV_LOG_WARNING,

                   "Missing one N-VOP packet, discarding one B-frame.\n");

            av_freep(&s->b_frame_buf);

            s->b_frame_buf_size = 0;

        }

        /* store the packed B-frame in the BSFContext */

        s->b_frame_buf_size = in->size - pos_vop2;

        s->b_frame_buf      = create_new_buffer(in->data + pos_vop2, s->b_frame_buf_size);

        if (!s->b_frame_buf) {

            s->b_frame_buf_size = 0;

            av_packet_free(&in);

            return AVERROR(ENOMEM);

        }

    }



    if (nb_vop > 2) {

        av_log(ctx, AV_LOG_WARNING,

       "Found %d VOP headers in one packet, only unpacking one.\n", nb_vop);

    }



    if (nb_vop == 1 && s->b_frame_buf) {

        /* use frame from BSFContext */

        ret = av_packet_copy_props(out, in);

        if (ret < 0) {

            av_packet_free(&in);

            return ret;

        }



        av_packet_from_data(out, s->b_frame_buf, s->b_frame_buf_size);

        if (in->size <= MAX_NVOP_SIZE) {

            /* N-VOP */

            av_log(ctx, AV_LOG_DEBUG, "Skipping N-VOP.\n");

            s->b_frame_buf      = NULL;

            s->b_frame_buf_size = 0;

        } else {

            /* copy packet into BSFContext */

            s->b_frame_buf_size = in->size;

            s->b_frame_buf      = create_new_buffer(in->data, in->size);

            if (!s->b_frame_buf) {

                s->b_frame_buf_size = 0;

                av_packet_unref(out);

                av_packet_free(&in);

                return AVERROR(ENOMEM);

            }

        }

    } else if (nb_vop >= 2) {

        /* use first frame of the packet */

        av_packet_move_ref(out, in);

        out->size = pos_vop2;

    } else if (pos_p >= 0) {

        av_log(ctx, AV_LOG_DEBUG, "Updating DivX userdata (remove trailing 'p').\n");

        av_packet_move_ref(out, in);

        /* remove 'p' (packed) from the end of the (DivX) userdata string */

        out->data[pos_p] = '\0';

    } else {

        /* copy packet */

        av_packet_move_ref(out, in);

    }



    av_packet_free(&in);



    return 0;

}
