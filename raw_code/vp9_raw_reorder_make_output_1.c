static int vp9_raw_reorder_make_output(AVBSFContext *bsf,

                                   AVPacket *out,

                                   VP9RawReorderFrame *last_frame)

{

    VP9RawReorderContext *ctx = bsf->priv_data;

    VP9RawReorderFrame *next_output = last_frame,

                      *next_display = last_frame, *frame;

    int s, err;



    for (s = 0; s < FRAME_SLOTS; s++) {

        frame = ctx->slot[s];

        if (!frame)

            continue;

        if (frame->needs_output && (!next_output ||

            frame->sequence < next_output->sequence))

            next_output = frame;

        if (frame->needs_display && (!next_display ||

            frame->pts < next_display->pts))

            next_display = frame;

    }



    if (!next_output && !next_display)

        return AVERROR_EOF;



    if (!next_display || (next_output &&

        next_output->sequence < next_display->sequence))

        frame = next_output;

    else

        frame = next_display;



    if (frame->needs_output && frame->needs_display &&

        next_output == next_display) {

        av_log(bsf, AV_LOG_DEBUG, "Output and display frame "

               "%"PRId64" (%"PRId64") in order.\n",

               frame->sequence, frame->pts);



        av_packet_move_ref(out, frame->packet);



        frame->needs_output = frame->needs_display = 0;

    } else if (frame->needs_output) {

        if (frame->needs_display) {

            av_log(bsf, AV_LOG_DEBUG, "Output frame %"PRId64" "

                   "(%"PRId64") for later display.\n",

                   frame->sequence, frame->pts);

        } else {

            av_log(bsf, AV_LOG_DEBUG, "Output unshown frame "

                   "%"PRId64" (%"PRId64") to keep order.\n",

                   frame->sequence, frame->pts);

        }



        av_packet_move_ref(out, frame->packet);

        out->pts = out->dts;



        frame->needs_output = 0;

    } else {

        PutBitContext pb;



        av_assert0(!frame->needs_output && frame->needs_display);



        if (frame->slots == 0) {

            av_log(bsf, AV_LOG_ERROR, "Attempting to display frame "

                   "which is no longer available?\n");

            frame->needs_display = 0;

            return AVERROR_INVALIDDATA;

        }



        s = ff_ctz(frame->slots);

        av_assert0(s < FRAME_SLOTS);



        av_log(bsf, AV_LOG_DEBUG, "Display frame %"PRId64" "

               "(%"PRId64") from slot %d.\n",

               frame->sequence, frame->pts, s);



        frame->packet = av_packet_alloc();

        if (!frame->packet)

            return AVERROR(ENOMEM);



        err = av_new_packet(out, 2);

        if (err < 0)

            return err;



        init_put_bits(&pb, out->data, 2);



        // frame_marker

        put_bits(&pb, 2, 2);

        // profile_low_bit

        put_bits(&pb, 1, frame->profile & 1);

        // profile_high_bit

        put_bits(&pb, 1, (frame->profile >> 1) & 1);

        if (frame->profile == 3) {

            // reserved_zero

            put_bits(&pb, 1, 0);

        }

        // show_existing_frame

        put_bits(&pb, 1, 1);

        // frame_to_show_map_idx

        put_bits(&pb, 3, s);



        while (put_bits_count(&pb) < 16)

            put_bits(&pb, 1, 0);



        flush_put_bits(&pb);

        out->pts = out->dts = frame->pts;



        frame->needs_display = 0;

    }



    return 0;

}
