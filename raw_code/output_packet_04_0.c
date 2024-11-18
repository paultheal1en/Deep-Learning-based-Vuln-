static void output_packet(AVFormatContext *s, AVPacket *pkt, OutputStream *ost)

{

    int ret = 0;



    /* apply the output bitstream filters, if any */

    if (ost->nb_bitstream_filters) {

        int idx;



        ret = av_bsf_send_packet(ost->bsf_ctx[0], pkt);

        if (ret < 0)

            goto finish;



        idx = 1;

        while (idx) {

            /* get a packet from the previous filter up the chain */

            ret = av_bsf_receive_packet(ost->bsf_ctx[idx - 1], pkt);

            if (ret == AVERROR(EAGAIN)) {

                ret = 0;

                idx--;

                continue;

            } else if (ret < 0)

                goto finish;



            /* send it to the next filter down the chain or to the muxer */

            if (idx < ost->nb_bitstream_filters) {

                ret = av_bsf_send_packet(ost->bsf_ctx[idx], pkt);

                if (ret < 0)

                    goto finish;

                idx++;

            } else

                write_packet(s, pkt, ost);

        }

    } else

        write_packet(s, pkt, ost);



finish:

    if (ret < 0 && ret != AVERROR_EOF) {

        av_log(NULL, AV_LOG_FATAL, "Error applying bitstream filters to an output "

               "packet for stream #%d:%d.\n", ost->file_index, ost->index);

        exit_program(1);

    }

}
