static int video_thread(void *arg)

{

    VideoState *is = arg;

    AVPacket pkt1, *pkt = &pkt1;

    int len1, got_picture;

    AVFrame *frame= avcodec_alloc_frame();

    double pts;



    for(;;) {

        while (is->paused && !is->videoq.abort_request) {

            SDL_Delay(10);

        }

        if (packet_queue_get(&is->videoq, pkt, 1) < 0)

            break;



        if(pkt->data == flush_pkt.data){

            avcodec_flush_buffers(is->video_st->codec);

            is->last_dts_for_fault_detection=

            is->last_pts_for_fault_detection= INT64_MIN;

            continue;

        }



        /* NOTE: ipts is the PTS of the _first_ picture beginning in

           this packet, if any */

        is->video_st->codec->reordered_opaque= pkt->pts;

        len1 = avcodec_decode_video2(is->video_st->codec,

                                    frame, &got_picture,

                                    pkt);



        if(pkt->dts != AV_NOPTS_VALUE){

            is->faulty_dts += pkt->dts <= is->last_dts_for_fault_detection;

            is->last_dts_for_fault_detection= pkt->dts;

        }

        if(frame->reordered_opaque != AV_NOPTS_VALUE){

            is->faulty_pts += frame->reordered_opaque <= is->last_pts_for_fault_detection;

            is->last_pts_for_fault_detection= frame->reordered_opaque;

        }



        if(   (   decoder_reorder_pts==1

               || decoder_reorder_pts && is->faulty_pts<is->faulty_dts

               || pkt->dts == AV_NOPTS_VALUE)

           && frame->reordered_opaque != AV_NOPTS_VALUE)

            pts= frame->reordered_opaque;

        else if(pkt->dts != AV_NOPTS_VALUE)

            pts= pkt->dts;

        else

            pts= 0;

        pts *= av_q2d(is->video_st->time_base);



//            if (len1 < 0)

//                break;

        if (got_picture) {

            if (output_picture2(is, frame, pts) < 0)

                goto the_end;

        }

        av_free_packet(pkt);

        if (step)

            if (cur_stream)

                stream_pause(cur_stream);

    }

 the_end:

    av_free(frame);

    return 0;

}
