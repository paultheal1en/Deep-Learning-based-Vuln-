static int movie_get_frame(AVFilterLink *outlink)

{

    MovieContext *movie = outlink->src->priv;

    AVPacket pkt;

    int ret, frame_decoded;

    AVStream *st = movie->format_ctx->streams[movie->stream_index];



    if (movie->is_done == 1)

        return 0;



    while ((ret = av_read_frame(movie->format_ctx, &pkt)) >= 0) {

        // Is this a packet from the video stream?

        if (pkt.stream_index == movie->stream_index) {

            avcodec_decode_video2(movie->codec_ctx, movie->frame, &frame_decoded, &pkt);



            if (frame_decoded) {

                /* FIXME: avoid the memcpy */

                movie->picref = avfilter_get_video_buffer(outlink, AV_PERM_WRITE | AV_PERM_PRESERVE |

                                                          AV_PERM_REUSE2, outlink->w, outlink->h);

                av_image_copy(movie->picref->data, movie->picref->linesize,

                              (void*)movie->frame->data,  movie->frame->linesize,

                              movie->picref->format, outlink->w, outlink->h);

                avfilter_copy_frame_props(movie->picref, movie->frame);



                /* FIXME: use a PTS correction mechanism as that in

                 * ffplay.c when some API will be available for that */

                /* use pkt_dts if pkt_pts is not available */

                movie->picref->pts = movie->frame->pkt_pts == AV_NOPTS_VALUE ?

                    movie->frame->pkt_dts : movie->frame->pkt_pts;



                if (!movie->frame->sample_aspect_ratio.num)

                    movie->picref->video->sample_aspect_ratio = st->sample_aspect_ratio;

                av_dlog(outlink->src,

                        "movie_get_frame(): file:'%s' pts:%"PRId64" time:%lf pos:%"PRId64" aspect:%d/%d\n",

                        movie->file_name, movie->picref->pts,

                        (double)movie->picref->pts * av_q2d(st->time_base),

                        movie->picref->pos,

                        movie->picref->video->sample_aspect_ratio.num,

                        movie->picref->video->sample_aspect_ratio.den);

                // We got it. Free the packet since we are returning

                av_free_packet(&pkt);



                return 0;

            }

        }

        // Free the packet that was allocated by av_read_frame

        av_free_packet(&pkt);

    }



    // On multi-frame source we should stop the mixing process when

    // the movie source does not have more frames

    if (ret == AVERROR_EOF)

        movie->is_done = 1;

    return ret;

}
