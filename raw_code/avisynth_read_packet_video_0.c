static int avisynth_read_packet_video(AVFormatContext *s, AVPacket *pkt,

                                      int discard)

{

    AviSynthContext *avs = s->priv_data;

    AVS_VideoFrame *frame;

    unsigned char *dst_p;

    const unsigned char *src_p;

    int n, i, plane, rowsize, planeheight, pitch, bits;

    const char *error;



    if (avs->curr_frame >= avs->vi->num_frames)

        return AVERROR_EOF;



    /* This must happen even if the stream is discarded to prevent desync. */

    n = avs->curr_frame++;

    if (discard)

        return 0;



#ifdef USING_AVISYNTH

    /* Define the bpp values for the new AviSynth 2.6 colorspaces.

     * Since AvxSynth doesn't have these functions, special-case

     * it in order to avoid implicit declaration errors. */



    if (avs_library.avs_is_yv24(avs->vi))

        bits = 24;

    else if (avs_library.avs_is_yv16(avs->vi))

        bits = 16;

    else if (avs_library.avs_is_yv411(avs->vi))

        bits = 12;

    else if (avs_library.avs_is_y8(avs->vi))

        bits = 8;

    else

        bits = avs_library.avs_bits_per_pixel(avs->vi);

#else

    bits = avs_bits_per_pixel(avs->vi);

#endif



    /* Without the cast to int64_t, calculation overflows at about 9k x 9k

     * resolution. */

    pkt->size = (((int64_t)avs->vi->width *

                  (int64_t)avs->vi->height) * bits) / 8;

    if (!pkt->size)

        return AVERROR_UNKNOWN;



    if (av_new_packet(pkt, pkt->size) < 0)

        return AVERROR(ENOMEM);



    pkt->pts      = n;

    pkt->dts      = n;

    pkt->duration = 1;

    pkt->stream_index = avs->curr_stream;



    frame = avs_library.avs_get_frame(avs->clip, n);

    error = avs_library.avs_clip_get_error(avs->clip);

    if (error) {

        av_log(s, AV_LOG_ERROR, "%s\n", error);

        avs->error = 1;

        av_packet_unref(pkt);

        return AVERROR_UNKNOWN;

    }



    dst_p = pkt->data;

    for (i = 0; i < avs->n_planes; i++) {

        plane = avs->planes[i];

#ifdef USING_AVISYNTH

        src_p = avs_library.avs_get_read_ptr_p(frame, plane);

        pitch = avs_library.avs_get_pitch_p(frame, plane);



        rowsize     = avs_library.avs_get_row_size_p(frame, plane);

        planeheight = avs_library.avs_get_height_p(frame, plane);

#else

        src_p = avs_get_read_ptr_p(frame, plane);

        pitch = avs_get_pitch_p(frame, plane);



        rowsize     = avs_get_row_size_p(frame, plane);

        planeheight = avs_get_height_p(frame, plane);

#endif



        /* Flip RGB video. */

        if (avs_is_rgb24(avs->vi) || avs_is_rgb(avs->vi)) {

            src_p = src_p + (planeheight - 1) * pitch;

            pitch = -pitch;

        }



        avs_library.avs_bit_blt(avs->env, dst_p, rowsize, src_p, pitch,

                                 rowsize, planeheight);

        dst_p += rowsize * planeheight;

    }



    avs_library.avs_release_video_frame(frame);

    return 0;

}
