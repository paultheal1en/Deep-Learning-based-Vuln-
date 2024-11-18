static int lavfi_read_packet(AVFormatContext *avctx, AVPacket *pkt)

{

    LavfiContext *lavfi = avctx->priv_data;

    double min_pts = DBL_MAX;

    int stream_idx, min_pts_sink_idx = 0;

    AVFilterBufferRef *ref;

    AVPicture pict;

    int ret, i;

    int size = 0;



    /* iterate through all the graph sinks. Select the sink with the

     * minimum PTS */

    for (i = 0; i < avctx->nb_streams; i++) {

        AVRational tb = lavfi->sinks[i]->inputs[0]->time_base;

        double d;

        int ret;



        if (lavfi->sink_eof[i])

            continue;



        ret = av_buffersink_get_buffer_ref(lavfi->sinks[i],

                                       &ref, AV_BUFFERSINK_FLAG_PEEK);

        if (ret == AVERROR_EOF) {

            av_dlog(avctx, "EOF sink_idx:%d\n", i);

            lavfi->sink_eof[i] = 1;

            continue;

        } else if (ret < 0)

            return ret;

        d = av_rescale_q(ref->pts, tb, AV_TIME_BASE_Q);

        av_dlog(avctx, "sink_idx:%d time:%f\n", i, d);



        if (d < min_pts) {

            min_pts = d;

            min_pts_sink_idx = i;

        }

    }

    if (min_pts == DBL_MAX)

        return AVERROR_EOF;



    av_dlog(avctx, "min_pts_sink_idx:%i\n", min_pts_sink_idx);



    av_buffersink_get_buffer_ref(lavfi->sinks[min_pts_sink_idx], &ref, 0);

    stream_idx = lavfi->sink_stream_map[min_pts_sink_idx];



    if (ref->video) {

        size = avpicture_get_size(ref->format, ref->video->w, ref->video->h);

        if ((ret = av_new_packet(pkt, size)) < 0)

            return ret;



        memcpy(pict.data,     ref->data,     4*sizeof(ref->data[0]));

        memcpy(pict.linesize, ref->linesize, 4*sizeof(ref->linesize[0]));



        avpicture_layout(&pict, ref->format, ref->video->w,

                         ref->video->h, pkt->data, size);

    } else if (ref->audio) {

        size = ref->audio->nb_samples *

            av_get_bytes_per_sample(ref->format) *

            av_get_channel_layout_nb_channels(ref->audio->channel_layout);

        if ((ret = av_new_packet(pkt, size)) < 0)

            return ret;

        memcpy(pkt->data, ref->data[0], size);

    }



    if (ref->metadata) {

        uint8_t *metadata;

        AVDictionaryEntry *e = NULL;

        AVBPrint meta_buf;



        av_bprint_init(&meta_buf, 0, AV_BPRINT_SIZE_UNLIMITED);

        while ((e = av_dict_get(ref->metadata, "", e, AV_DICT_IGNORE_SUFFIX))) {

            av_bprintf(&meta_buf, "%s", e->key);

            av_bprint_chars(&meta_buf, '\0', 1);

            av_bprintf(&meta_buf, "%s", e->value);

            av_bprint_chars(&meta_buf, '\0', 1);

        }

        if (!av_bprint_is_complete(&meta_buf) ||

            !(metadata = av_packet_new_side_data(pkt, AV_PKT_DATA_STRINGS_METADATA,

                                                 meta_buf.len))) {

            av_bprint_finalize(&meta_buf, NULL);

            return AVERROR(ENOMEM);

        }

        memcpy(metadata, meta_buf.str, meta_buf.len);

        av_bprint_finalize(&meta_buf, NULL);

    }



    pkt->stream_index = stream_idx;

    pkt->pts = ref->pts;

    pkt->pos = ref->pos;

    pkt->size = size;

    avfilter_unref_buffer(ref);

    return size;

}
