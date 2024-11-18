static int dash_flush(AVFormatContext *s, int final, int stream)

{

    DASHContext *c = s->priv_data;

    int i, ret = 0;



    const char *proto = avio_find_protocol_name(s->filename);

    int use_rename = proto && !strcmp(proto, "file");



    int cur_flush_segment_index = 0;

    if (stream >= 0)

        cur_flush_segment_index = c->streams[stream].segment_index;



    for (i = 0; i < s->nb_streams; i++) {

        OutputStream *os = &c->streams[i];

        AVStream *st = s->streams[i];

        char filename[1024] = "", full_path[1024], temp_path[1024];

        int range_length, index_length = 0;



        if (!os->packets_written)

            continue;



        // Flush the single stream that got a keyframe right now.

        // Flush all audio streams as well, in sync with video keyframes,

        // but not the other video streams.

        if (stream >= 0 && i != stream) {

            if (s->streams[i]->codecpar->codec_type != AVMEDIA_TYPE_AUDIO)

                continue;

            // Make sure we don't flush audio streams multiple times, when

            // all video streams are flushed one at a time.

            if (c->has_video && os->segment_index > cur_flush_segment_index)

                continue;

        }



        if (!os->init_range_length) {

            flush_init_segment(s, os);

        }



        if (!c->single_file) {

            ff_dash_fill_tmpl_params(filename, sizeof(filename), c->media_seg_name, i, os->segment_index, os->bit_rate, os->start_pts);

            snprintf(full_path, sizeof(full_path), "%s%s", c->dirname, filename);

            snprintf(temp_path, sizeof(temp_path), use_rename ? "%s.tmp" : "%s", full_path);

            ret = s->io_open(s, &os->out, temp_path, AVIO_FLAG_WRITE, NULL);

            if (ret < 0)

                break;

            if (!strcmp(os->format_name, "mp4"))

                write_styp(os->ctx->pb);

        } else {

            snprintf(full_path, sizeof(full_path), "%s%s", c->dirname, os->initfile);

        }



        ret = flush_dynbuf(os, &range_length);

        if (ret < 0)

            break;

        os->packets_written = 0;



        if (c->single_file) {

            find_index_range(s, full_path, os->pos, &index_length);

        } else {

            ff_format_io_close(s, &os->out);



            if (use_rename) {

                ret = avpriv_io_move(temp_path, full_path);

                if (ret < 0)

                    break;

            }

        }



        if (!os->bit_rate) {

            // calculate average bitrate of first segment

            int64_t bitrate = (int64_t) range_length * 8 * AV_TIME_BASE / av_rescale_q(os->max_pts - os->start_pts,

                                                                                       st->time_base,

                                                                                       AV_TIME_BASE_Q);

            if (bitrate >= 0) {

                os->bit_rate = bitrate;

                snprintf(os->bandwidth_str, sizeof(os->bandwidth_str),

                     " bandwidth=\"%d\"", os->bit_rate);

            }

        }

        add_segment(os, filename, os->start_pts, os->max_pts - os->start_pts, os->pos, range_length, index_length);

        av_log(s, AV_LOG_VERBOSE, "Representation %d media segment %d written to: %s\n", i, os->segment_index, full_path);



        os->pos += range_length;

    }



    if (c->window_size || (final && c->remove_at_exit)) {

        for (i = 0; i < s->nb_streams; i++) {

            OutputStream *os = &c->streams[i];

            int j;

            int remove = os->nb_segments - c->window_size - c->extra_window_size;

            if (final && c->remove_at_exit)

                remove = os->nb_segments;

            if (remove > 0) {

                for (j = 0; j < remove; j++) {

                    char filename[1024];

                    snprintf(filename, sizeof(filename), "%s%s", c->dirname, os->segments[j]->file);

                    unlink(filename);

                    av_free(os->segments[j]);

                }

                os->nb_segments -= remove;

                memmove(os->segments, os->segments + remove, os->nb_segments * sizeof(*os->segments));

            }

        }

    }



    if (ret >= 0)

        ret = write_manifest(s, final);

    return ret;

}
