static void build_file_streams(void)

{

    FFStream *stream, *stream_next;

    AVFormatContext *infile;

    int i;



    /* gather all streams */

    for(stream = first_stream; stream != NULL; stream = stream_next) {

        stream_next = stream->next;

        if (stream->stream_type == STREAM_TYPE_LIVE &&

            !stream->feed) {

            /* the stream comes from a file */

            /* try to open the file */

            /* open stream */

            stream->ap_in = av_mallocz(sizeof(AVFormatParameters));

            if (!strcmp(stream->fmt->name, "rtp")) {

                /* specific case : if transport stream output to RTP,

                   we use a raw transport stream reader */

                stream->ap_in->mpeg2ts_raw = 1;

                stream->ap_in->mpeg2ts_compute_pcr = 1;

            }



            if (av_open_input_file(&infile, stream->feed_filename,

                                   stream->ifmt, 0, stream->ap_in) < 0) {

                http_log("%s not found", stream->feed_filename);

                /* remove stream (no need to spend more time on it) */

            fail:

                remove_stream(stream);

            } else {

                /* find all the AVStreams inside and reference them in

                   'stream' */

                if (av_find_stream_info(infile) < 0) {

                    http_log("Could not find codec parameters from '%s'",

                             stream->feed_filename);

                    av_close_input_file(infile);

                    goto fail;

                }

                extract_mpeg4_header(infile);



                for(i=0;i<infile->nb_streams;i++)

                    add_av_stream1(stream, infile->streams[i]->codec);



                av_close_input_file(infile);

            }

        }

    }

}
