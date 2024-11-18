static void flush_encoders(OutputStream *ost_table, int nb_ostreams)

{

    int i, ret;



    for (i = 0; i < nb_ostreams; i++) {

        OutputStream   *ost = &ost_table[i];

        AVCodecContext *enc = ost->st->codec;

        AVFormatContext *os = output_files[ost->file_index].ctx;



        if (!ost->encoding_needed)

            continue;



        if (ost->st->codec->codec_type == AVMEDIA_TYPE_AUDIO && enc->frame_size <=1)

            continue;

        if (ost->st->codec->codec_type == AVMEDIA_TYPE_VIDEO && (os->oformat->flags & AVFMT_RAWPICTURE))

            continue;



        for(;;) {

            AVPacket pkt;

            int fifo_bytes;

            av_init_packet(&pkt);

            pkt.stream_index= ost->index;



            switch (ost->st->codec->codec_type) {

            case AVMEDIA_TYPE_AUDIO:

                fifo_bytes = av_fifo_size(ost->fifo);

                ret = 0;

                /* encode any samples remaining in fifo */

                if (fifo_bytes > 0) {

                    int osize = av_get_bytes_per_sample(enc->sample_fmt);

                    int fs_tmp = enc->frame_size;



                    av_fifo_generic_read(ost->fifo, audio_buf, fifo_bytes, NULL);

                    if (enc->codec->capabilities & CODEC_CAP_SMALL_LAST_FRAME) {

                        enc->frame_size = fifo_bytes / (osize * enc->channels);

                    } else { /* pad */

                        int frame_bytes = enc->frame_size*osize*enc->channels;

                        if (allocated_audio_buf_size < frame_bytes)

                            exit_program(1);

                        generate_silence(audio_buf+fifo_bytes, enc->sample_fmt, frame_bytes - fifo_bytes);

                    }



                    ret = avcodec_encode_audio(enc, bit_buffer, bit_buffer_size, (short *)audio_buf);

                    pkt.duration = av_rescale((int64_t)enc->frame_size*ost->st->time_base.den,

                                              ost->st->time_base.num, enc->sample_rate);

                    enc->frame_size = fs_tmp;

                }

                if (ret <= 0) {

                    ret = avcodec_encode_audio(enc, bit_buffer, bit_buffer_size, NULL);

                }

                if (ret < 0) {

                    av_log(NULL, AV_LOG_FATAL, "Audio encoding failed\n");

                    exit_program(1);

                }

                audio_size += ret;

                pkt.flags |= AV_PKT_FLAG_KEY;

                break;

            case AVMEDIA_TYPE_VIDEO:

                ret = avcodec_encode_video(enc, bit_buffer, bit_buffer_size, NULL);

                if (ret < 0) {

                    av_log(NULL, AV_LOG_FATAL, "Video encoding failed\n");

                    exit_program(1);

                }

                video_size += ret;

                if(enc->coded_frame && enc->coded_frame->key_frame)

                    pkt.flags |= AV_PKT_FLAG_KEY;

                if (ost->logfile && enc->stats_out) {

                    fprintf(ost->logfile, "%s", enc->stats_out);

                }

                break;

            default:

                ret=-1;

            }



            if (ret <= 0)

                break;

            pkt.data = bit_buffer;

            pkt.size = ret;

            if (enc->coded_frame && enc->coded_frame->pts != AV_NOPTS_VALUE)

                pkt.pts= av_rescale_q(enc->coded_frame->pts, enc->time_base, ost->st->time_base);

            write_frame(os, &pkt, ost->st->codec, ost->bitstream_filters);

        }

    }

}
