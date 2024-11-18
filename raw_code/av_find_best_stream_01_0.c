int av_find_best_stream(AVFormatContext *ic, enum AVMediaType type,

                        int wanted_stream_nb, int related_stream,

                        AVCodec **decoder_ret, int flags)

{

    int i, nb_streams = ic->nb_streams;

    int ret = AVERROR_STREAM_NOT_FOUND, best_count = -1, best_bitrate = -1, best_multiframe = -1, count, bitrate, multiframe;

    unsigned *program = NULL;

    const AVCodec *decoder = NULL, *best_decoder = NULL;



    if (related_stream >= 0 && wanted_stream_nb < 0) {

        AVProgram *p = av_find_program_from_stream(ic, NULL, related_stream);

        if (p) {

            program    = p->stream_index;

            nb_streams = p->nb_stream_indexes;

        }

    }

    for (i = 0; i < nb_streams; i++) {

        int real_stream_index = program ? program[i] : i;

        AVStream *st          = ic->streams[real_stream_index];

        AVCodecContext *avctx = st->codec;

        if (avctx->codec_type != type)

            continue;

        if (wanted_stream_nb >= 0 && real_stream_index != wanted_stream_nb)

            continue;

        if (wanted_stream_nb != real_stream_index &&

            st->disposition & (AV_DISPOSITION_HEARING_IMPAIRED |

                               AV_DISPOSITION_VISUAL_IMPAIRED))

            continue;

        if (type == AVMEDIA_TYPE_AUDIO && !avctx->channels)

            continue;

        if (decoder_ret) {

            decoder = find_decoder(ic, st, st->codec->codec_id);

            if (!decoder) {

                if (ret < 0)

                    ret = AVERROR_DECODER_NOT_FOUND;

                continue;

            }

        }

        count = st->codec_info_nb_frames;

        bitrate = avctx->bit_rate;

        if (!bitrate)

            bitrate = avctx->rc_max_rate;

        multiframe = FFMIN(5, count);

        if ((best_multiframe >  multiframe) ||

            (best_multiframe == multiframe && best_bitrate >  bitrate) ||

            (best_multiframe == multiframe && best_bitrate == bitrate && best_count >= count))

            continue;

        best_count   = count;

        best_bitrate = bitrate;

        best_multiframe = multiframe;

        ret          = real_stream_index;

        best_decoder = decoder;

        if (program && i == nb_streams - 1 && ret < 0) {

            program    = NULL;

            nb_streams = ic->nb_streams;

            /* no related stream found, try again with everything */

            i = 0;

        }

    }

    if (decoder_ret)

        *decoder_ret = (AVCodec*)best_decoder;

    return ret;

}
