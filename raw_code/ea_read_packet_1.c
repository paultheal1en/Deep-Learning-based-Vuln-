static int ea_read_packet(AVFormatContext *s,

                          AVPacket *pkt)

{

    EaDemuxContext *ea = s->priv_data;

    ByteIOContext *pb = s->pb;

    int ret = 0;

    int packet_read = 0;

    unsigned int chunk_type, chunk_size;

    int key = 0;

    int av_uninit(num_samples);



    while (!packet_read) {

        chunk_type = get_le32(pb);

        chunk_size = (ea->big_endian ? get_be32(pb) : get_le32(pb)) - 8;



        switch (chunk_type) {

        /* audio data */

        case ISNh_TAG:

            /* header chunk also contains data; skip over the header portion*/

            url_fskip(pb, 32);

            chunk_size -= 32;

        case ISNd_TAG:

        case SCDl_TAG:

        case SNDC_TAG:

        case SDEN_TAG:

            if (!ea->audio_codec) {

                url_fskip(pb, chunk_size);

                break;

            } else if (ea->audio_codec == CODEC_ID_PCM_S16LE_PLANAR ||

                       ea->audio_codec == CODEC_ID_MP3) {

                num_samples = get_le32(pb);

                url_fskip(pb, 8);

                chunk_size -= 12;

            }

            ret = av_get_packet(pb, pkt, chunk_size);

            if (ret != chunk_size)

                ret = AVERROR(EIO);

            else {

                    pkt->stream_index = ea->audio_stream_index;

                    pkt->pts = 90000;

                    pkt->pts *= ea->audio_frame_counter;

                    pkt->pts /= ea->sample_rate;



                    switch (ea->audio_codec) {

                    case CODEC_ID_ADPCM_EA:

                    /* 2 samples/byte, 1 or 2 samples per frame depending

                     * on stereo; chunk also has 12-byte header */

                    ea->audio_frame_counter += ((chunk_size - 12) * 2) /

                        ea->num_channels;

                        break;

                    case CODEC_ID_PCM_S16LE_PLANAR:

                    case CODEC_ID_MP3:

                        ea->audio_frame_counter += num_samples;

                        break;

                    default:

                        ea->audio_frame_counter += chunk_size /

                            (ea->bytes * ea->num_channels);

                    }

            }



            packet_read = 1;

            break;



        /* ending tag */

        case 0:

        case ISNe_TAG:

        case SCEl_TAG:

        case SEND_TAG:

        case SEEN_TAG:

            ret = AVERROR(EIO);

            packet_read = 1;

            break;



        case MVIh_TAG:

        case kVGT_TAG:

        case pQGT_TAG:

        case TGQs_TAG:

            key = PKT_FLAG_KEY;

        case MVIf_TAG:

        case fVGT_TAG:

            url_fseek(pb, -8, SEEK_CUR);     // include chunk preamble

            chunk_size += 8;

            goto get_video_packet;



        case mTCD_TAG:

            url_fseek(pb, 8, SEEK_CUR);  // skip ea dct header

            chunk_size -= 8;

            goto get_video_packet;



        case MV0K_TAG:

        case MPCh_TAG:

        case pIQT_TAG:

            key = PKT_FLAG_KEY;

        case MV0F_TAG:

get_video_packet:

            ret = av_get_packet(pb, pkt, chunk_size);

            if (ret != chunk_size)

                ret = AVERROR_IO;

            else {

                pkt->stream_index = ea->video_stream_index;

                pkt->flags |= key;

            }

            packet_read = 1;

            break;



        default:

            url_fseek(pb, chunk_size, SEEK_CUR);

            break;

        }

    }



    return ret;

}
