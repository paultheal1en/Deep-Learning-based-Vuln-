static int flic_read_packet(AVFormatContext *s,

                            AVPacket *pkt)

{

    FlicDemuxContext *flic = (FlicDemuxContext *)s->priv_data;

    ByteIOContext *pb = &s->pb;

    int packet_read = 0;

    unsigned int size;

    int magic;

    int ret = 0;

    unsigned char preamble[FLIC_PREAMBLE_SIZE];



    while (!packet_read) {



        if ((ret = get_buffer(pb, preamble, FLIC_PREAMBLE_SIZE)) !=

            FLIC_PREAMBLE_SIZE) {

            ret = AVERROR_IO;

            break;

        }



        size = LE_32(&preamble[0]);

        magic = LE_16(&preamble[4]);



        if ((magic == FLIC_CHUNK_MAGIC_1) || (magic == FLIC_CHUNK_MAGIC_2)) {

            if (av_new_packet(pkt, size)) {

                ret = AVERROR_IO;

                break;

            }

            pkt->stream_index = flic->video_stream_index;

            pkt->pts = flic->pts;

            memcpy(pkt->data, preamble, FLIC_PREAMBLE_SIZE);

            ret = get_buffer(pb, pkt->data + FLIC_PREAMBLE_SIZE, 

                size - FLIC_PREAMBLE_SIZE);

            if (ret != size - FLIC_PREAMBLE_SIZE) {

                av_free_packet(pkt);

                ret = AVERROR_IO;

            }

            flic->pts += flic->frame_pts_inc;

            packet_read = 1;

        } else {

            /* not interested in this chunk */

            url_fseek(pb, size - 6, SEEK_CUR);

        }

    }



    return ret;

}
