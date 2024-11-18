static int matroska_parse_rm_audio(MatroskaDemuxContext *matroska,

                                   MatroskaTrack *track,

                                   AVStream *st,

                                   uint8_t *data, int size,

                                   uint64_t timecode,

                                   int64_t pos)

{

    int a = st->codec->block_align;

    int sps = track->audio.sub_packet_size;

    int cfs = track->audio.coded_framesize;

    int h = track->audio.sub_packet_h;

    int y = track->audio.sub_packet_cnt;

    int w = track->audio.frame_size;

    int x;



    if (!track->audio.pkt_cnt) {

        if (track->audio.sub_packet_cnt == 0)

            track->audio.buf_timecode = timecode;

        if (st->codec->codec_id == AV_CODEC_ID_RA_288) {

            if (size < cfs * h / 2) {

                av_log(matroska->ctx, AV_LOG_ERROR,

                       "Corrupt int4 RM-style audio packet size\n");

                return AVERROR_INVALIDDATA;

            }

            for (x=0; x<h/2; x++)

                memcpy(track->audio.buf+x*2*w+y*cfs,

                       data+x*cfs, cfs);

        } else if (st->codec->codec_id == AV_CODEC_ID_SIPR) {

            if (size < w) {

                av_log(matroska->ctx, AV_LOG_ERROR,

                       "Corrupt sipr RM-style audio packet size\n");

                return AVERROR_INVALIDDATA;

            }

            memcpy(track->audio.buf + y*w, data, w);

        } else {

            if (size < sps * w / sps) {

                av_log(matroska->ctx, AV_LOG_ERROR,

                       "Corrupt generic RM-style audio packet size\n");

                return AVERROR_INVALIDDATA;

            }

            for (x=0; x<w/sps; x++)

                memcpy(track->audio.buf+sps*(h*x+((h+1)/2)*(y&1)+(y>>1)), data+x*sps, sps);

        }



        if (++track->audio.sub_packet_cnt >= h) {

            if (st->codec->codec_id == AV_CODEC_ID_SIPR)

                ff_rm_reorder_sipr_data(track->audio.buf, h, w);

            track->audio.sub_packet_cnt = 0;

            track->audio.pkt_cnt = h*w / a;

        }

    }



    while (track->audio.pkt_cnt) {

        AVPacket *pkt = NULL;

        if (!(pkt = av_mallocz(sizeof(AVPacket))) || av_new_packet(pkt, a) < 0){

            av_free(pkt);

            return AVERROR(ENOMEM);

        }

        memcpy(pkt->data, track->audio.buf

               + a * (h*w / a - track->audio.pkt_cnt--), a);

        pkt->pts = track->audio.buf_timecode;

        track->audio.buf_timecode = AV_NOPTS_VALUE;

        pkt->pos = pos;

        pkt->stream_index = st->index;

        dynarray_add(&matroska->packets,&matroska->num_packets,pkt);

    }



    return 0;

}
