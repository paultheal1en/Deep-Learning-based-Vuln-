static int str_read_packet(AVFormatContext *s,

                           AVPacket *ret_pkt)

{

    ByteIOContext *pb = s->pb;

    StrDemuxContext *str = s->priv_data;

    unsigned char sector[RAW_CD_SECTOR_SIZE];

    int channel;

    AVPacket *pkt;

    AVStream *st;



    while (1) {



        if (get_buffer(pb, sector, RAW_CD_SECTOR_SIZE) != RAW_CD_SECTOR_SIZE)

            return AVERROR(EIO);



        channel = sector[0x11];

        if (channel >= 32)

            return AVERROR_INVALIDDATA;



        switch (sector[0x12] & CDXA_TYPE_MASK) {



        case CDXA_TYPE_DATA:

        case CDXA_TYPE_VIDEO:

            {



                int current_sector = AV_RL16(&sector[0x1C]);

                int sector_count   = AV_RL16(&sector[0x1E]);

                int frame_size = AV_RL32(&sector[0x24]);



                if(!(   frame_size>=0

                     && current_sector < sector_count

                     && sector_count*VIDEO_DATA_CHUNK_SIZE >=frame_size)){

                    av_log(s, AV_LOG_ERROR, "Invalid parameters %d %d %d\n", current_sector, sector_count, frame_size);

                    return AVERROR_INVALIDDATA;

                }



                if(str->channels[channel].video_stream_index < 0){

                    /* allocate a new AVStream */

                    st = av_new_stream(s, 0);

                    if (!st)

                        return AVERROR(ENOMEM);

                    av_set_pts_info(st, 64, 1, 15);



                    str->channels[channel].video_stream_index = st->index;



                    st->codec->codec_type = CODEC_TYPE_VIDEO;

                    st->codec->codec_id   = CODEC_ID_MDEC;

                    st->codec->codec_tag  = 0;  /* no fourcc */

                    st->codec->width      = AV_RL16(&sector[0x28]);

                    st->codec->height     = AV_RL16(&sector[0x2A]);

                }



                /* if this is the first sector of the frame, allocate a pkt */

                pkt = &str->channels[channel].tmp_pkt;



                if(pkt->size != sector_count*VIDEO_DATA_CHUNK_SIZE){

                    if(pkt->data)

                        av_log(s, AV_LOG_ERROR, "missmatching sector_count\n");

                    av_free_packet(pkt);

                    if (av_new_packet(pkt, sector_count*VIDEO_DATA_CHUNK_SIZE))

                        return AVERROR(EIO);



                    pkt->pos= url_ftell(pb) - RAW_CD_SECTOR_SIZE;

                    pkt->stream_index =

                        str->channels[channel].video_stream_index;

                }



                memcpy(pkt->data + current_sector*VIDEO_DATA_CHUNK_SIZE,

                       sector + VIDEO_DATA_HEADER_SIZE,

                       VIDEO_DATA_CHUNK_SIZE);



                if (current_sector == sector_count-1) {

                    pkt->size= frame_size;

                    *ret_pkt = *pkt;

                    pkt->data= NULL;

                    pkt->size= -1;

                    return 0;

                }



            }

            break;



        case CDXA_TYPE_AUDIO:

            if(str->channels[channel].audio_stream_index < 0){

                int fmt = sector[0x13];

                /* allocate a new AVStream */

                st = av_new_stream(s, 0);

                if (!st)

                    return AVERROR(ENOMEM);



                str->channels[channel].audio_stream_index = st->index;



                st->codec->codec_type  = CODEC_TYPE_AUDIO;

                st->codec->codec_id    = CODEC_ID_ADPCM_XA;

                st->codec->codec_tag   = 0;  /* no fourcc */

                st->codec->channels    = (fmt&1)?2:1;

                st->codec->sample_rate = (fmt&4)?18900:37800;

            //    st->codec->bit_rate = 0; //FIXME;

                st->codec->block_align = 128;



                av_set_pts_info(st, 64, 128, st->codec->sample_rate);

            }

                pkt = ret_pkt;

                if (av_new_packet(pkt, 2304))

                    return AVERROR(EIO);

                memcpy(pkt->data,sector+24,2304);



                pkt->stream_index =

                    str->channels[channel].audio_stream_index;

                return 0;

            break;

        default:

            /* drop the sector and move on */

            break;

        }



        if (url_feof(pb))

            return AVERROR(EIO);

    }

}
