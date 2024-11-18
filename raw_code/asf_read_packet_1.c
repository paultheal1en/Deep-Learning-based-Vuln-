static int asf_read_packet(AVFormatContext *s, AVPacket *pkt)

{

    ASFContext *asf = s->priv_data;

    ASFStream *asf_st = 0;

    ByteIOContext *pb = &s->pb;

    //static int pc = 0;

    for (;;) {

        int rsize = 0;

        if (asf->packet_size_left < FRAME_HEADER_SIZE

            || asf->packet_segments < 1) {

            //asf->packet_size_left <= asf->packet_padsize) {

            int ret = asf->packet_size_left + asf->packet_padsize;

            //printf("PacketLeftSize:%d  Pad:%d Pos:%Ld\n", asf->packet_size_left, asf->packet_padsize, url_ftell(pb));

            if((url_ftell(&s->pb) + ret - s->data_offset) % asf->packet_size)

                ret += asf->packet_size - ((url_ftell(&s->pb) + ret - s->data_offset) % asf->packet_size);

            /* fail safe */

            url_fskip(pb, ret);

            asf->packet_pos= url_ftell(&s->pb);




            ret = asf_get_packet(s);

            //printf("READ ASF PACKET  %d   r:%d   c:%d\n", ret, asf->packet_size_left, pc++);

            if (ret < 0 || url_feof(pb))

                return AVERROR_IO;

            asf->packet_time_start = 0;

            continue;

        }

        if (asf->packet_time_start == 0) {

            /* read frame header */

            int num = get_byte(pb);

            asf->packet_segments--;

            rsize++;

            asf->packet_key_frame = (num & 0x80) >> 7;

            asf->stream_index = asf->asfid2avid[num & 0x7f];

            // sequence should be ignored!

            DO_2BITS(asf->packet_property >> 4, asf->packet_seq, 0);

            DO_2BITS(asf->packet_property >> 2, asf->packet_frag_offset, 0);

            DO_2BITS(asf->packet_property, asf->packet_replic_size, 0);

//printf("key:%d stream:%d seq:%d offset:%d replic_size:%d\n", asf->packet_key_frame, asf->stream_index, asf->packet_seq, //asf->packet_frag_offset, asf->packet_replic_size);

            if (asf->packet_replic_size > 1) {

                assert(asf->packet_replic_size >= 8);

                // it should be always at least 8 bytes - FIXME validate

                asf->packet_obj_size = get_le32(pb);

                asf->packet_frag_timestamp = get_le32(pb); // timestamp

                if (asf->packet_replic_size > 8)

                    url_fskip(pb, asf->packet_replic_size - 8);

                rsize += asf->packet_replic_size; // FIXME - check validity

            } else if (asf->packet_replic_size==1){

                // multipacket - frag_offset is begining timestamp

                asf->packet_time_start = asf->packet_frag_offset;

                asf->packet_frag_offset = 0;

                asf->packet_frag_timestamp = asf->packet_timestamp;



                asf->packet_time_delta = get_byte(pb);

                rsize++;

            }else{

                assert(asf->packet_replic_size==0);

            }

            if (asf->packet_flags & 0x01) {

                DO_2BITS(asf->packet_segsizetype >> 6, asf->packet_frag_size, 0); // 0 is illegal

#undef DO_2BITS

                //printf("Fragsize %d\n", asf->packet_frag_size);

            } else {

                asf->packet_frag_size = asf->packet_size_left - rsize;

                //printf("Using rest  %d %d %d\n", asf->packet_frag_size, asf->packet_size_left, rsize);

            }

            if (asf->packet_replic_size == 1) {

                asf->packet_multi_size = asf->packet_frag_size;

                if (asf->packet_multi_size > asf->packet_size_left) {

                    asf->packet_segments = 0;

                    continue;

                }

            }

            asf->packet_size_left -= rsize;

            //printf("___objsize____  %d   %d    rs:%d\n", asf->packet_obj_size, asf->packet_frag_offset, rsize);



            if (asf->stream_index < 0

                || s->streams[asf->stream_index]->discard >= AVDISCARD_ALL

                || (!asf->packet_key_frame && s->streams[asf->stream_index]->discard >= AVDISCARD_NONKEY)

                ) {

                asf->packet_time_start = 0;

                /* unhandled packet (should not happen) */

                url_fskip(pb, asf->packet_frag_size);

                asf->packet_size_left -= asf->packet_frag_size;

                if(asf->stream_index < 0)

                    av_log(s, AV_LOG_ERROR, "ff asf skip %d  %d\n", asf->packet_frag_size, num & 0x7f);

                continue;

            }

            asf->asf_st = s->streams[asf->stream_index]->priv_data;

        }

        asf_st = asf->asf_st;



        if ((asf->packet_frag_offset != asf_st->frag_offset

             || (asf->packet_frag_offset

                 && asf->packet_seq != asf_st->seq)) // seq should be ignored

           ) {

            /* cannot continue current packet: free it */

            // FIXME better check if packet was already allocated

            av_log(s, AV_LOG_INFO, "ff asf parser skips: %d - %d     o:%d - %d    %d %d   fl:%d\n",

                   asf_st->pkt.size,

                   asf->packet_obj_size,

                   asf->packet_frag_offset, asf_st->frag_offset,

                   asf->packet_seq, asf_st->seq, asf->packet_frag_size);

            if (asf_st->pkt.size)

                av_free_packet(&asf_st->pkt);

            asf_st->frag_offset = 0;

            if (asf->packet_frag_offset != 0) {

                url_fskip(pb, asf->packet_frag_size);

                av_log(s, AV_LOG_INFO, "ff asf parser skipping %db\n", asf->packet_frag_size);

                asf->packet_size_left -= asf->packet_frag_size;

                continue;

            }

        }

        if (asf->packet_replic_size == 1) {

            // frag_offset is here used as the begining timestamp

            asf->packet_frag_timestamp = asf->packet_time_start;

            asf->packet_time_start += asf->packet_time_delta;

            asf->packet_obj_size = asf->packet_frag_size = get_byte(pb);

            asf->packet_size_left--;

            asf->packet_multi_size--;

            if (asf->packet_multi_size < asf->packet_obj_size)

            {

                asf->packet_time_start = 0;

                url_fskip(pb, asf->packet_multi_size);

                asf->packet_size_left -= asf->packet_multi_size;

                continue;

            }

            asf->packet_multi_size -= asf->packet_obj_size;

            //printf("COMPRESS size  %d  %d  %d   ms:%d\n", asf->packet_obj_size, asf->packet_frag_timestamp, asf->packet_size_left, asf->packet_multi_size);

        }

        if (asf_st->frag_offset == 0) {

            /* new packet */

            av_new_packet(&asf_st->pkt, asf->packet_obj_size);

            asf_st->seq = asf->packet_seq;

            asf_st->pkt.pts = asf->packet_frag_timestamp;

            asf_st->pkt.stream_index = asf->stream_index;

            asf_st->pkt.pos =

            asf_st->packet_pos= asf->packet_pos;

//printf("new packet: stream:%d key:%d packet_key:%d audio:%d size:%d\n",

//asf->stream_index, asf->packet_key_frame, asf_st->pkt.flags & PKT_FLAG_KEY,

//s->streams[asf->stream_index]->codec->codec_type == CODEC_TYPE_AUDIO, asf->packet_obj_size);

            if (s->streams[asf->stream_index]->codec->codec_type == CODEC_TYPE_AUDIO)

                asf->packet_key_frame = 1;

            if (asf->packet_key_frame)

                asf_st->pkt.flags |= PKT_FLAG_KEY;

        }



        /* read data */

        //printf("READ PACKET s:%d  os:%d  o:%d,%d  l:%d   DATA:%p\n",

        //       asf->packet_size, asf_st->pkt.size, asf->packet_frag_offset,

        //       asf_st->frag_offset, asf->packet_frag_size, asf_st->pkt.data);

        asf->packet_size_left -= asf->packet_frag_size;

        if (asf->packet_size_left < 0)

            continue;

        get_buffer(pb, asf_st->pkt.data + asf->packet_frag_offset,

                   asf->packet_frag_size);

        asf_st->frag_offset += asf->packet_frag_size;

        /* test if whole packet is read */

        if (asf_st->frag_offset == asf_st->pkt.size) {

            /* return packet */

            if (asf_st->ds_span > 1) {

                /* packet descrambling */

                char* newdata = av_malloc(asf_st->pkt.size);

                if (newdata) {

                    int offset = 0;

                    while (offset < asf_st->pkt.size) {

                        int off = offset / asf_st->ds_chunk_size;

                        int row = off / asf_st->ds_span;

                        int col = off % asf_st->ds_span;

                        int idx = row + col * asf_st->ds_packet_size / asf_st->ds_chunk_size;

                        //printf("off:%d  row:%d  col:%d  idx:%d\n", off, row, col, idx);

                        memcpy(newdata + offset,

                               asf_st->pkt.data + idx * asf_st->ds_chunk_size,

                               asf_st->ds_chunk_size);

                        offset += asf_st->ds_chunk_size;

                    }

                    av_free(asf_st->pkt.data);

                    asf_st->pkt.data = newdata;

                }

            }

            asf_st->frag_offset = 0;

            memcpy(pkt, &asf_st->pkt, sizeof(AVPacket));

            //printf("packet %d %d\n", asf_st->pkt.size, asf->packet_frag_size);

            asf_st->pkt.size = 0;

            asf_st->pkt.data = 0;

            break; // packet completed

        }

    }

    return 0;

}