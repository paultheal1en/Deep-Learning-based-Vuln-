static int sync(AVFormatContext *s, int64_t *timestamp, int *flags, int *stream_index, int64_t *pos){

    RMDemuxContext *rm = s->priv_data;

    ByteIOContext *pb = s->pb;

    int len, num, res, i;

    AVStream *st;

    uint32_t state=0xFFFFFFFF;



    while(!url_feof(pb)){

        *pos= url_ftell(pb) - 3;

        if(rm->remaining_len > 0){

            num= rm->current_stream;

            len= rm->remaining_len;

            *timestamp = AV_NOPTS_VALUE;

            *flags= 0;

        }else{

            state= (state<<8) + get_byte(pb);



            if(state == MKBETAG('I', 'N', 'D', 'X')){

                int n_pkts, expected_len;

                len = get_be32(pb);

                url_fskip(pb, 2);

                n_pkts = get_be32(pb);

                expected_len = 20 + n_pkts * 14;

                if (len == 20)

                    /* some files don't add index entries to chunk size... */

                    len = expected_len;

                else if (len != expected_len)

                    av_log(s, AV_LOG_WARNING,

                           "Index size %d (%d pkts) is wrong, should be %d.\n",

                           len, n_pkts, expected_len);

                len -= 14; // we already read part of the index header

                if(len<0)

                    continue;

                goto skip;

            }



            if(state > (unsigned)0xFFFF || state < 12)

                continue;

            len=state;

            state= 0xFFFFFFFF;



            num = get_be16(pb);

            *timestamp = get_be32(pb);

            res= get_byte(pb); /* reserved */

            *flags = get_byte(pb); /* flags */





            len -= 12;

        }

        for(i=0;i<s->nb_streams;i++) {

            st = s->streams[i];

            if (num == st->id)

                break;

        }

        if (i == s->nb_streams) {

skip:

            /* skip packet if unknown number */

            url_fskip(pb, len);

            rm->remaining_len = 0;

            continue;

        }

        *stream_index= i;



        return len;

    }

    return -1;

}
