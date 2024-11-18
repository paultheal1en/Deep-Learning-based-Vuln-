static int get_packet_payload_size(AVFormatContext *ctx, int stream_index,

                                   int64_t pts, int64_t dts)

{

    MpegMuxContext *s = ctx->priv_data;

    int buf_index;

    StreamInfo *stream;



    stream = ctx->streams[stream_index]->priv_data;



    buf_index = 0;

    if (((s->packet_number % s->pack_header_freq) == 0)) {

        /* pack header size */

        if (s->is_mpeg2) 

            buf_index += 14;

        else

            buf_index += 12;

        

        if (s->is_vcd) {

            /* there is exactly one system header for each stream in a VCD MPEG,

               One in the very first video packet and one in the very first

               audio packet (see VCD standard p. IV-7 and IV-8).*/

            

            if (stream->packet_number==0)

                /* The system headers refer only to the stream they occur in,

                   so they have a constant size.*/

                buf_index += 15;



        } else {            

            if ((s->packet_number % s->system_header_freq) == 0)

                buf_index += s->system_header_size;

        }

    }



    if (s->is_vcd && stream->packet_number==0)

        /* the first pack of each stream contains only the pack header,

           the system header and some padding (see VCD standard p. IV-6) 

           Add the padding size, so that the actual payload becomes 0.*/

        buf_index += s->packet_size - buf_index;

    else {

        /* packet header size */

        buf_index += 6;

        if (s->is_mpeg2)

            buf_index += 3;

        if (pts != AV_NOPTS_VALUE) {

            if (dts != pts)

                buf_index += 5 + 5;

            else

                buf_index += 5;



        } else {

            if (!s->is_mpeg2)

                buf_index++;

        }

    

        if (stream->id < 0xc0) {

            /* AC3/LPCM private data header */

            buf_index += 4;

            if (stream->id >= 0xa0) {

                int n;

                buf_index += 3;

                /* NOTE: we round the payload size to an integer number of

                   LPCM samples */

                n = (s->packet_size - buf_index) % stream->lpcm_align;

                if (n)

                    buf_index += (stream->lpcm_align - n);

            }

        }



        if (s->is_vcd && stream->id == AUDIO_ID)

            /* The VCD standard demands that 20 zero bytes follow

               each audio packet (see standard p. IV-8).*/

            buf_index+=20;

    }

    return s->packet_size - buf_index; 

}
