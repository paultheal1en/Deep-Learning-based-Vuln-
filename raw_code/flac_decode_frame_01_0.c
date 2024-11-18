static int flac_decode_frame(AVCodecContext *avctx,

                            void *data, int *data_size,

                            const uint8_t *buf, int buf_size)

{

    FLACContext *s = avctx->priv_data;

    int tmp = 0, i, j = 0, input_buf_size = 0;

    int16_t *samples_16 = data;

    int32_t *samples_32 = data;

    int alloc_data_size= *data_size;



    *data_size=0;



    if (s->max_framesize == 0) {

        s->max_framesize= FFMAX(4, buf_size); // should hopefully be enough for the first header

        s->bitstream= av_fast_realloc(s->bitstream, &s->allocated_bitstream_size, s->max_framesize);

    }



    if (1 && s->max_framesize) { //FIXME truncated

        if (s->bitstream_size < 4 || AV_RL32(s->bitstream) != MKTAG('f','L','a','C'))

            buf_size= FFMIN(buf_size, s->max_framesize - FFMIN(s->bitstream_size, s->max_framesize));

        input_buf_size= buf_size;



        if (s->bitstream_size + buf_size < buf_size || s->bitstream_index + s->bitstream_size + buf_size < s->bitstream_index)

            return -1;



        if (s->allocated_bitstream_size < s->bitstream_size + buf_size)

            s->bitstream= av_fast_realloc(s->bitstream, &s->allocated_bitstream_size, s->bitstream_size + buf_size);



        if (s->bitstream_index + s->bitstream_size + buf_size > s->allocated_bitstream_size) {

            memmove(s->bitstream, &s->bitstream[s->bitstream_index],

                    s->bitstream_size);

            s->bitstream_index=0;

        }

        memcpy(&s->bitstream[s->bitstream_index + s->bitstream_size],

               buf, buf_size);

        buf= &s->bitstream[s->bitstream_index];

        buf_size += s->bitstream_size;

        s->bitstream_size= buf_size;



        if (buf_size < s->max_framesize && input_buf_size) {

            return input_buf_size;

        }

    }



    init_get_bits(&s->gb, buf, buf_size*8);



    if (metadata_parse(s))

        goto end;



    tmp = show_bits(&s->gb, 16);

    if ((tmp & 0xFFFE) != 0xFFF8) {

        av_log(s->avctx, AV_LOG_ERROR, "FRAME HEADER not here\n");

        while (get_bits_count(&s->gb)/8+2 < buf_size && (show_bits(&s->gb, 16) & 0xFFFE) != 0xFFF8)

            skip_bits(&s->gb, 8);

        goto end; // we may not have enough bits left to decode a frame, so try next time

    }

    skip_bits(&s->gb, 16);

    if (decode_frame(s, alloc_data_size) < 0) {

        av_log(s->avctx, AV_LOG_ERROR, "decode_frame() failed\n");

        s->bitstream_size=0;

        s->bitstream_index=0;

        return -1;

    }



#define DECORRELATE(left, right)\

            assert(s->channels == 2);\

            for (i = 0; i < s->blocksize; i++) {\

                int a= s->decoded[0][i];\

                int b= s->decoded[1][i];\

                if (s->is32) {\

                    *samples_32++ = (left)  << s->sample_shift;\

                    *samples_32++ = (right) << s->sample_shift;\

                } else {\

                    *samples_16++ = (left)  << s->sample_shift;\

                    *samples_16++ = (right) << s->sample_shift;\

                }\

            }\

            break;



    switch (s->decorrelation) {

    case INDEPENDENT:

        for (j = 0; j < s->blocksize; j++) {

            for (i = 0; i < s->channels; i++) {

                if (s->is32)

                    *samples_32++ = s->decoded[i][j] << s->sample_shift;

                else

                    *samples_16++ = s->decoded[i][j] << s->sample_shift;

            }

        }

        break;

    case LEFT_SIDE:

        DECORRELATE(a,a-b)

    case RIGHT_SIDE:

        DECORRELATE(a+b,b)

    case MID_SIDE:

        DECORRELATE( (a-=b>>1) + b, a)

    }



    *data_size = s->blocksize * s->channels * (s->is32 ? 4 : 2);



end:

    i= (get_bits_count(&s->gb)+7)/8;

    if (i > buf_size) {

        av_log(s->avctx, AV_LOG_ERROR, "overread: %d\n", i - buf_size);

        s->bitstream_size=0;

        s->bitstream_index=0;

        return -1;

    }



    if (s->bitstream_size) {

        s->bitstream_index += i;

        s->bitstream_size  -= i;

        return input_buf_size;

    } else

        return i;

}
