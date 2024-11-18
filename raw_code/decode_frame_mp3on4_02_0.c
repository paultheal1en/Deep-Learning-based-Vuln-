static int decode_frame_mp3on4(AVCodecContext * avctx,

                        void *data, int *data_size,

                        const uint8_t * buf, int buf_size)

{

    MP3On4DecodeContext *s = avctx->priv_data;

    MPADecodeContext *m;

    int len, out_size = 0;

    uint32_t header;

    OUT_INT *out_samples = data;

    OUT_INT decoded_buf[MPA_FRAME_SIZE * MPA_MAX_CHANNELS];

    OUT_INT *outptr, *bp;

    int fsize;

    int fr, i, j, n;



    len = buf_size;



    *data_size = 0;

    // Discard too short frames

    if (buf_size < HEADER_SIZE)

        return -1;



    // If only one decoder interleave is not needed

    outptr = s->frames == 1 ? out_samples : decoded_buf;



    for (fr = 0; fr < s->frames; fr++) {

        fsize = AV_RB16(buf) >> 4;

        fsize = FFMIN3(fsize, len, MPA_MAX_CODED_FRAME_SIZE);

        m = s->mp3decctx[fr];

        assert (m != NULL);



        header = (AV_RB32(buf) & 0x000fffff) | s->syncword; // patch header



        if (ff_mpa_check_header(header) < 0) { // Bad header, discard block

            *data_size = 0;

            return buf_size;

        }



        ff_mpegaudio_decode_header(m, header);

        out_size += mp_decode_frame(m, outptr, buf, fsize);

        buf += fsize;

        len -= fsize;



        if(s->frames > 1) {

            n = m->avctx->frame_size*m->nb_channels;

            /* interleave output data */

            bp = out_samples + s->coff[fr];

            if(m->nb_channels == 1) {

                for(j = 0; j < n; j++) {

                    *bp = decoded_buf[j];

                    bp += avctx->channels;

                }

            } else {

                for(j = 0; j < n; j++) {

                    bp[0] = decoded_buf[j++];

                    bp[1] = decoded_buf[j];

                    bp += avctx->channels;

                }

            }

        }

    }



    /* update codec info */

    avctx->sample_rate = s->mp3decctx[0]->sample_rate;

    avctx->bit_rate = 0;

    for (i = 0; i < s->frames; i++)

        avctx->bit_rate += s->mp3decctx[i]->bit_rate;



    *data_size = out_size;

    return buf_size;

}
