static inline void encode_dc(MpegEncContext *s, int diff, int component)

{

    if (((unsigned) (diff + 255)) >= 511) {

        int index;



        if (diff < 0) {

            index = av_log2_16bit(-2 * diff);

            diff--;

        } else {

            index = av_log2_16bit(2 * diff);

        }

        if (component == 0)

            put_bits(&s->pb,

                     ff_mpeg12_vlc_dc_lum_bits[index] + index,

                     (ff_mpeg12_vlc_dc_lum_code[index] << index) +

                     (diff & ((1 << index) - 1)));

        else

            put_bits(&s->pb,

                     ff_mpeg12_vlc_dc_chroma_bits[index] + index,

                     (ff_mpeg12_vlc_dc_chroma_code[index] << index) +

                     (diff & ((1 << index) - 1)));

    } else {

        if (component == 0)

            put_bits(&s->pb,

                     mpeg1_lum_dc_uni[diff + 255] & 0xFF,

                     mpeg1_lum_dc_uni[diff + 255] >> 8);

        else

            put_bits(&s->pb,

                     mpeg1_chr_dc_uni[diff + 255] & 0xFF,

                     mpeg1_chr_dc_uni[diff + 255] >> 8);

    }

}
