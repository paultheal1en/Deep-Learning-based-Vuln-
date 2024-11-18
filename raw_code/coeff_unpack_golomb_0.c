static inline int coeff_unpack_golomb(GetBitContext *gb, int qfactor, int qoffset)

{

    int sign, coeff;

    uint32_t buf;



    OPEN_READER(re, gb);

    UPDATE_CACHE(re, gb);

    buf = GET_CACHE(re, gb);



    if (buf & 0xAA800000) {

        buf >>= 32 - 8;

        SKIP_BITS(re, gb, ff_interleaved_golomb_vlc_len[buf]);



        coeff = ff_interleaved_ue_golomb_vlc_code[buf];

    } else {

        unsigned ret = 1;



        do {

            buf >>= 32 - 8;

            SKIP_BITS(re, gb,

                           FFMIN(ff_interleaved_golomb_vlc_len[buf], 8));



            if (ff_interleaved_golomb_vlc_len[buf] != 9) {

                ret <<= (ff_interleaved_golomb_vlc_len[buf] - 1) >> 1;

                ret  |= ff_interleaved_dirac_golomb_vlc_code[buf];

                break;

            }

            ret = (ret << 4) | ff_interleaved_dirac_golomb_vlc_code[buf];

            UPDATE_CACHE(re, gb);

            buf = GET_CACHE(re, gb);

        } while (ret<0x8000000U && BITS_AVAILABLE(re, gb));



        coeff = ret - 1;

    }

    if (coeff) {

        coeff = (coeff * qfactor + qoffset + 2) >> 2;

        sign  = SHOW_SBITS(re, gb, 1);

        LAST_SKIP_BITS(re, gb, 1);

        coeff = (coeff ^ sign) - sign;

    }

    CLOSE_READER(re, gb);

    return coeff;

}
