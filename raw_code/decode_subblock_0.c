static inline void decode_subblock(DCTELEM *dst, int code, const int is_block2, GetBitContext *gb, VLC *vlc, int q)

{

    int coeffs[4];



    coeffs[0] = modulo_three_table[code][0];

    coeffs[1] = modulo_three_table[code][1];

    coeffs[2] = modulo_three_table[code][2];

    coeffs[3] = modulo_three_table[code][3];

    decode_coeff(dst  , coeffs[0], 3, gb, vlc, q);

    if(is_block2){

        decode_coeff(dst+8, coeffs[1], 2, gb, vlc, q);

        decode_coeff(dst+1, coeffs[2], 2, gb, vlc, q);

    }else{

        decode_coeff(dst+1, coeffs[1], 2, gb, vlc, q);

        decode_coeff(dst+8, coeffs[2], 2, gb, vlc, q);

    }

    decode_coeff(dst+9, coeffs[3], 2, gb, vlc, q);

}
