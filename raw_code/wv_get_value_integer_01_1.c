static inline int wv_get_value_integer(WavpackFrameContext *s, uint32_t *crc, int S)

{

    int bit;



    if(s->extra_bits){

        S <<= s->extra_bits;



        if(s->got_extra_bits){

            S |= get_bits(&s->gb_extra_bits, s->extra_bits);

            *crc = *crc * 9 + (S&0xffff) * 3 + ((unsigned)S>>16);

        }

    }

    bit = (S & s->and) | s->or;

    return (((S + bit) << s->shift) - bit) << s->post_shift;

}
