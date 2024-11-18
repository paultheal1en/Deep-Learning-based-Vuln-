static void decode(GetByteContext *gb, RangeCoder *rc, unsigned cumFreq, unsigned freq, unsigned total_freq)

{

    rc->code -= cumFreq * rc->range;

    rc->range *= freq;



    while (rc->range < TOP && bytestream2_get_bytes_left(gb) > 0) {

        unsigned byte = bytestream2_get_byte(gb);

        rc->code = (rc->code << 8) | byte;

        rc->range <<= 8;

    }

}
