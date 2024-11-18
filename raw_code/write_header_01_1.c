static int write_header(FlashSV2Context * s, uint8_t * buf, int buf_size)

{

    PutBitContext pb;

    int buf_pos, len;



    if (buf_size < 5)

        return -1;



    init_put_bits(&pb, buf, buf_size * 8);



    put_bits(&pb, 4, (s->block_width  >> 4) - 1);

    put_bits(&pb, 12, s->image_width);

    put_bits(&pb, 4, (s->block_height >> 4) - 1);

    put_bits(&pb, 12, s->image_height);



    flush_put_bits(&pb);

    buf_pos = 4;



    buf[buf_pos++] = s->flags;



    if (s->flags & HAS_PALLET_INFO) {

        len = write_palette(s, buf + buf_pos, buf_size - buf_pos);

        if (len < 0)

            return -1;

        buf_pos += len;

    }



    return buf_pos;

}
