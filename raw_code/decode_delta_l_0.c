static void decode_delta_l(uint8_t *dst,

                           const uint8_t *buf, const uint8_t *buf_end,

                           int w, int flag, int bpp, int dst_size)

{

    GetByteContext off0, off1, dgb, ogb;

    PutByteContext pb;

    unsigned poff0, poff1;

    int i, k, dstpitch;

    int planepitch_byte = (w + 7) / 8;

    int planepitch = ((w + 15) / 16) * 2;

    int pitch = planepitch * bpp;



    if (buf_end - buf <= 64)

        return;



    bytestream2_init(&off0, buf, buf_end - buf);

    bytestream2_init(&off1, buf + 32, buf_end - (buf + 32));

    bytestream2_init_writer(&pb, dst, dst_size);



    dstpitch = flag ? (((w + 7) / 8) * bpp): 2;



    for (k = 0; k < bpp; k++) {

        poff0 = bytestream2_get_be32(&off0);

        poff1 = bytestream2_get_be32(&off1);



        if (!poff0)

            continue;



        if (2LL * poff0 >= buf_end - buf)

            return;



        if (2LL * poff1 >= buf_end - buf)

            return;



        bytestream2_init(&dgb, buf + 2 * poff0, buf_end - (buf + 2 * poff0));

        bytestream2_init(&ogb, buf + 2 * poff1, buf_end - (buf + 2 * poff1));



        while ((bytestream2_peek_be16(&ogb)) != 0xFFFF) {

            uint32_t offset = bytestream2_get_be16(&ogb);

            int16_t cnt = bytestream2_get_be16(&ogb);

            uint16_t data;



            offset = ((2 * offset) / planepitch_byte) * pitch + ((2 * offset) % planepitch_byte) + k * planepitch;

            if (cnt < 0) {

                bytestream2_seek_p(&pb, offset, SEEK_SET);

                cnt = -cnt;

                data = bytestream2_get_be16(&dgb);

                for (i = 0; i < cnt; i++) {

                    bytestream2_put_be16(&pb, data);

                    bytestream2_skip_p(&pb, dstpitch - 2);

                }

            } else {

                bytestream2_seek_p(&pb, offset, SEEK_SET);

                for (i = 0; i < cnt; i++) {

                    data = bytestream2_get_be16(&dgb);

                    bytestream2_put_be16(&pb, data);

                    bytestream2_skip_p(&pb, dstpitch - 2);

                }

            }

        }

    }

}
