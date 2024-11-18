static int decode_0(AVCodecContext *avctx, uint8_t code, uint8_t *pkt)

{

    PAFVideoDecContext *c = avctx->priv_data;

    uint32_t opcode_size, offset;

    uint8_t *dst, *dend, mask = 0, color = 0, a, b, p;

    const uint8_t *src, *send, *opcodes;

    int  i, j, x = 0;



    i = bytestream2_get_byte(&c->gb);

    if (i) {

        if (code & 0x10) {

            int align;



            align = bytestream2_tell(&c->gb) & 3;

            if (align)

                bytestream2_skip(&c->gb, 4 - align);

        }

        do {

            a      = bytestream2_get_byte(&c->gb);

            b      = bytestream2_get_byte(&c->gb);

            p      = (a & 0xC0) >> 6;

            dst    = c->frame[p] + get_video_page_offset(avctx, a, b);

            dend   = c->frame[p] + c->frame_size;

            offset = (b & 0x7F) * 2;

            j      = bytestream2_get_le16(&c->gb) + offset;



            do {

                offset++;

                if (dst + 3 * avctx->width + 4 > dend)

                    return AVERROR_INVALIDDATA;

                copy4h(avctx, dst);

                if ((offset & 0x3F) == 0)

                    dst += avctx->width * 3;

                dst += 4;

            } while (offset < j);

        } while (--i);

    }



    dst = c->frame[c->current_frame];

    do {

        a    = bytestream2_get_byte(&c->gb);

        b    = bytestream2_get_byte(&c->gb);

        p    = (a & 0xC0) >> 6;

        src  = c->frame[p] + get_video_page_offset(avctx, a, b);

        send = c->frame[p] + c->frame_size;

        if (src + 3 * avctx->width + 4 > send)

            return AVERROR_INVALIDDATA;

        copy_block4(dst, src, avctx->width, avctx->width, 4);

        i++;

        if ((i & 0x3F) == 0)

            dst += avctx->width * 3;

        dst += 4;

    } while (i < c->video_size / 16);



    opcode_size = bytestream2_get_le16(&c->gb);

    bytestream2_skip(&c->gb, 2);



    if (bytestream2_get_bytes_left(&c->gb) < opcode_size)

        return AVERROR_INVALIDDATA;



    opcodes = pkt + bytestream2_tell(&c->gb);

    bytestream2_skipu(&c->gb, opcode_size);



    dst = c->frame[c->current_frame];



    for (i = 0; i < avctx->height; i += 4, dst += avctx->width * 3) {

        for (j = 0; j < avctx->width; j += 4, dst += 4) {

            int opcode, k = 0;



            if (x > opcode_size)

                return AVERROR_INVALIDDATA;

            if (j & 4) {

                opcode = opcodes[x] & 15;

                x++;

            } else {

                opcode = opcodes[x] >> 4;

            }



            while (block_sequences[opcode][k]) {



                offset = avctx->width * 2;

                code   = block_sequences[opcode][k++];



                switch (code) {

                case 2:

                    offset = 0;

                case 3:

                    color  = bytestream2_get_byte(&c->gb);

                case 4:

                    mask   = bytestream2_get_byte(&c->gb);

                    copy_color_mask(avctx, mask, dst + offset, color);

                    break;

                case 5:

                    offset = 0;

                case 6:

                    a    = bytestream2_get_byte(&c->gb);

                    b    = bytestream2_get_byte(&c->gb);

                    p    = (a & 0xC0) >> 6;

                    src  = c->frame[p] + get_video_page_offset(avctx, a, b);

                    send = c->frame[p] + c->frame_size;

                case 7:

                    if (src + offset + avctx->width + 4 > send)

                        return AVERROR_INVALIDDATA;

                    mask = bytestream2_get_byte(&c->gb);

                    copy_src_mask(avctx, mask, dst + offset, src + offset);

                    break;

                }

            }

        }

    }



    return 0;

}
