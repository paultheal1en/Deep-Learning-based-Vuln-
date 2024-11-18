static int decode_0(PAFVideoDecContext *c, uint8_t *pkt, uint8_t code)

{

    uint32_t opcode_size, offset;

    uint8_t *dst, *dend, mask = 0, color = 0;

    const uint8_t *src, *send, *opcodes;

    int i, j, op = 0;



    i = bytestream2_get_byte(&c->gb);

    if (i) {

        if (code & 0x10) {

            int align;



            align = bytestream2_tell(&c->gb) & 3;

            if (align)

                bytestream2_skip(&c->gb, 4 - align);

        }

        do {

            int page, val, x, y;

            val    = bytestream2_get_be16(&c->gb);

            page   = val >> 14;

            x      = (val & 0x7F) * 2;

            y      = ((val >> 7) & 0x7F) * 2;

            dst    = c->frame[page] + x + y * c->width;

            dend   = c->frame[page] + c->frame_size;

            offset = (x & 0x7F) * 2;

            j      = bytestream2_get_le16(&c->gb) + offset;

            do {

                offset++;

                if (dst + 3 * c->width + 4 > dend)

                    return AVERROR_INVALIDDATA;

                read4x4block(c, dst, c->width);

                if ((offset & 0x3F) == 0)

                    dst += c->width * 3;

                dst += 4;

            } while (offset < j);

        } while (--i);

    }



    dst  = c->frame[c->current_frame];

    dend = c->frame[c->current_frame] + c->frame_size;

    do {

        set_src_position(c, &src, &send);

        if ((src + 3 * c->width + 4 > send) ||

            (dst + 3 * c->width + 4 > dend))

            return AVERROR_INVALIDDATA;

        copy_block4(dst, src, c->width, c->width, 4);

        i++;

        if ((i & 0x3F) == 0)

            dst += c->width * 3;

        dst += 4;

    } while (i < c->video_size / 16);



    opcode_size = bytestream2_get_le16(&c->gb);

    bytestream2_skip(&c->gb, 2);



    if (bytestream2_get_bytes_left(&c->gb) < opcode_size)

        return AVERROR_INVALIDDATA;



    opcodes = pkt + bytestream2_tell(&c->gb);

    bytestream2_skipu(&c->gb, opcode_size);



    dst = c->frame[c->current_frame];



    for (i = 0; i < c->height; i += 4, dst += c->width * 3)

        for (j = 0; j < c->width; j += 4, dst += 4) {

            int opcode, k = 0;

            if (op > opcode_size)

                return AVERROR_INVALIDDATA;

            if (j & 4) {

                opcode = opcodes[op] & 15;

                op++;

            } else {

                opcode = opcodes[op] >> 4;

            }



            while (block_sequences[opcode][k]) {

                offset = c->width * 2;

                code   = block_sequences[opcode][k++];



                switch (code) {

                case 2:

                    offset = 0;

                case 3:

                    color = bytestream2_get_byte(&c->gb);

                case 4:

                    mask = bytestream2_get_byte(&c->gb);

                    copy_color_mask(dst + offset, c->width, mask, color);

                    break;

                case 5:

                    offset = 0;

                case 6:

                    set_src_position(c, &src, &send);

                case 7:

                    if (src + offset + c->width + 4 > send)

                        return AVERROR_INVALIDDATA;

                    mask = bytestream2_get_byte(&c->gb);

                    copy_src_mask(dst + offset, c->width, mask, src + offset);

                    break;

                }

            }

        }



    return 0;

}
