static void decode_interframe_v4(AVCodecContext *avctx, uint8_t *src, uint32_t size)

{

    Hnm4VideoContext *hnm = avctx->priv_data;

    GetByteContext gb;

    uint32_t writeoffset = 0, count, left, offset;

    uint8_t tag, previous, backline, backward, swap;



    bytestream2_init(&gb, src, size);



    while (bytestream2_tell(&gb) < size) {

        count = bytestream2_peek_byte(&gb) & 0x1F;

        if (count == 0) {

            tag = bytestream2_get_byte(&gb) & 0xE0;

            tag = tag >> 5;

            if (tag == 0) {

                hnm->current[writeoffset++] = bytestream2_get_byte(&gb);

                hnm->current[writeoffset++] = bytestream2_get_byte(&gb);

            } else if (tag == 1) {

                writeoffset += bytestream2_get_byte(&gb) * 2;

            } else if (tag == 2) {

                count = bytestream2_get_le16(&gb);

                count *= 2;

                writeoffset += count;

            } else if (tag == 3) {

                count = bytestream2_get_byte(&gb) * 2;

                while (count > 0) {

                    hnm->current[writeoffset++] = bytestream2_peek_byte(&gb);

                    count--;

                }

                bytestream2_skip(&gb, 1);

            } else {

                break;

            }

        } else {

            previous = bytestream2_peek_byte(&gb) & 0x20;

            backline = bytestream2_peek_byte(&gb) & 0x40;

            backward = bytestream2_peek_byte(&gb) & 0x80;

            bytestream2_skip(&gb, 1);

            swap   = bytestream2_peek_byte(&gb) & 0x01;

            offset = bytestream2_get_le16(&gb);

            offset = (offset >> 1) & 0x7FFF;

            offset = writeoffset + (offset * 2) - 0x8000;



            left = count;



            if (!backward && offset + count >= hnm->width * hnm->height) {

                av_log(avctx, AV_LOG_ERROR, "Attempting to read out of bounds");

                break;

            } else if (backward && offset >= hnm->width * hnm->height) {

                av_log(avctx, AV_LOG_ERROR, "Attempting to read out of bounds");

                break;

            } else if (writeoffset + count >= hnm->width * hnm->height) {

                av_log(avctx, AV_LOG_ERROR,

                       "Attempting to write out of bounds");

                break;

            }



            if (previous) {

                while (left > 0) {

                    if (backline) {

                        hnm->current[writeoffset++] = hnm->previous[offset - (2 * hnm->width) + 1];

                        hnm->current[writeoffset++] = hnm->previous[offset++];

                        offset++;

                    } else {

                        hnm->current[writeoffset++] = hnm->previous[offset++];

                        hnm->current[writeoffset++] = hnm->previous[offset++];

                    }

                    if (backward)

                        offset -= 4;

                    left--;

                }

            } else {

                while (left > 0) {

                    if (backline) {

                        hnm->current[writeoffset++] = hnm->current[offset - (2 * hnm->width) + 1];

                        hnm->current[writeoffset++] = hnm->current[offset++];

                        offset++;

                    } else {

                        hnm->current[writeoffset++] = hnm->current[offset++];

                        hnm->current[writeoffset++] = hnm->current[offset++];

                    }

                    if (backward)

                        offset -= 4;

                    left--;

                }

            }



            if (swap) {

                left         = count;

                writeoffset -= count * 2;

                while (left > 0) {

                    swap = hnm->current[writeoffset];

                    hnm->current[writeoffset] = hnm->current[writeoffset + 1];

                    hnm->current[writeoffset + 1] = swap;

                    left--;

                    writeoffset += 2;

                }

            }

        }

    }

}
