static void decode_interframe_v4a(AVCodecContext *avctx, uint8_t *src,

                                  uint32_t size)

{

    Hnm4VideoContext *hnm = avctx->priv_data;

    GetByteContext gb;

    uint32_t writeoffset = 0, offset;

    uint8_t tag, count, previous, delta;



    bytestream2_init(&gb, src, size);



    while (bytestream2_tell(&gb) < size) {

        count = bytestream2_peek_byte(&gb) & 0x3F;

        if (count == 0) {

            tag = bytestream2_get_byte(&gb) & 0xC0;

            tag = tag >> 6;

            if (tag == 0) {

                writeoffset += bytestream2_get_byte(&gb);

            } else if (tag == 1) {

                if (writeoffset + hnm->width >= hnm->width * hnm->height) {

                    av_log(avctx, AV_LOG_ERROR, "writeoffset out of bounds\n");

                    break;

                }

                hnm->current[writeoffset]              = bytestream2_get_byte(&gb);

                hnm->current[writeoffset + hnm->width] = bytestream2_get_byte(&gb);

                writeoffset++;

            } else if (tag == 2) {

                writeoffset += hnm->width;

            } else if (tag == 3) {

                break;

            }

            if (writeoffset > hnm->width * hnm->height) {

                av_log(avctx, AV_LOG_ERROR, "writeoffset out of bounds\n");

                break;

            }

        } else {

            delta    = bytestream2_peek_byte(&gb) & 0x80;

            previous = bytestream2_peek_byte(&gb) & 0x40;

            bytestream2_skip(&gb, 1);



            offset  = writeoffset;

            offset += bytestream2_get_le16(&gb);



            if (delta)

                offset -= 0x10000;



            if (offset + hnm->width + count >= hnm->width * hnm->height) {

                av_log(avctx, AV_LOG_ERROR, "Attempting to read out of bounds\n");

                break;

            } else if (writeoffset + hnm->width + count >= hnm->width * hnm->height) {

                av_log(avctx, AV_LOG_ERROR, "Attempting to write out of bounds\n");

                break;

            }



            if (previous) {

                while (count > 0) {

                    hnm->current[writeoffset]              = hnm->previous[offset];

                    hnm->current[writeoffset + hnm->width] = hnm->previous[offset + hnm->width];

                    writeoffset++;

                    offset++;

                    count--;

                }

            } else {

                while (count > 0) {

                    hnm->current[writeoffset]              = hnm->current[offset];

                    hnm->current[writeoffset + hnm->width] = hnm->current[offset + hnm->width];

                    writeoffset++;

                    offset++;

                    count--;

                }

            }

        }

    }

}
