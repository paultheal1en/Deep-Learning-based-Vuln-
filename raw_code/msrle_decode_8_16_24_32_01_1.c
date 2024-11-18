static int msrle_decode_8_16_24_32(AVCodecContext *avctx, AVPicture *pic,

                                   int depth, GetByteContext *gb)

{

    uint8_t *output, *output_end;

    int p1, p2, line=avctx->height - 1, pos=0, i;

    uint16_t pix16;

    uint32_t pix32;

    unsigned int width= FFABS(pic->linesize[0]) / (depth >> 3);



    output     = pic->data[0] + (avctx->height - 1) * pic->linesize[0];

    output_end = pic->data[0] +  avctx->height      * pic->linesize[0];

    while (bytestream2_get_bytes_left(gb) > 0) {

        p1 = bytestream2_get_byteu(gb);

        if(p1 == 0) { //Escape code

            p2 = bytestream2_get_byte(gb);

            if(p2 == 0) { //End-of-line

                output = pic->data[0] + (--line) * pic->linesize[0];

                if (line < 0) {

                    if (bytestream2_get_be16(gb) == 1) { // end-of-picture

                        return 0;

                    } else {

                        av_log(avctx, AV_LOG_ERROR,

                               "Next line is beyond picture bounds (%d bytes left)\n",

                               bytestream2_get_bytes_left(gb));

                        return AVERROR_INVALIDDATA;

                    }

                }

                pos = 0;

                continue;

            } else if(p2 == 1) { //End-of-picture

                return 0;

            } else if(p2 == 2) { //Skip

                p1 = bytestream2_get_byte(gb);

                p2 = bytestream2_get_byte(gb);

                line -= p2;

                pos += p1;

                if (line < 0 || pos >= width){

                    av_log(avctx, AV_LOG_ERROR, "Skip beyond picture bounds\n");

                    return -1;

                }

                output = pic->data[0] + line * pic->linesize[0] + pos * (depth >> 3);

                continue;

            }

            // Copy data

            if ((pic->linesize[0] > 0 && output + p2 * (depth >> 3) > output_end) ||

                (pic->linesize[0] < 0 && output + p2 * (depth >> 3) < output_end)) {

                bytestream2_skip(gb, 2 * (depth >> 3));

                continue;

            } else if (bytestream2_get_bytes_left(gb) < p2 * (depth >> 3)) {

                av_log(avctx, AV_LOG_ERROR, "bytestream overrun\n");

                return AVERROR_INVALIDDATA;

            }



            if ((depth == 8) || (depth == 24)) {

                for(i = 0; i < p2 * (depth >> 3); i++) {

                    *output++ = bytestream2_get_byteu(gb);

                }

                // RLE8 copy is actually padded - and runs are not!

                if(depth == 8 && (p2 & 1)) {

                    bytestream2_skip(gb, 1);

                }

            } else if (depth == 16) {

                for(i = 0; i < p2; i++) {

                    *(uint16_t*)output = bytestream2_get_le16u(gb);

                    output += 2;

                }

            } else if (depth == 32) {

                for(i = 0; i < p2; i++) {

                    *(uint32_t*)output = bytestream2_get_le32u(gb);

                    output += 4;

                }

            }

            pos += p2;

        } else { //run of pixels

            uint8_t pix[3]; //original pixel

            if ((pic->linesize[0] > 0 && output + p1 * (depth >> 3) > output_end) ||

                (pic->linesize[0] < 0 && output + p1 * (depth >> 3) < output_end))

                continue;

            switch(depth){

            case  8: pix[0] = bytestream2_get_byte(gb);

                     break;

            case 16: pix16  = bytestream2_get_le16(gb);

                     break;

            case 24: pix[0] = bytestream2_get_byte(gb);

                     pix[1] = bytestream2_get_byte(gb);

                     pix[2] = bytestream2_get_byte(gb);

                     break;

            case 32: pix32  = bytestream2_get_le32(gb);

                     break;

            }

            switch(depth){

            case  8:

                for(i = 0; i < p1; i++)

                        *output++ = pix[0];

                break;

            case 16:

                for(i = 0; i < p1; i++) {

                        *(uint16_t*)output = pix16;

                        output += 2;

                }

                break;

            case 24:

                for(i = 0; i < p1; i++) {

                        *output++ = pix[0];

                        *output++ = pix[1];

                        *output++ = pix[2];

                }

                break;

            case 32:

                for(i = 0; i < p1; i++) {

                        *(uint32_t*)output = pix32;

                        output += 4;

                }

                break;

            }

            pos += p1;

        }

    }



    av_log(avctx, AV_LOG_WARNING, "MS RLE warning: no end-of-picture code\n");

    return 0;

}
