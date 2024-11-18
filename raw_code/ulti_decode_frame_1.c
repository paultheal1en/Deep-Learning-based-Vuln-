static int ulti_decode_frame(AVCodecContext *avctx,

                             void *data, int *data_size,

                             AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

    UltimotionDecodeContext *s=avctx->priv_data;

    int modifier = 0;

    int uniq = 0;

    int mode = 0;

    int blocks = 0;

    int done = 0;

    int x = 0, y = 0;

    int i;

    int skip;

    int tmp;



    s->frame.reference = 1;

    s->frame.buffer_hints = FF_BUFFER_HINTS_VALID | FF_BUFFER_HINTS_PRESERVE | FF_BUFFER_HINTS_REUSABLE;

    if (avctx->reget_buffer(avctx, &s->frame) < 0) {

        av_log(avctx, AV_LOG_ERROR, "reget_buffer() failed\n");

        return -1;

    }



    while(!done) {

        int idx;

        if(blocks >= s->blocks || y >= s->height)

            break;//all blocks decoded



        idx = *buf++;

        if((idx & 0xF8) == 0x70) {

            switch(idx) {

            case 0x70: //change modifier

                modifier = *buf++;

                if(modifier>1)

                    av_log(avctx, AV_LOG_INFO, "warning: modifier must be 0 or 1, got %i\n", modifier);

                break;

            case 0x71: // set uniq flag

                uniq = 1;

                break;

            case 0x72: //toggle mode

                mode = !mode;

                break;

            case 0x73: //end-of-frame

                done = 1;

                break;

            case 0x74: //skip some blocks

                skip = *buf++;

                if ((blocks + skip) >= s->blocks)

                    break;

                blocks += skip;

                x += skip * 8;

                while(x >= s->width) {

                    x -= s->width;

                    y += 8;

                }

                break;

            default:

                av_log(avctx, AV_LOG_INFO, "warning: unknown escape 0x%02X\n", idx);

            }

        } else { //handle one block

            int code;

            int cf;

            int angle = 0;

            uint8_t Y[4]; // luma samples of block

            int tx = 0, ty = 0; //coords of subblock

            int chroma = 0;

            if (mode || uniq) {

                uniq = 0;

                cf = 1;

                chroma = 0;

            } else {

                cf = 0;

                if (idx)

                    chroma = *buf++;

            }

            for (i = 0; i < 4; i++) { // for every subblock

                code = (idx >> (6 - i*2)) & 3; //extract 2 bits

                if(!code) //skip subblock

                    continue;

                if(cf)

                    chroma = *buf++;

                tx = x + block_coords[i * 2];

                ty = y + block_coords[(i * 2) + 1];

                switch(code) {

                case 1:

                    tmp = *buf++;



                    angle = angle_by_index[(tmp >> 6) & 0x3];



                    Y[0] = tmp & 0x3F;

                    Y[1] = Y[0];



                    if (angle) {

                        Y[2] = Y[0]+1;

                        if (Y[2] > 0x3F)

                            Y[2] = 0x3F;

                        Y[3] = Y[2];

                    } else {

                        Y[2] = Y[0];

                        Y[3] = Y[0];

                    }

                    break;



                case 2:

                    if (modifier) { // unpack four luma samples

                        tmp = bytestream_get_be24(&buf);



                        Y[0] = (tmp >> 18) & 0x3F;

                        Y[1] = (tmp >> 12) & 0x3F;

                        Y[2] = (tmp >> 6) & 0x3F;

                        Y[3] = tmp & 0x3F;

                        angle = 16;

                    } else { // retrieve luma samples from codebook

                        tmp = bytestream_get_be16(&buf);



                        angle = (tmp >> 12) & 0xF;

                        tmp &= 0xFFF;

                        tmp <<= 2;

                        Y[0] = s->ulti_codebook[tmp];

                        Y[1] = s->ulti_codebook[tmp + 1];

                        Y[2] = s->ulti_codebook[tmp + 2];

                        Y[3] = s->ulti_codebook[tmp + 3];

                    }

                    break;



                case 3:

                    if (modifier) { // all 16 luma samples

                        uint8_t Luma[16];



                        tmp = bytestream_get_be24(&buf);

                        Luma[0] = (tmp >> 18) & 0x3F;

                        Luma[1] = (tmp >> 12) & 0x3F;

                        Luma[2] = (tmp >> 6) & 0x3F;

                        Luma[3] = tmp & 0x3F;



                        tmp = bytestream_get_be24(&buf);

                        Luma[4] = (tmp >> 18) & 0x3F;

                        Luma[5] = (tmp >> 12) & 0x3F;

                        Luma[6] = (tmp >> 6) & 0x3F;

                        Luma[7] = tmp & 0x3F;



                        tmp = bytestream_get_be24(&buf);

                        Luma[8] = (tmp >> 18) & 0x3F;

                        Luma[9] = (tmp >> 12) & 0x3F;

                        Luma[10] = (tmp >> 6) & 0x3F;

                        Luma[11] = tmp & 0x3F;



                        tmp = bytestream_get_be24(&buf);

                        Luma[12] = (tmp >> 18) & 0x3F;

                        Luma[13] = (tmp >> 12) & 0x3F;

                        Luma[14] = (tmp >> 6) & 0x3F;

                        Luma[15] = tmp & 0x3F;



                        ulti_convert_yuv(&s->frame, tx, ty, Luma, chroma);

                    } else {

                        tmp = *buf++;

                        if(tmp & 0x80) {

                            angle = (tmp >> 4) & 0x7;

                            tmp = (tmp << 8) + *buf++;

                            Y[0] = (tmp >> 6) & 0x3F;

                            Y[1] = tmp & 0x3F;

                            Y[2] = (*buf++) & 0x3F;

                            Y[3] = (*buf++) & 0x3F;

                            ulti_grad(&s->frame, tx, ty, Y, chroma, angle); //draw block

                        } else { // some patterns

                            int f0, f1;

                            f0 = *buf++;

                            f1 = tmp;

                            Y[0] = (*buf++) & 0x3F;

                            Y[1] = (*buf++) & 0x3F;

                            ulti_pattern(&s->frame, tx, ty, f1, f0, Y[0], Y[1], chroma);

                        }

                    }

                    break;

                }

                if(code != 3)

                    ulti_grad(&s->frame, tx, ty, Y, chroma, angle); // draw block

            }

            blocks++;

                x += 8;

            if(x >= s->width) {

                x = 0;

                y += 8;

            }

        }

    }



    *data_size=sizeof(AVFrame);

    *(AVFrame*)data= s->frame;



    return buf_size;

}
