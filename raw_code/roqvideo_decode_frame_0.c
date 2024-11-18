static void roqvideo_decode_frame(RoqContext *ri)

{

    unsigned int chunk_id = 0, chunk_arg = 0;

    unsigned long chunk_size = 0;

    int i, j, k, nv1, nv2, vqflg = 0, vqflg_pos = -1;

    int vqid, bpos, xpos, ypos, xp, yp, x, y, mx, my;

    int frame_stats[2][4] = {{0},{0}};

    roq_qcell *qcell;

    const unsigned char *buf = ri->buf;

    const unsigned char *buf_end = ri->buf + ri->size;



    while (buf < buf_end) {

        chunk_id = bytestream_get_le16(&buf);

        chunk_size = bytestream_get_le32(&buf);

        chunk_arg = bytestream_get_le16(&buf);



        if(chunk_id == RoQ_QUAD_VQ)

            break;

        if(chunk_id == RoQ_QUAD_CODEBOOK) {

            if((nv1 = chunk_arg >> 8) == 0)

                nv1 = 256;

            if((nv2 = chunk_arg & 0xff) == 0 && nv1 * 6 < chunk_size)

                nv2 = 256;

            for(i = 0; i < nv1; i++) {

                ri->cb2x2[i].y[0] = *buf++;

                ri->cb2x2[i].y[1] = *buf++;

                ri->cb2x2[i].y[2] = *buf++;

                ri->cb2x2[i].y[3] = *buf++;

                ri->cb2x2[i].u = *buf++;

                ri->cb2x2[i].v = *buf++;

            }

            for(i = 0; i < nv2; i++)

                for(j = 0; j < 4; j++)

                    ri->cb4x4[i].idx[j] = *buf++;

        }

    }



    bpos = xpos = ypos = 0;

    if (chunk_size > buf_end - buf) {

        av_log(ri->avctx, AV_LOG_ERROR, "Chunk does not fit in input buffer\n");

        chunk_size = buf_end - buf;

    }

    while(bpos < chunk_size) {

        for (yp = ypos; yp < ypos + 16; yp += 8)

            for (xp = xpos; xp < xpos + 16; xp += 8) {

                if (bpos >= chunk_size) {

                    av_log(ri->avctx, AV_LOG_ERROR, "Input buffer too small\n");

                    return;

                }

                if (vqflg_pos < 0) {

                    vqflg = buf[bpos++]; vqflg |= (buf[bpos++] << 8);

                    vqflg_pos = 7;

                }

                vqid = (vqflg >> (vqflg_pos * 2)) & 0x3;

                frame_stats[0][vqid]++;

                vqflg_pos--;



                switch(vqid) {

                case RoQ_ID_MOT:

                    break;

                case RoQ_ID_FCC:

                    mx = 8 - (buf[bpos] >> 4) - ((signed char) (chunk_arg >> 8));

                    my = 8 - (buf[bpos++] & 0xf) - ((signed char) chunk_arg);

                    ff_apply_motion_8x8(ri, xp, yp, mx, my);

                    break;

                case RoQ_ID_SLD:

                    qcell = ri->cb4x4 + buf[bpos++];

                    ff_apply_vector_4x4(ri, xp, yp, ri->cb2x2 + qcell->idx[0]);

                    ff_apply_vector_4x4(ri, xp+4, yp, ri->cb2x2 + qcell->idx[1]);

                    ff_apply_vector_4x4(ri, xp, yp+4, ri->cb2x2 + qcell->idx[2]);

                    ff_apply_vector_4x4(ri, xp+4, yp+4, ri->cb2x2 + qcell->idx[3]);

                    break;

                case RoQ_ID_CCC:

                    for (k = 0; k < 4; k++) {

                        x = xp; y = yp;

                        if(k & 0x01) x += 4;

                        if(k & 0x02) y += 4;



                        if (bpos >= chunk_size) {

                            av_log(ri->avctx, AV_LOG_ERROR, "Input buffer too small\n");

                            return;

                        }

                        if (vqflg_pos < 0) {

                            vqflg = buf[bpos++];

                            vqflg |= (buf[bpos++] << 8);

                            vqflg_pos = 7;

                        }

                        vqid = (vqflg >> (vqflg_pos * 2)) & 0x3;

                        frame_stats[1][vqid]++;

                        vqflg_pos--;

                        switch(vqid) {

                        case RoQ_ID_MOT:

                            break;

                        case RoQ_ID_FCC:

                            mx = 8 - (buf[bpos] >> 4) - ((signed char) (chunk_arg >> 8));

                            my = 8 - (buf[bpos++] & 0xf) - ((signed char) chunk_arg);

                            ff_apply_motion_4x4(ri, x, y, mx, my);

                            break;

                        case RoQ_ID_SLD:

                            qcell = ri->cb4x4 + buf[bpos++];

                            ff_apply_vector_2x2(ri, x, y, ri->cb2x2 + qcell->idx[0]);

                            ff_apply_vector_2x2(ri, x+2, y, ri->cb2x2 + qcell->idx[1]);

                            ff_apply_vector_2x2(ri, x, y+2, ri->cb2x2 + qcell->idx[2]);

                            ff_apply_vector_2x2(ri, x+2, y+2, ri->cb2x2 + qcell->idx[3]);

                            break;

                        case RoQ_ID_CCC:

                            ff_apply_vector_2x2(ri, x, y, ri->cb2x2 + buf[bpos]);

                            ff_apply_vector_2x2(ri, x+2, y, ri->cb2x2 + buf[bpos+1]);

                            ff_apply_vector_2x2(ri, x, y+2, ri->cb2x2 + buf[bpos+2]);

                            ff_apply_vector_2x2(ri, x+2, y+2, ri->cb2x2 + buf[bpos+3]);

                            bpos += 4;

                            break;

                        }

                    }

                    break;

                default:

                    av_log(ri->avctx, AV_LOG_ERROR, "Unknown vq code: %d\n", vqid);

            }

        }



        xpos += 16;

        if (xpos >= ri->width) {

            xpos -= ri->width;

            ypos += 16;

        }

        if(ypos >= ri->height)

            break;

    }

}
