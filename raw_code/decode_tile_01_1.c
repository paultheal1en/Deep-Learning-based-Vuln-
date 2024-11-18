static int decode_tile(J2kDecoderContext *s, J2kTile *tile)

{

    int compno, reslevelno, bandno;

    int x, y, *src[4];

    uint8_t *line;

    J2kT1Context t1;



    for (compno = 0; compno < s->ncomponents; compno++){

        J2kComponent *comp = tile->comp + compno;

        J2kCodingStyle *codsty = tile->codsty + compno;



        for (reslevelno = 0; reslevelno < codsty->nreslevels; reslevelno++){

            J2kResLevel *rlevel = comp->reslevel + reslevelno;

            for (bandno = 0; bandno < rlevel->nbands; bandno++){

                J2kBand *band = rlevel->band + bandno;

                int cblkx, cblky, cblkno=0, xx0, x0, xx1, y0, yy0, yy1, bandpos;



                bandpos = bandno + (reslevelno > 0);



                yy0 = bandno == 0 ? 0 : comp->reslevel[reslevelno-1].coord[1][1] - comp->reslevel[reslevelno-1].coord[1][0];

                y0 = yy0;

                yy1 = FFMIN(ff_j2k_ceildiv(band->coord[1][0] + 1, band->codeblock_height) * band->codeblock_height,

                            band->coord[1][1]) - band->coord[1][0] + yy0;



                if (band->coord[0][0] == band->coord[0][1] || band->coord[1][0] == band->coord[1][1])

                    continue;



                for (cblky = 0; cblky < band->cblkny; cblky++){

                    if (reslevelno == 0 || bandno == 1)

                        xx0 = 0;

                    else

                        xx0 = comp->reslevel[reslevelno-1].coord[0][1] - comp->reslevel[reslevelno-1].coord[0][0];

                    x0 = xx0;

                    xx1 = FFMIN(ff_j2k_ceildiv(band->coord[0][0] + 1, band->codeblock_width) * band->codeblock_width,

                                band->coord[0][1]) - band->coord[0][0] + xx0;



                    for (cblkx = 0; cblkx < band->cblknx; cblkx++, cblkno++){

                        int y, x;

                        decode_cblk(s, codsty, &t1, band->cblk + cblkno, xx1 - xx0, yy1 - yy0, bandpos);

                        if (codsty->transform == FF_DWT53){

                            for (y = yy0; y < yy1; y+=s->cdy[compno]){

                                int *ptr = t1.data[y-yy0];

                                for (x = xx0; x < xx1; x+=s->cdx[compno]){

                                    comp->data[(comp->coord[0][1] - comp->coord[0][0]) * y + x] = *ptr++ >> 1;

                                }

                            }

                        } else{

                            for (y = yy0; y < yy1; y+=s->cdy[compno]){

                                int *ptr = t1.data[y-yy0];

                                for (x = xx0; x < xx1; x+=s->cdx[compno]){

                                    int tmp = ((int64_t)*ptr++) * ((int64_t)band->stepsize) >> 13, tmp2;

                                    tmp2 = FFABS(tmp>>1) + FFABS(tmp&1);

                                    comp->data[(comp->coord[0][1] - comp->coord[0][0]) * y + x] = tmp < 0 ? -tmp2 : tmp2;

                                }

                            }

                        }

                        xx0 = xx1;

                        xx1 = FFMIN(xx1 + band->codeblock_width, band->coord[0][1] - band->coord[0][0] + x0);

                    }

                    yy0 = yy1;

                    yy1 = FFMIN(yy1 + band->codeblock_height, band->coord[1][1] - band->coord[1][0] + y0);

                }

            }

        }

        ff_j2k_dwt_decode(&comp->dwt, comp->data);

        src[compno] = comp->data;

    }

    if (tile->codsty[0].mct)

        mct_decode(s, tile);



    if (s->avctx->pix_fmt == PIX_FMT_BGRA) // RGBA -> BGRA

        FFSWAP(int *, src[0], src[2]);



    if (s->precision <= 8) {

        for (compno = 0; compno < s->ncomponents; compno++){

            y = tile->comp[compno].coord[1][0] - s->image_offset_y;

            line = s->picture.data[0] + y * s->picture.linesize[0];

            for (; y < tile->comp[compno].coord[1][1] - s->image_offset_y; y += s->cdy[compno]){

                uint8_t *dst;



                x = tile->comp[compno].coord[0][0] - s->image_offset_x;

                dst = line + x * s->ncomponents + compno;



                for (; x < tile->comp[compno].coord[0][1] - s->image_offset_x; x += s->cdx[compno]) {

                    *src[compno] += 1 << (s->cbps[compno]-1);

                    if (*src[compno] < 0)

                        *src[compno] = 0;

                    else if (*src[compno] >= (1 << s->cbps[compno]))

                        *src[compno] = (1 << s->cbps[compno]) - 1;

                    *dst = *src[compno]++;

                    dst += s->ncomponents;

                }

                line += s->picture.linesize[0];

            }

        }

    } else {

        for (compno = 0; compno < s->ncomponents; compno++) {

            y = tile->comp[compno].coord[1][0] - s->image_offset_y;

            line = s->picture.data[0] + y * s->picture.linesize[0];

            for (; y < tile->comp[compno].coord[1][1] - s->image_offset_y; y += s->cdy[compno]) {

                uint16_t *dst;

                x = tile->comp[compno].coord[0][0] - s->image_offset_x;

                dst = line + (x * s->ncomponents + compno) * 2;

                for (; x < tile->comp[compno].coord[0][1] - s->image_offset_x; x += s-> cdx[compno]) {

                    int32_t val;

                    val = *src[compno]++ << (16 - s->cbps[compno]);

                    val += 1 << 15;

                    val = av_clip(val, 0, (1 << 16) - 1);

                    *dst = val;

                    dst += s->ncomponents;

                }

                line += s->picture.linesize[0];

            }

        }

    }

    return 0;

}
