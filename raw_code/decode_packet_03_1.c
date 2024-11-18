static int decode_packet(J2kDecoderContext *s, J2kCodingStyle *codsty, J2kResLevel *rlevel, int precno,

                         int layno, uint8_t *expn, int numgbits)

{

    int bandno, cblkny, cblknx, cblkno, ret;



    if (!(ret = get_bits(s, 1))){

        j2k_flush(s);

        return 0;

    } else if (ret < 0)

        return ret;



    for (bandno = 0; bandno < rlevel->nbands; bandno++){

        J2kBand *band = rlevel->band + bandno;

        J2kPrec *prec = band->prec + precno;

        int pos = 0;



        if (band->coord[0][0] == band->coord[0][1]

        ||  band->coord[1][0] == band->coord[1][1])

            continue;



        for (cblkny = prec->yi0; cblkny < prec->yi1; cblkny++)

            for(cblknx = prec->xi0, cblkno = cblkny * band->cblknx + cblknx; cblknx < prec->xi1; cblknx++, cblkno++, pos++){

                J2kCblk *cblk = band->cblk + cblkno;

                int incl, newpasses, llen;



                if (cblk->npasses)

                    incl = get_bits(s, 1);

                else

                    incl = tag_tree_decode(s, prec->cblkincl + pos, layno+1) == layno;

                if (!incl)

                    continue;

                else if (incl < 0)

                    return incl;



                if (!cblk->npasses)

                    cblk->nonzerobits = expn[bandno] + numgbits - 1 - tag_tree_decode(s, prec->zerobits + pos, 100);

                if ((newpasses = getnpasses(s)) < 0)

                    return newpasses;

                if ((llen = getlblockinc(s)) < 0)

                    return llen;

                cblk->lblock += llen;

                if ((ret = get_bits(s, av_log2(newpasses) + cblk->lblock)) < 0)

                    return ret;

                cblk->lengthinc = ret;

                cblk->npasses += newpasses;

            }

    }

    j2k_flush(s);



    if (codsty->csty & J2K_CSTY_EPH) {

        if (AV_RB16(s->buf) == J2K_EPH) {

            s->buf += 2;

        } else {

            av_log(s->avctx, AV_LOG_ERROR, "EPH marker not found.\n");

        }

    }



    for (bandno = 0; bandno < rlevel->nbands; bandno++){

        J2kBand *band = rlevel->band + bandno;

        int yi, cblknw = band->prec[precno].xi1 - band->prec[precno].xi0;

        for (yi = band->prec[precno].yi0; yi < band->prec[precno].yi1; yi++){

            int xi;

            for (xi = band->prec[precno].xi0; xi < band->prec[precno].xi1; xi++){

                J2kCblk *cblk = band->cblk + yi * cblknw + xi;

                if (s->buf_end - s->buf < cblk->lengthinc)

                    return AVERROR(EINVAL);

                bytestream_get_buffer(&s->buf, cblk->data, cblk->lengthinc);

                cblk->length += cblk->lengthinc;

                cblk->lengthinc = 0;

            }

        }

    }

    return 0;

}
