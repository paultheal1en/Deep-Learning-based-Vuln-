static int encode_packet(Jpeg2000EncoderContext *s, Jpeg2000ResLevel *rlevel, int precno,

                          uint8_t *expn, int numgbits)

{

    int bandno, empty = 1;



    // init bitstream

    *s->buf = 0;

    s->bit_index = 0;



    // header



    // is the packet empty?

    for (bandno = 0; bandno < rlevel->nbands; bandno++){

        if (rlevel->band[bandno].coord[0][0] < rlevel->band[bandno].coord[0][1]

        &&  rlevel->band[bandno].coord[1][0] < rlevel->band[bandno].coord[1][1]){

            empty = 0;

            break;

        }

    }



    put_bits(s, !empty, 1);

    if (empty){

        j2k_flush(s);

        return 0;

    }



    for (bandno = 0; bandno < rlevel->nbands; bandno++){

        Jpeg2000Band *band = rlevel->band + bandno;

        Jpeg2000Prec *prec = band->prec + precno;

        int yi, xi, pos;

        int cblknw = prec->nb_codeblocks_width;



        if (band->coord[0][0] == band->coord[0][1]

        ||  band->coord[1][0] == band->coord[1][1])

            continue;



        for (pos=0, yi = 0; yi < prec->nb_codeblocks_height; yi++){

            for (xi = 0; xi < cblknw; xi++, pos++){

                prec->cblkincl[pos].val = prec->cblk[yi * cblknw + xi].ninclpasses == 0;

                tag_tree_update(prec->cblkincl + pos);

                prec->zerobits[pos].val = expn[bandno] + numgbits - 1 - prec->cblk[yi * cblknw + xi].nonzerobits;

                tag_tree_update(prec->zerobits + pos);

            }

        }



        for (pos=0, yi = 0; yi < prec->nb_codeblocks_height; yi++){

            for (xi = 0; xi < cblknw; xi++, pos++){

                int pad = 0, llen, length;

                Jpeg2000Cblk *cblk = prec->cblk + yi * cblknw + xi;



                if (s->buf_end - s->buf < 20) // approximately

                    return -1;



                // inclusion information

                tag_tree_code(s, prec->cblkincl + pos, 1);

                if (!cblk->ninclpasses)

                    continue;

                // zerobits information

                tag_tree_code(s, prec->zerobits + pos, 100);

                // number of passes

                putnumpasses(s, cblk->ninclpasses);



                length = cblk->passes[cblk->ninclpasses-1].rate;

                llen = av_log2(length) - av_log2(cblk->ninclpasses) - 2;

                if (llen < 0){

                    pad = -llen;

                    llen = 0;

                }

                // length of code block

                put_bits(s, 1, llen);

                put_bits(s, 0, 1);

                put_num(s, length, av_log2(length)+1+pad);

            }

        }

    }

    j2k_flush(s);

    for (bandno = 0; bandno < rlevel->nbands; bandno++){

        Jpeg2000Band *band = rlevel->band + bandno;

        Jpeg2000Prec *prec = band->prec + precno;

        int yi, cblknw = prec->nb_codeblocks_width;

        for (yi =0; yi < prec->nb_codeblocks_height; yi++){

            int xi;

            for (xi = 0; xi < cblknw; xi++){

                Jpeg2000Cblk *cblk = prec->cblk + yi * cblknw + xi;

                if (cblk->ninclpasses){

                    if (s->buf_end - s->buf < cblk->passes[cblk->ninclpasses-1].rate)

                        return -1;

                    bytestream_put_buffer(&s->buf, cblk->data,   cblk->passes[cblk->ninclpasses-1].rate

                                                               - cblk->passes[cblk->ninclpasses-1].flushed_len);

                    bytestream_put_buffer(&s->buf, cblk->passes[cblk->ninclpasses-1].flushed,

                                                   cblk->passes[cblk->ninclpasses-1].flushed_len);

                }

            }

        }

    }

    return 0;

}
