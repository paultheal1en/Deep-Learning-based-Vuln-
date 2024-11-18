static int jpeg2000_decode_packets(Jpeg2000DecoderContext *s, Jpeg2000Tile *tile)

{

    int layno, reslevelno, compno, precno, ok_reslevel, ret;

    uint8_t prog_order = tile->codsty[0].prog_order;

    uint16_t x;

    uint16_t y;



    s->bit_index = 8;

    switch (prog_order) {

    case JPEG2000_PGOD_LRCP:

        for (layno = 0; layno < tile->codsty[0].nlayers; layno++) {

            ok_reslevel = 1;

            for (reslevelno = 0; ok_reslevel; reslevelno++) {

                ok_reslevel = 0;

                for (compno = 0; compno < s->ncomponents; compno++) {

                    Jpeg2000CodingStyle *codsty = tile->codsty + compno;

                    Jpeg2000QuantStyle *qntsty  = tile->qntsty + compno;

                    if (reslevelno < codsty->nreslevels) {

                        Jpeg2000ResLevel *rlevel = tile->comp[compno].reslevel +

                                                   reslevelno;

                        ok_reslevel = 1;

                        for (precno = 0; precno < rlevel->num_precincts_x * rlevel->num_precincts_y; precno++)

                            if ((ret = jpeg2000_decode_packet(s,

                                                              codsty, rlevel,

                                                              precno, layno,

                                                              qntsty->expn + (reslevelno ? 3 * (reslevelno - 1) + 1 : 0),

                                                              qntsty->nguardbits)) < 0)

                                return ret;

                    }

                }

            }

        }

        break;



    case JPEG2000_PGOD_CPRL:

        for (compno = 0; compno < s->ncomponents; compno++) {

            Jpeg2000CodingStyle *codsty = tile->codsty + compno;

            Jpeg2000QuantStyle *qntsty  = tile->qntsty + compno;



            /* Set bit stream buffer address according to tile-part.

             * For DCinema one tile-part per component, so can be

             * indexed by component. */

            s->buf = tile->tile_part[compno].tp_start_bstrm;



            /* Position loop (y axis)

             * TODO: Automate computing of step 256.

             * Fixed here, but to be computed before entering here. */

            for (y = 0; y < s->height; y += 256) {

                /* Position loop (y axis)

                 * TODO: automate computing of step 256.

                 * Fixed here, but to be computed before entering here. */

                for (x = 0; x < s->width; x += 256) {

                    for (reslevelno = 0; reslevelno < codsty->nreslevels; reslevelno++) {

                        uint16_t prcx, prcy;

                        uint8_t reducedresno = codsty->nreslevels - 1 -reslevelno; //  ==> N_L - r

                        Jpeg2000ResLevel *rlevel = tile->comp[compno].reslevel + reslevelno;



                        if (!((y % (1 << (rlevel->log2_prec_height + reducedresno)) == 0) ||

                              (y == 0))) // TODO: 2nd condition simplified as try0 always =0 for dcinema

                            continue;



                        if (!((x % (1 << (rlevel->log2_prec_width + reducedresno)) == 0) ||

                              (x == 0))) // TODO: 2nd condition simplified as try0 always =0 for dcinema

                            continue;



                        // check if a precinct exists

                        prcx   = ff_jpeg2000_ceildivpow2(x, reducedresno) >> rlevel->log2_prec_width;

                        prcy   = ff_jpeg2000_ceildivpow2(y, reducedresno) >> rlevel->log2_prec_height;

                        precno = prcx + rlevel->num_precincts_x * prcy;

                        for (layno = 0; layno < tile->codsty[0].nlayers; layno++) {

                            if ((ret = jpeg2000_decode_packet(s, codsty, rlevel,

                                                              precno, layno,

                                                              qntsty->expn + (reslevelno ? 3 * (reslevelno - 1) + 1 : 0),

                                                              qntsty->nguardbits)) < 0)

                                return ret;

                        }

                    }

                }

            }

        }

        break;



    default:

        break;

    }



    /* EOC marker reached */

    s->buf += 2;



    return 0;

}
