static int ac3_parse_audio_block(AC3DecodeContext * ctx, int index)

{

    ac3_audio_block *ab = &ctx->audio_block;

    int nfchans = ctx->bsi.nfchans;

    int acmod = ctx->bsi.acmod;

    int i, bnd, rbnd, grp, seg;

    GetBitContext *gb = &ctx->gb;

    uint32_t *flags = &ab->flags;

    int bit_alloc_flags = 0;

    float drange;



    *flags = 0;

    ab->blksw = 0;

    for (i = 0; i < 5; i++)

        ab->chcoeffs[i] = 1.0;

    for (i = 0; i < nfchans; i++) /*block switch flag */

        ab->blksw |= get_bits(gb, 1) << i;

    ab->dithflag = 0;

    for (i = 0; i < nfchans; i++) /* dithering flag */

        ab->dithflag |= get_bits(gb, 1) << i;

    if (get_bits(gb, 1)) { /* dynamic range */

        *flags |= AC3_AB_DYNRNGE;

        ab->dynrng = get_bits(gb, 8);

        drange = ((((ab->dynrng & 0x1f) | 0x20) << 13) * scale_factors[3 - (ab->dynrng >> 5)]);

        for (i = 0; i < nfchans; i++)

            ab->chcoeffs[i] *= drange;

    }

    if (acmod == 0x00) { /* dynamic range 1+1 mode */

        if (get_bits(gb, 1)) {

            *flags |= AC3_AB_DYNRNG2E;

            ab->dynrng2 = get_bits(gb, 8);

            drange = ((((ab->dynrng2 & 0x1f) | 0x20) << 13) * scale_factors[3 - (ab->dynrng2 >> 5)]);

            ab->chcoeffs[1] *= drange;

        }

    }

    get_downmix_coeffs(ctx);

    ab->chincpl = 0;

    if (get_bits(gb, 1)) { /* coupling strategy */

        *flags |= AC3_AB_CPLSTRE;

        ab->cplbndstrc = 0;

        if (get_bits(gb, 1)) { /* coupling in use */

            *flags |= AC3_AB_CPLINU;

            for (i = 0; i < nfchans; i++)

                ab->chincpl |= get_bits(gb, 1) << i;

            if (acmod == 0x02)

                if (get_bits(gb, 1)) /* phase flag in use */

                    *flags |= AC3_AB_PHSFLGINU;

            ab->cplbegf = get_bits(gb, 4);

            ab->cplendf = get_bits(gb, 4);

            assert((ab->ncplsubnd = 3 + ab->cplendf - ab->cplbegf) > 0);

            ab->ncplbnd = ab->ncplsubnd;

            for (i = 0; i < ab->ncplsubnd - 1; i++) /* coupling band structure */

                if (get_bits(gb, 1)) {

                    ab->cplbndstrc |= 1 << i;

                    ab->ncplbnd--;

                }

        }

    }

    if (*flags & AC3_AB_CPLINU) {

        ab->cplcoe = 0;

        for (i = 0; i < nfchans; i++)

            if (ab->chincpl & (1 << i))

                if (get_bits(gb, 1)) { /* coupling co-ordinates */

                    ab->cplcoe |= 1 << i;

                    ab->mstrcplco[i] = get_bits(gb, 2);

                    for (bnd = 0; bnd < ab->ncplbnd; bnd++) {

                        ab->cplcoexp[i][bnd] = get_bits(gb, 4);

                        ab->cplcomant[i][bnd] = get_bits(gb, 4);

                    }

                }

    }

    ab->phsflg = 0;

    if ((acmod == 0x02) && (*flags & AC3_AB_PHSFLGINU) && (ab->cplcoe & 1 || ab->cplcoe & (1 << 1))) {

        for (bnd = 0; bnd < ab->ncplbnd; bnd++)

            if (get_bits(gb, 1))

                ab->phsflg |= 1 << bnd;

    }

    generate_coupling_coordinates(ctx);

    ab->rematflg = 0;

    if (acmod == 0x02) /* rematrixing */

        if (get_bits(gb, 1)) {

            *flags |= AC3_AB_REMATSTR;

            if (ab->cplbegf > 2 || !(*flags & AC3_AB_CPLINU))

                for (rbnd = 0; rbnd < 4; rbnd++)

                    ab->rematflg |= get_bits(gb, 1) << bnd;

            else if (ab->cplbegf > 0 && ab->cplbegf <= 2 && *flags & AC3_AB_CPLINU)

                for (rbnd = 0; rbnd < 3; rbnd++)

                    ab->rematflg |= get_bits(gb, 1) << bnd;

            else if (!(ab->cplbegf) && *flags & AC3_AB_CPLINU)

                for (rbnd = 0; rbnd < 2; rbnd++)

                    ab->rematflg |= get_bits(gb, 1) << bnd;

        }

    if (*flags & AC3_AB_CPLINU) /* coupling exponent strategy */

        ab->cplexpstr = get_bits(gb, 2);

    for (i = 0; i < nfchans; i++) /* channel exponent strategy */

        ab->chexpstr[i] = get_bits(gb, 2);

    if (ctx->bsi.flags & AC3_BSI_LFEON) /* lfe exponent strategy */

        ab->lfeexpstr = get_bits(gb, 1);

    for (i = 0; i < nfchans; i++) /* channel bandwidth code */

        if (ab->chexpstr[i] != AC3_EXPSTR_REUSE)

            if (!(ab->chincpl & (1 << i))) {

                ab->chbwcod[i] = get_bits(gb, 6);

                assert (ab->chbwcod[i] <= 60);

            }

    if (*flags & AC3_AB_CPLINU)

        if (ab->cplexpstr != AC3_EXPSTR_REUSE) {/* coupling exponents */

            bit_alloc_flags |= 64;

            ab->cplabsexp = get_bits(gb, 4) << 1;

            ab->cplstrtmant = (ab->cplbegf * 12) + 37;

            ab->cplendmant = ((ab->cplendmant + 3) * 12) + 37;

            ab->ncplgrps = (ab->cplendmant - ab->cplstrtmant) / (3 << (ab->cplexpstr - 1));

            for (grp = 0; grp < ab->ncplgrps; grp++)

                ab->cplexps[grp] = get_bits(gb, 7);

        }

    for (i = 0; i < nfchans; i++) /* fbw channel exponents */

        if (ab->chexpstr[i] != AC3_EXPSTR_REUSE) {

            bit_alloc_flags |= 1 << i;

            if (ab->chincpl & (1 << i))

                ab->endmant[i] = (ab->cplbegf * 12) + 37;

            else

                ab->endmant[i] = ((ab->chbwcod[i] + 3) * 12) + 37;

            ab->nchgrps[i] =

                (ab->endmant[i] + (3 << (ab->chexpstr[i] - 1)) - 4) / (3 << (ab->chexpstr[i] - 1));

            ab->exps[i][0] = ab->dexps[i][0] = get_bits(gb, 4);

            for (grp = 1; grp <= ab->nchgrps[i]; grp++)

                ab->exps[i][grp] = get_bits(gb, 7);

            ab->gainrng[i] = get_bits(gb, 2);

        }

    if (ctx->bsi.flags & AC3_BSI_LFEON) /* lfe exponents */

        if (ab->lfeexpstr != AC3_EXPSTR_REUSE) {

            bit_alloc_flags |= 32;

            ab->lfeexps[0] = ab->dlfeexps[0] = get_bits(gb, 4);

            ab->lfeexps[1] = get_bits(gb, 7);

            ab->lfeexps[2] = get_bits(gb, 7);

        }

    if (decode_exponents(ctx)) {/* decode the exponents for this block */

        av_log(NULL, AV_LOG_ERROR, "Error parsing exponents\n");

        return -1;

    }



    if (get_bits(gb, 1)) { /* bit allocation information */

        *flags |= AC3_AB_BAIE;

        bit_alloc_flags |= 127;

        ab->sdcycod = get_bits(gb, 2);

        ab->fdcycod = get_bits(gb, 2);

        ab->sgaincod = get_bits(gb, 2);

        ab->dbpbcod = get_bits(gb, 2);

        ab->floorcod = get_bits(gb, 3);

    }

    if (get_bits(gb, 1)) { /* snroffset */

        *flags |= AC3_AB_SNROFFSTE;

        bit_alloc_flags |= 127;

        ab->csnroffst = get_bits(gb, 6);

        if (*flags & AC3_AB_CPLINU) { /* couling fine snr offset and fast gain code */

            ab->cplfsnroffst = get_bits(gb, 4);

            ab->cplfgaincod = get_bits(gb, 3);

        }

        for (i = 0; i < nfchans; i++) { /* channel fine snr offset and fast gain code */

            ab->fsnroffst[i] = get_bits(gb, 4);

            ab->fgaincod[i] = get_bits(gb, 3);

        }

        if (ctx->bsi.flags & AC3_BSI_LFEON) { /* lfe fine snr offset and fast gain code */

            ab->lfefsnroffst = get_bits(gb, 4);

            ab->lfefgaincod = get_bits(gb, 3);

        }

    }

    if (*flags & AC3_AB_CPLINU)

        if (get_bits(gb, 1)) { /* coupling leak information */

            bit_alloc_flags |= 64;

            *flags |= AC3_AB_CPLLEAKE;

            ab->cplfleak = get_bits(gb, 3);

            ab->cplsleak = get_bits(gb, 3);

        }

    if (get_bits(gb, 1)) { /* delta bit allocation information */

        *flags |= AC3_AB_DELTBAIE;

        bit_alloc_flags |= 127;

        if (*flags & AC3_AB_CPLINU) {

            ab->cpldeltbae = get_bits(gb, 2);

            if (ab->cpldeltbae == AC3_DBASTR_RESERVED) {

                av_log(NULL, AV_LOG_ERROR, "coupling delta bit allocation strategy reserved\n");

                return -1;

            }

        }

        for (i = 0; i < nfchans; i++) {

            ab->deltbae[i] = get_bits(gb, 2);

            if (ab->deltbae[i] == AC3_DBASTR_RESERVED) {

                av_log(NULL, AV_LOG_ERROR, "delta bit allocation strategy reserved\n");

                return -1;

            }

        }

        if (*flags & AC3_AB_CPLINU)

            if (ab->cpldeltbae == AC3_DBASTR_NEW) { /*coupling delta offset, len and bit allocation */

                ab->cpldeltnseg = get_bits(gb, 3);

                for (seg = 0; seg <= ab->cpldeltnseg; seg++) {

                    ab->cpldeltoffst[seg] = get_bits(gb, 5);

                    ab->cpldeltlen[seg] = get_bits(gb, 4);

                    ab->cpldeltba[seg] = get_bits(gb, 3);

                }

            }

        for (i = 0; i < nfchans; i++)

            if (ab->deltbae[i] == AC3_DBASTR_NEW) {/*channel delta offset, len and bit allocation */

                ab->deltnseg[i] = get_bits(gb, 3);

                for (seg = 0; seg <= ab->deltnseg[i]; seg++) {

                    ab->deltoffst[i][seg] = get_bits(gb, 5);

                    ab->deltlen[i][seg] = get_bits(gb, 4);

                    ab->deltba[i][seg] = get_bits(gb, 3);

                }

            }

    }

    if (do_bit_allocation (ctx, bit_alloc_flags)) /* perform the bit allocation */ {

        av_log(NULL, AV_LOG_ERROR, "Error in bit allocation routine\n");

        return -1;

    }

    if (get_bits(gb, 1)) { /* unused dummy data */

        *flags |= AC3_AB_SKIPLE;

        ab->skipl = get_bits(gb, 9);

        while (ab->skipl) {

            get_bits(gb, 8);

            ab->skipl--;

        }

    }

    /* unpack the transform coefficients

     * * this also uncouples channels if coupling is in use.

     */

    if (get_transform_coeffs(ctx)) {

        av_log(NULL, AV_LOG_ERROR, "Error in routine get_transform_coeffs\n");

        return -1;

    }

    /* recover coefficients if rematrixing is in use */

    if (*flags & AC3_AB_REMATSTR)

        do_rematrixing(ctx);



    if (ctx->output != AC3_OUTPUT_UNMODIFIED)

        do_downmix(ctx);



    return 0;

}
