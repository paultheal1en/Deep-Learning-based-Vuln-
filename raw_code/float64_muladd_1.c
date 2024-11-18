float64 float64_muladd(float64 a, float64 b, float64 c, int flags STATUS_PARAM)

{

    flag aSign, bSign, cSign, zSign;

    int_fast16_t aExp, bExp, cExp, pExp, zExp, expDiff;

    uint64_t aSig, bSig, cSig;

    flag pInf, pZero, pSign;

    uint64_t pSig0, pSig1, cSig0, cSig1, zSig0, zSig1;

    int shiftcount;

    flag signflip, infzero;



    a = float64_squash_input_denormal(a STATUS_VAR);

    b = float64_squash_input_denormal(b STATUS_VAR);

    c = float64_squash_input_denormal(c STATUS_VAR);

    aSig = extractFloat64Frac(a);

    aExp = extractFloat64Exp(a);

    aSign = extractFloat64Sign(a);

    bSig = extractFloat64Frac(b);

    bExp = extractFloat64Exp(b);

    bSign = extractFloat64Sign(b);

    cSig = extractFloat64Frac(c);

    cExp = extractFloat64Exp(c);

    cSign = extractFloat64Sign(c);



    infzero = ((aExp == 0 && aSig == 0 && bExp == 0x7ff && bSig == 0) ||

               (aExp == 0x7ff && aSig == 0 && bExp == 0 && bSig == 0));



    /* It is implementation-defined whether the cases of (0,inf,qnan)

     * and (inf,0,qnan) raise InvalidOperation or not (and what QNaN

     * they return if they do), so we have to hand this information

     * off to the target-specific pick-a-NaN routine.

     */

    if (((aExp == 0x7ff) && aSig) ||

        ((bExp == 0x7ff) && bSig) ||

        ((cExp == 0x7ff) && cSig)) {

        return propagateFloat64MulAddNaN(a, b, c, infzero STATUS_VAR);

    }



    if (infzero) {

        float_raise(float_flag_invalid STATUS_VAR);

        return float64_default_nan;

    }



    if (flags & float_muladd_negate_c) {

        cSign ^= 1;

    }



    signflip = (flags & float_muladd_negate_result) ? 1 : 0;



    /* Work out the sign and type of the product */

    pSign = aSign ^ bSign;

    if (flags & float_muladd_negate_product) {

        pSign ^= 1;

    }

    pInf = (aExp == 0x7ff) || (bExp == 0x7ff);

    pZero = ((aExp | aSig) == 0) || ((bExp | bSig) == 0);



    if (cExp == 0x7ff) {

        if (pInf && (pSign ^ cSign)) {

            /* addition of opposite-signed infinities => InvalidOperation */

            float_raise(float_flag_invalid STATUS_VAR);

            return float64_default_nan;

        }

        /* Otherwise generate an infinity of the same sign */

        return packFloat64(cSign ^ signflip, 0x7ff, 0);

    }



    if (pInf) {

        return packFloat64(pSign ^ signflip, 0x7ff, 0);

    }



    if (pZero) {

        if (cExp == 0) {

            if (cSig == 0) {

                /* Adding two exact zeroes */

                if (pSign == cSign) {

                    zSign = pSign;

                } else if (STATUS(float_rounding_mode) == float_round_down) {

                    zSign = 1;

                } else {

                    zSign = 0;

                }

                return packFloat64(zSign ^ signflip, 0, 0);

            }

            /* Exact zero plus a denorm */

            if (STATUS(flush_to_zero)) {

                float_raise(float_flag_output_denormal STATUS_VAR);

                return packFloat64(cSign ^ signflip, 0, 0);

            }

        }

        /* Zero plus something non-zero : just return the something */

        return packFloat64(cSign ^ signflip, cExp, cSig);

    }



    if (aExp == 0) {

        normalizeFloat64Subnormal(aSig, &aExp, &aSig);

    }

    if (bExp == 0) {

        normalizeFloat64Subnormal(bSig, &bExp, &bSig);

    }



    /* Calculate the actual result a * b + c */



    /* Multiply first; this is easy. */

    /* NB: we subtract 0x3fe where float64_mul() subtracts 0x3ff

     * because we want the true exponent, not the "one-less-than"

     * flavour that roundAndPackFloat64() takes.

     */

    pExp = aExp + bExp - 0x3fe;

    aSig = (aSig | LIT64(0x0010000000000000))<<10;

    bSig = (bSig | LIT64(0x0010000000000000))<<11;

    mul64To128(aSig, bSig, &pSig0, &pSig1);

    if ((int64_t)(pSig0 << 1) >= 0) {

        shortShift128Left(pSig0, pSig1, 1, &pSig0, &pSig1);

        pExp--;

    }



    zSign = pSign ^ signflip;



    /* Now [pSig0:pSig1] is the significand of the multiply, with the explicit

     * bit in position 126.

     */

    if (cExp == 0) {

        if (!cSig) {

            /* Throw out the special case of c being an exact zero now */

            shift128RightJamming(pSig0, pSig1, 64, &pSig0, &pSig1);

            return roundAndPackFloat64(zSign, pExp - 1,

                                       pSig1 STATUS_VAR);

        }

        normalizeFloat64Subnormal(cSig, &cExp, &cSig);

    }



    /* Shift cSig and add the explicit bit so [cSig0:cSig1] is the

     * significand of the addend, with the explicit bit in position 126.

     */

    cSig0 = cSig << (126 - 64 - 52);

    cSig1 = 0;

    cSig0 |= LIT64(0x4000000000000000);

    expDiff = pExp - cExp;



    if (pSign == cSign) {

        /* Addition */

        if (expDiff > 0) {

            /* scale c to match p */

            shift128RightJamming(cSig0, cSig1, expDiff, &cSig0, &cSig1);

            zExp = pExp;

        } else if (expDiff < 0) {

            /* scale p to match c */

            shift128RightJamming(pSig0, pSig1, -expDiff, &pSig0, &pSig1);

            zExp = cExp;

        } else {

            /* no scaling needed */

            zExp = cExp;

        }

        /* Add significands and make sure explicit bit ends up in posn 126 */

        add128(pSig0, pSig1, cSig0, cSig1, &zSig0, &zSig1);

        if ((int64_t)zSig0 < 0) {

            shift128RightJamming(zSig0, zSig1, 1, &zSig0, &zSig1);

        } else {

            zExp--;

        }

        shift128RightJamming(zSig0, zSig1, 64, &zSig0, &zSig1);

        return roundAndPackFloat64(zSign, zExp, zSig1 STATUS_VAR);

    } else {

        /* Subtraction */

        if (expDiff > 0) {

            shift128RightJamming(cSig0, cSig1, expDiff, &cSig0, &cSig1);

            sub128(pSig0, pSig1, cSig0, cSig1, &zSig0, &zSig1);

            zExp = pExp;

        } else if (expDiff < 0) {

            shift128RightJamming(pSig0, pSig1, -expDiff, &pSig0, &pSig1);

            sub128(cSig0, cSig1, pSig0, pSig1, &zSig0, &zSig1);

            zExp = cExp;

            zSign ^= 1;

        } else {

            zExp = pExp;

            if (lt128(cSig0, cSig1, pSig0, pSig1)) {

                sub128(pSig0, pSig1, cSig0, cSig1, &zSig0, &zSig1);

            } else if (lt128(pSig0, pSig1, cSig0, cSig1)) {

                sub128(cSig0, cSig1, pSig0, pSig1, &zSig0, &zSig1);

                zSign ^= 1;

            } else {

                /* Exact zero */

                zSign = signflip;

                if (STATUS(float_rounding_mode) == float_round_down) {

                    zSign ^= 1;

                }

                return packFloat64(zSign, 0, 0);

            }

        }

        --zExp;

        /* Do the equivalent of normalizeRoundAndPackFloat64() but

         * starting with the significand in a pair of uint64_t.

         */

        if (zSig0) {

            shiftcount = countLeadingZeros64(zSig0) - 1;

            shortShift128Left(zSig0, zSig1, shiftcount, &zSig0, &zSig1);

            if (zSig1) {

                zSig0 |= 1;

            }

            zExp -= shiftcount;

        } else {

            shiftcount = countLeadingZeros64(zSig1) - 1;

            zSig0 = zSig1 << shiftcount;

            zExp -= (shiftcount + 64);

        }

        return roundAndPackFloat64(zSign, zExp, zSig0 STATUS_VAR);

    }

}
