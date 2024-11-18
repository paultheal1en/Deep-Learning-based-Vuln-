 roundAndPackFloatx80(

     int8 roundingPrecision, flag zSign, int32 zExp, uint64_t zSig0, uint64_t zSig1

 STATUS_PARAM)

{

    int8 roundingMode;

    flag roundNearestEven, increment, isTiny;

    int64 roundIncrement, roundMask, roundBits;



    roundingMode = STATUS(float_rounding_mode);

    roundNearestEven = ( roundingMode == float_round_nearest_even );

    if ( roundingPrecision == 80 ) goto precision80;

    if ( roundingPrecision == 64 ) {

        roundIncrement = LIT64( 0x0000000000000400 );

        roundMask = LIT64( 0x00000000000007FF );

    }

    else if ( roundingPrecision == 32 ) {

        roundIncrement = LIT64( 0x0000008000000000 );

        roundMask = LIT64( 0x000000FFFFFFFFFF );

    }

    else {

        goto precision80;

    }

    zSig0 |= ( zSig1 != 0 );

    if ( ! roundNearestEven ) {

        if ( roundingMode == float_round_to_zero ) {

            roundIncrement = 0;

        }

        else {

            roundIncrement = roundMask;

            if ( zSign ) {

                if ( roundingMode == float_round_up ) roundIncrement = 0;

            }

            else {

                if ( roundingMode == float_round_down ) roundIncrement = 0;

            }

        }

    }

    roundBits = zSig0 & roundMask;

    if ( 0x7FFD <= (uint32_t) ( zExp - 1 ) ) {

        if (    ( 0x7FFE < zExp )

             || ( ( zExp == 0x7FFE ) && ( zSig0 + roundIncrement < zSig0 ) )

           ) {

            goto overflow;

        }

        if ( zExp <= 0 ) {

            if ( STATUS(flush_to_zero) ) return packFloatx80( zSign, 0, 0 );

            isTiny =

                   ( STATUS(float_detect_tininess) == float_tininess_before_rounding )

                || ( zExp < 0 )

                || ( zSig0 <= zSig0 + roundIncrement );

            shift64RightJamming( zSig0, 1 - zExp, &zSig0 );

            zExp = 0;

            roundBits = zSig0 & roundMask;

            if ( isTiny && roundBits ) float_raise( float_flag_underflow STATUS_VAR);

            if ( roundBits ) STATUS(float_exception_flags) |= float_flag_inexact;

            zSig0 += roundIncrement;

            if ( (int64_t) zSig0 < 0 ) zExp = 1;

            roundIncrement = roundMask + 1;

            if ( roundNearestEven && ( roundBits<<1 == roundIncrement ) ) {

                roundMask |= roundIncrement;

            }

            zSig0 &= ~ roundMask;

            return packFloatx80( zSign, zExp, zSig0 );

        }

    }

    if ( roundBits ) STATUS(float_exception_flags) |= float_flag_inexact;

    zSig0 += roundIncrement;

    if ( zSig0 < roundIncrement ) {

        ++zExp;

        zSig0 = LIT64( 0x8000000000000000 );

    }

    roundIncrement = roundMask + 1;

    if ( roundNearestEven && ( roundBits<<1 == roundIncrement ) ) {

        roundMask |= roundIncrement;

    }

    zSig0 &= ~ roundMask;

    if ( zSig0 == 0 ) zExp = 0;

    return packFloatx80( zSign, zExp, zSig0 );

 precision80:

    increment = ( (int64_t) zSig1 < 0 );

    if ( ! roundNearestEven ) {

        if ( roundingMode == float_round_to_zero ) {

            increment = 0;

        }

        else {

            if ( zSign ) {

                increment = ( roundingMode == float_round_down ) && zSig1;

            }

            else {

                increment = ( roundingMode == float_round_up ) && zSig1;

            }

        }

    }

    if ( 0x7FFD <= (uint32_t) ( zExp - 1 ) ) {

        if (    ( 0x7FFE < zExp )

             || (    ( zExp == 0x7FFE )

                  && ( zSig0 == LIT64( 0xFFFFFFFFFFFFFFFF ) )

                  && increment

                )

           ) {

            roundMask = 0;

 overflow:

            float_raise( float_flag_overflow | float_flag_inexact STATUS_VAR);

            if (    ( roundingMode == float_round_to_zero )

                 || ( zSign && ( roundingMode == float_round_up ) )

                 || ( ! zSign && ( roundingMode == float_round_down ) )

               ) {

                return packFloatx80( zSign, 0x7FFE, ~ roundMask );

            }

            return packFloatx80( zSign, 0x7FFF, LIT64( 0x8000000000000000 ) );

        }

        if ( zExp <= 0 ) {

            isTiny =

                   ( STATUS(float_detect_tininess) == float_tininess_before_rounding )

                || ( zExp < 0 )

                || ! increment

                || ( zSig0 < LIT64( 0xFFFFFFFFFFFFFFFF ) );

            shift64ExtraRightJamming( zSig0, zSig1, 1 - zExp, &zSig0, &zSig1 );

            zExp = 0;

            if ( isTiny && zSig1 ) float_raise( float_flag_underflow STATUS_VAR);

            if ( zSig1 ) STATUS(float_exception_flags) |= float_flag_inexact;

            if ( roundNearestEven ) {

                increment = ( (int64_t) zSig1 < 0 );

            }

            else {

                if ( zSign ) {

                    increment = ( roundingMode == float_round_down ) && zSig1;

                }

                else {

                    increment = ( roundingMode == float_round_up ) && zSig1;

                }

            }

            if ( increment ) {

                ++zSig0;

                zSig0 &=

                    ~ ( ( (uint64_t) ( zSig1<<1 ) == 0 ) & roundNearestEven );

                if ( (int64_t) zSig0 < 0 ) zExp = 1;

            }

            return packFloatx80( zSign, zExp, zSig0 );

        }

    }

    if ( zSig1 ) STATUS(float_exception_flags) |= float_flag_inexact;

    if ( increment ) {

        ++zSig0;

        if ( zSig0 == 0 ) {

            ++zExp;

            zSig0 = LIT64( 0x8000000000000000 );

        }

        else {

            zSig0 &= ~ ( ( (uint64_t) ( zSig1<<1 ) == 0 ) & roundNearestEven );

        }

    }

    else {

        if ( zSig0 == 0 ) zExp = 0;

    }

    return packFloatx80( zSign, zExp, zSig0 );



}
