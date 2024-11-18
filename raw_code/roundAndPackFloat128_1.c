 roundAndPackFloat128(

     flag zSign, int32 zExp, uint64_t zSig0, uint64_t zSig1, uint64_t zSig2 STATUS_PARAM)

{

    int8 roundingMode;

    flag roundNearestEven, increment, isTiny;



    roundingMode = STATUS(float_rounding_mode);

    roundNearestEven = ( roundingMode == float_round_nearest_even );

    increment = ( (int64_t) zSig2 < 0 );

    if ( ! roundNearestEven ) {

        if ( roundingMode == float_round_to_zero ) {

            increment = 0;

        }

        else {

            if ( zSign ) {

                increment = ( roundingMode == float_round_down ) && zSig2;

            }

            else {

                increment = ( roundingMode == float_round_up ) && zSig2;

            }

        }

    }

    if ( 0x7FFD <= (uint32_t) zExp ) {

        if (    ( 0x7FFD < zExp )

             || (    ( zExp == 0x7FFD )

                  && eq128(

                         LIT64( 0x0001FFFFFFFFFFFF ),

                         LIT64( 0xFFFFFFFFFFFFFFFF ),

                         zSig0,

                         zSig1

                     )

                  && increment

                )

           ) {

            float_raise( float_flag_overflow | float_flag_inexact STATUS_VAR);

            if (    ( roundingMode == float_round_to_zero )

                 || ( zSign && ( roundingMode == float_round_up ) )

                 || ( ! zSign && ( roundingMode == float_round_down ) )

               ) {

                return

                    packFloat128(

                        zSign,

                        0x7FFE,

                        LIT64( 0x0000FFFFFFFFFFFF ),

                        LIT64( 0xFFFFFFFFFFFFFFFF )

                    );

            }

            return packFloat128( zSign, 0x7FFF, 0, 0 );

        }

        if ( zExp < 0 ) {

            if ( STATUS(flush_to_zero) ) return packFloat128( zSign, 0, 0, 0 );

            isTiny =

                   ( STATUS(float_detect_tininess) == float_tininess_before_rounding )

                || ( zExp < -1 )

                || ! increment

                || lt128(

                       zSig0,

                       zSig1,

                       LIT64( 0x0001FFFFFFFFFFFF ),

                       LIT64( 0xFFFFFFFFFFFFFFFF )

                   );

            shift128ExtraRightJamming(

                zSig0, zSig1, zSig2, - zExp, &zSig0, &zSig1, &zSig2 );

            zExp = 0;

            if ( isTiny && zSig2 ) float_raise( float_flag_underflow STATUS_VAR);

            if ( roundNearestEven ) {

                increment = ( (int64_t) zSig2 < 0 );

            }

            else {

                if ( zSign ) {

                    increment = ( roundingMode == float_round_down ) && zSig2;

                }

                else {

                    increment = ( roundingMode == float_round_up ) && zSig2;

                }

            }

        }

    }

    if ( zSig2 ) STATUS(float_exception_flags) |= float_flag_inexact;

    if ( increment ) {

        add128( zSig0, zSig1, 0, 1, &zSig0, &zSig1 );

        zSig1 &= ~ ( ( zSig2 + zSig2 == 0 ) & roundNearestEven );

    }

    else {

        if ( ( zSig0 | zSig1 ) == 0 ) zExp = 0;

    }

    return packFloat128( zSign, zExp, zSig0, zSig1 );



}
