int32 float32_to_int32_round_to_zero( float32 a STATUS_PARAM )

{

    flag aSign;

    int16 aExp, shiftCount;

    bits32 aSig;

    int32 z;



    aSig = extractFloat32Frac( a );

    aExp = extractFloat32Exp( a );

    aSign = extractFloat32Sign( a );

    shiftCount = aExp - 0x9E;

    if ( 0 <= shiftCount ) {

        if ( a != 0xCF000000 ) {

            float_raise( float_flag_invalid STATUS_VAR);

            if ( ! aSign || ( ( aExp == 0xFF ) && aSig ) ) return 0x7FFFFFFF;

        }

        return (sbits32) 0x80000000;

    }

    else if ( aExp <= 0x7E ) {

        if ( aExp | aSig ) STATUS(float_exception_flags) |= float_flag_inexact;

        return 0;

    }

    aSig = ( aSig | 0x00800000 )<<8;

    z = aSig>>( - shiftCount );

    if ( (bits32) ( aSig<<( shiftCount & 31 ) ) ) {

        STATUS(float_exception_flags) |= float_flag_inexact;

    }

    if ( aSign ) z = - z;

    return z;



}
