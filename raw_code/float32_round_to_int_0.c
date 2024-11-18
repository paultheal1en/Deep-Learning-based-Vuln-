float32 float32_round_to_int( float32 a STATUS_PARAM)

{

    flag aSign;

    int16 aExp;

    bits32 lastBitMask, roundBitsMask;

    int8 roundingMode;

    float32 z;



    aExp = extractFloat32Exp( a );

    if ( 0x96 <= aExp ) {

        if ( ( aExp == 0xFF ) && extractFloat32Frac( a ) ) {

            return propagateFloat32NaN( a, a STATUS_VAR );

        }

        return a;

    }

    if ( aExp <= 0x7E ) {

        if ( (bits32) ( a<<1 ) == 0 ) return a;

        STATUS(float_exception_flags) |= float_flag_inexact;

        aSign = extractFloat32Sign( a );

        switch ( STATUS(float_rounding_mode) ) {

         case float_round_nearest_even:

            if ( ( aExp == 0x7E ) && extractFloat32Frac( a ) ) {

                return packFloat32( aSign, 0x7F, 0 );

            }

            break;

         case float_round_down:

            return aSign ? 0xBF800000 : 0;

         case float_round_up:

            return aSign ? 0x80000000 : 0x3F800000;

        }

        return packFloat32( aSign, 0, 0 );

    }

    lastBitMask = 1;

    lastBitMask <<= 0x96 - aExp;

    roundBitsMask = lastBitMask - 1;

    z = a;

    roundingMode = STATUS(float_rounding_mode);

    if ( roundingMode == float_round_nearest_even ) {

        z += lastBitMask>>1;

        if ( ( z & roundBitsMask ) == 0 ) z &= ~ lastBitMask;

    }

    else if ( roundingMode != float_round_to_zero ) {

        if ( extractFloat32Sign( z ) ^ ( roundingMode == float_round_up ) ) {

            z += roundBitsMask;

        }

    }

    z &= ~ roundBitsMask;

    if ( z != a ) STATUS(float_exception_flags) |= float_flag_inexact;

    return z;



}
