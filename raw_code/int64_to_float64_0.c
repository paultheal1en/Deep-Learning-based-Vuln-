float64 int64_to_float64( int64 a STATUS_PARAM )

{

    flag zSign;



    if ( a == 0 ) return 0;

    if ( a == (sbits64) LIT64( 0x8000000000000000 ) ) {

        return packFloat64( 1, 0x43E, 0 );

    }

    zSign = ( a < 0 );

    return normalizeRoundAndPackFloat64( zSign, 0x43C, zSign ? - a : a STATUS_VAR );



}
