float32 int32_to_float32( int32 a STATUS_PARAM )

{

    flag zSign;



    if ( a == 0 ) return 0;

    if ( a == (sbits32) 0x80000000 ) return packFloat32( 1, 0x9E, 0 );

    zSign = ( a < 0 );

    return normalizeRoundAndPackFloat32( zSign, 0x9C, zSign ? - a : a STATUS_VAR );



}
