float64 uint64_to_float64( uint64 a STATUS_PARAM )

{

    if ( a == 0 ) return 0;

    return normalizeRoundAndPackFloat64( 0, 0x43C, a STATUS_VAR );



}
