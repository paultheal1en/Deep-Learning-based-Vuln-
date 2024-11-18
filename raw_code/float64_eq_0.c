int float64_eq( float64 a, float64 b STATUS_PARAM )

{



    if (    ( ( extractFloat64Exp( a ) == 0x7FF ) && extractFloat64Frac( a ) )

         || ( ( extractFloat64Exp( b ) == 0x7FF ) && extractFloat64Frac( b ) )

       ) {

        if ( float64_is_signaling_nan( a ) || float64_is_signaling_nan( b ) ) {

            float_raise( float_flag_invalid STATUS_VAR);

        }

        return 0;

    }

    return ( a == b ) || ( (bits64) ( ( a | b )<<1 ) == 0 );



}
