int float32_eq( float32 a, float32 b STATUS_PARAM )

{



    if (    ( ( extractFloat32Exp( a ) == 0xFF ) && extractFloat32Frac( a ) )

         || ( ( extractFloat32Exp( b ) == 0xFF ) && extractFloat32Frac( b ) )

       ) {

        if ( float32_is_signaling_nan( a ) || float32_is_signaling_nan( b ) ) {

            float_raise( float_flag_invalid STATUS_VAR);

        }

        return 0;

    }

    return ( a == b ) || ( (bits32) ( ( a | b )<<1 ) == 0 );



}
