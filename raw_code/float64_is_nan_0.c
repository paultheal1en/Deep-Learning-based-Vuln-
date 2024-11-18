int float64_is_nan( float64 a1 )

{

    float64u u;

    uint64_t a;

    u.f = a1;

    a = u.i;



    return ( LIT64( 0xFFF0000000000000 ) < (bits64) ( a<<1 ) );



}
