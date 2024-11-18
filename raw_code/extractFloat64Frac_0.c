INLINE bits64 extractFloat64Frac( float64 a )

{



    return a & LIT64( 0x000FFFFFFFFFFFFF );



}
