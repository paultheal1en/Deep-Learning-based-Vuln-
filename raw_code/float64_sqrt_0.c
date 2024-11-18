float64 float64_sqrt( float64 a STATUS_PARAM )

{

    flag aSign;

    int16 aExp, zExp;

    bits64 aSig, zSig, doubleZSig;

    bits64 rem0, rem1, term0, term1;



    aSig = extractFloat64Frac( a );

    aExp = extractFloat64Exp( a );

    aSign = extractFloat64Sign( a );

    if ( aExp == 0x7FF ) {

        if ( aSig ) return propagateFloat64NaN( a, a STATUS_VAR );

        if ( ! aSign ) return a;

        float_raise( float_flag_invalid STATUS_VAR);

        return float64_default_nan;

    }

    if ( aSign ) {

        if ( ( aExp | aSig ) == 0 ) return a;

        float_raise( float_flag_invalid STATUS_VAR);

        return float64_default_nan;

    }

    if ( aExp == 0 ) {

        if ( aSig == 0 ) return 0;

        normalizeFloat64Subnormal( aSig, &aExp, &aSig );

    }

    zExp = ( ( aExp - 0x3FF )>>1 ) + 0x3FE;

    aSig |= LIT64( 0x0010000000000000 );

    zSig = estimateSqrt32( aExp, aSig>>21 );

    aSig <<= 9 - ( aExp & 1 );

    zSig = estimateDiv128To64( aSig, 0, zSig<<32 ) + ( zSig<<30 );

    if ( ( zSig & 0x1FF ) <= 5 ) {

        doubleZSig = zSig<<1;

        mul64To128( zSig, zSig, &term0, &term1 );

        sub128( aSig, 0, term0, term1, &rem0, &rem1 );

        while ( (sbits64) rem0 < 0 ) {

            --zSig;

            doubleZSig -= 2;

            add128( rem0, rem1, zSig>>63, doubleZSig | 1, &rem0, &rem1 );

        }

        zSig |= ( ( rem0 | rem1 ) != 0 );

    }

    return roundAndPackFloat64( 0, zExp, zSig STATUS_VAR );



}
