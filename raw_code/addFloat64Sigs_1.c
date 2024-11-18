static float64 addFloat64Sigs( float64 a, float64 b, flag zSign STATUS_PARAM )

{

    int16 aExp, bExp, zExp;

    uint64_t aSig, bSig, zSig;

    int16 expDiff;



    aSig = extractFloat64Frac( a );

    aExp = extractFloat64Exp( a );

    bSig = extractFloat64Frac( b );

    bExp = extractFloat64Exp( b );

    expDiff = aExp - bExp;

    aSig <<= 9;

    bSig <<= 9;

    if ( 0 < expDiff ) {

        if ( aExp == 0x7FF ) {

            if ( aSig ) return propagateFloat64NaN( a, b STATUS_VAR );

            return a;

        }

        if ( bExp == 0 ) {

            --expDiff;

        }

        else {

            bSig |= LIT64( 0x2000000000000000 );

        }

        shift64RightJamming( bSig, expDiff, &bSig );

        zExp = aExp;

    }

    else if ( expDiff < 0 ) {

        if ( bExp == 0x7FF ) {

            if ( bSig ) return propagateFloat64NaN( a, b STATUS_VAR );

            return packFloat64( zSign, 0x7FF, 0 );

        }

        if ( aExp == 0 ) {

            ++expDiff;

        }

        else {

            aSig |= LIT64( 0x2000000000000000 );

        }

        shift64RightJamming( aSig, - expDiff, &aSig );

        zExp = bExp;

    }

    else {

        if ( aExp == 0x7FF ) {

            if ( aSig | bSig ) return propagateFloat64NaN( a, b STATUS_VAR );

            return a;

        }

        if ( aExp == 0 ) {

            if ( STATUS(flush_to_zero) ) return packFloat64( zSign, 0, 0 );

            return packFloat64( zSign, 0, ( aSig + bSig )>>9 );

        }

        zSig = LIT64( 0x4000000000000000 ) + aSig + bSig;

        zExp = aExp;

        goto roundAndPack;

    }

    aSig |= LIT64( 0x2000000000000000 );

    zSig = ( aSig + bSig )<<1;

    --zExp;

    if ( (int64_t) zSig < 0 ) {

        zSig = aSig + bSig;

        ++zExp;

    }

 roundAndPack:

    return roundAndPackFloat64( zSign, zExp, zSig STATUS_VAR );



}
