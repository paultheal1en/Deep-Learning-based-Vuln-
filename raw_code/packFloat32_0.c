INLINE float32 packFloat32( flag zSign, int16 zExp, bits32 zSig )

{



    return ( ( (bits32) zSign )<<31 ) + ( ( (bits32) zExp )<<23 ) + zSig;



}
