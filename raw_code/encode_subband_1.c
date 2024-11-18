static void encode_subband(SnowContext *s, SubBand *b, DWTELEM *src, DWTELEM *parent, int stride, int orientation){    

//    encode_subband_qtree(s, b, src, parent, stride, orientation);

//    encode_subband_z0run(s, b, src, parent, stride, orientation);

    encode_subband_c0run(s, b, src, parent, stride, orientation);

//    encode_subband_dzr(s, b, src, parent, stride, orientation);

}
