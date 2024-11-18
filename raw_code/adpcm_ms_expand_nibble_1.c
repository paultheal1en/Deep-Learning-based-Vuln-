static inline short adpcm_ms_expand_nibble(ADPCMChannelStatus *c, int nibble)

{

    int predictor;



    predictor = (((c->sample1) * (c->coeff1)) + ((c->sample2) * (c->coeff2))) / 64;

    predictor += ((nibble & 0x08)?(nibble - 0x10):(nibble)) * c->idelta;



    c->sample2 = c->sample1;

    c->sample1 = av_clip_int16(predictor);

    c->idelta = (ff_adpcm_AdaptationTable[(int)nibble] * c->idelta) >> 8;

    if (c->idelta < 16) c->idelta = 16;







    return c->sample1;
