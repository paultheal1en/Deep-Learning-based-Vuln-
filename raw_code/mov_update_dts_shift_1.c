static void mov_update_dts_shift(MOVStreamContext *sc, int duration)

{

    if (duration < 0) {





        sc->dts_shift = FFMAX(sc->dts_shift, -duration);

