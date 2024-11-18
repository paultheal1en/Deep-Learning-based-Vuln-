void nand_setpins(DeviceState *dev, uint8_t cle, uint8_t ale,

                  uint8_t ce, uint8_t wp, uint8_t gnd)

{

    NANDFlashState *s = (NANDFlashState *) dev;

    s->cle = cle;

    s->ale = ale;

    s->ce = ce;

    s->wp = wp;

    s->gnd = gnd;

    if (wp)

        s->status |= NAND_IOSTATUS_UNPROTCT;

    else

        s->status &= ~NAND_IOSTATUS_UNPROTCT;

}
