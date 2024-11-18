static int dscm1xxxx_attach(PCMCIACardState *card)

{

    MicroDriveState *md = MICRODRIVE(card);

    PCMCIACardClass *pcc = PCMCIA_CARD_GET_CLASS(card);



    md->attr_base = pcc->cis[0x74] | (pcc->cis[0x76] << 8);

    md->io_base = 0x0;



    device_reset(DEVICE(md));

    md_interrupt_update(md);



    card->slot->card_string = "DSCM-1xxxx Hitachi Microdrive";

    return 0;

}
